import torch
import numpy as np
import numpy.random as rd
from AgentBase import AgentBase
from Actor import Actor
from Critic import Critic
import math


class AgentDDPG(AgentBase):
    """
    Bases: ``AgentBase``

    Deep Deterministic Policy Gradient algorithm. “Continuous control with deep reinforcement learning”. T. Lillicrap et al.. 2015.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(self):
        AgentBase.__init__(self)
        self.ClassAct = Actor
        self.ClassCri = Critic
        self.if_use_cri_target = True
        self.if_use_act_target = True

        self.explore_noise = 0.3  # explore noise of action (OrnsteinUhlenbeckNoise)
        self.ou_noise = None

    def init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        """
        Explict call ``self.init()`` to overwrite the ``self.object`` in ``__init__()`` for multiprocessing.
        """
        AgentBase.init(self, net_dim=net_dim, state_dim=state_dim, action_dim=action_dim,
                       reward_scale=reward_scale, gamma=gamma,
                       learning_rate=learning_rate, if_per_or_gae=if_per_or_gae,
                       env_num=env_num, gpu_id=gpu_id, )
        self.ou_noise = OrnsteinUhlenbeckNoise(size=action_dim, sigma=self.explore_noise)
        self.last_action = torch.zeros([1, 1, 1, 29])
        # print("if_per_or_gae:",if_per_or_gae)
        # if_per_or_gae = False
        if if_per_or_gae:
            # self.criterion = torch.nn.MSELoss(reduction='none' if if_per_or_gae else 'mean')
            # self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per_or_gae else 'mean', beta = 0.03)
            self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per_or_gae else 'mean')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            # print("if_per_or_g ae:",if_per_or_gae)
            # self.criterion = torch.nn.MSELoss(reduction='none' if if_per_or_gae else 'mean')
            self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per_or_gae else 'mean')
            self.criterion_of_Q_prediction = torch.nn.MSELoss(reduction= 'mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, states: torch.Tensor) -> torch.Tensor:
        """
        Select actions given an array of states.

        .. note::
            Using ϵ-greedy with Ornstein–Uhlenbeck noise to add noise to actions for randomness.

        :param states: an array of states in a shape (batch_size, state_dim, ).
        :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        # print("==================1========================")
        action = self.act(states.to(self.device), self.last_action.to(self.device))
        self.last_action = action[:, 1:].unsqueeze(0).unsqueeze(0)
        # print("action", self.last_action.size())
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            ou_noise = torch.as_tensor(self.ou_noise(), dtype=torch.float32, device=self.device).unsqueeze(0)
            action = (action + ou_noise).clamp(-1, 1)
        return action.detach().cpu()

    def calculation_of_Q_real(self, state):
        state = state.permute(0, 3, 1, 2)
        state = state[:, 0:1, :, :]
        future_return_tensor = torch.log2(state[:, :, -5:, :])
        Q_real = torch.mean(future_return_tensor, axis=2) * 100
        return Q_real.squeeze(1)

    def deep_learning_loss_function(self, z_post, z_prior, y, y_rec, gamma=10):
        mu, sigma = y_rec
        f = lambda x: torch.exp(-torch.square(x - mu) / (torch.as_tensor(2) * torch.square(sigma))) / (
                    sigma * torch.sqrt(torch.as_tensor(2) * math.pi))
        term1 = -torch.mean(torch.log(f(y)))
        term2 = self.compute_kl(z_post[0], z_post[1], z_prior[0], z_prior[1])
        return term1 + gamma * term2

    def compute_kl(self, u1, sigma1, u2, sigma2):
        """计算两个多元高斯分布之间KL散度KL(N1||N2)；
        所有的shape均为(B1,B2,...,dim)，表示协方差为0的多元高斯分布
        这里我们假设加上Batch_size，即形状为(B,dim)
        dim:特征的维度
        """
        sigma1_matrix = torch.diag_embed(sigma1)  # (B,dim,dim)
        sigma1_matrix_det = torch.det(sigma1_matrix)  # (B,)

        sigma2_matrix = torch.diag_embed(sigma2)  # (B,dim,dim)
        sigma2_matrix_det = torch.det(sigma2_matrix)  # (B,)
        sigma2_matrix_inv = torch.diag_embed(1. / sigma2)  # (B,dim,dim)

        delta_u = (u2 - u1).unsqueeze(-1)  # (B,dim,1)
        delta_u_transpose = delta_u.transpose(-1, -2)  # (B,1,dim)

        term1 = torch.sum((1. / sigma2) * sigma1, dim=-1)  # (B,) represent trace term
        term2 = torch.matmul(torch.matmul(delta_u_transpose, sigma2_matrix_inv), delta_u)  # (B,)
        term3 = -u1.shape[-1]
        term4 = torch.log(sigma2_matrix_det) - torch.log(sigma1_matrix_det)

        KL = 0.5 * (term1 + term2 + term3 + term4)

        # if you want to compute the mean of a batch,then,
        KL_mean = torch.mean(KL)

        return KL_mean

    def future_period_return_calculation(self, state):
        state = state.permute(0, 3, 1, 2)
        state = state[:, 0, -5:, :]
        future_period_return = torch.mean(torch.log2(state), dim = 1) * 100
        return future_period_return

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        for _ in range(int(repeat_times)):
            # print("int(buffer.now_len / batch_size * repeat_times", int(buffer.now_len / batch_size * repeat_times))
            state, state_EIIE, last_action, future_return_based_on_closing, obj_critic, indices = self.get_obj_critic(buffer, batch_size)
            last_action_actor = last_action[:, 1:].unsqueeze(1).unsqueeze(1)
            last_action_critic = last_action
            action_pg = self.act(state_EIIE, last_action_actor)  # policy gradient
            obj_actor = -self.cri(future_return_based_on_closing, action_pg, last_action_critic).mean()
            self.optim_update(self.act_optim, obj_actor)
            action_update = self.act(state_EIIE, last_action_actor)
            # buffer.update_action_buffer(action_update, indices)
        return (obj_critic.item(), obj_actor.item()), action_update, indices

    def get_obj_critic_raw(self, buffer, batch_size):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, last_action, action_update, state_EIIE, future_return_based_on_closing, next_s, indices = buffer.sample_batch(batch_size)
            q_label = reward
        q_value = self.cri(future_return_based_on_closing, action_update, last_action)
        # obj_critic = self.criterion(q_value, q_label)
        return state, state_EIIE, last_action, future_return_based_on_closing, q_value.mean(), indices

    def get_obj_critic_per(self, buffer, batch_size):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q

        q_value = self.cri(state, action)
        td_error = self.criterion(q_value, q_label)  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class OrnsteinUhlenbeckNoise:  # NOT suggest to use it
    def __init__(self, size, theta=0.15, sigma=0.3, ou_noise=0.0, dt=1e-2):
        """
        The noise of Ornstein-Uhlenbeck Process

        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        Don't abuse OU Process. OU process has too much hyper-parameters and over fine-tuning make no sense.

        :int size: the size of noise, noise.shape==(-1, action_dim)
        :float theta: related to the not independent of OU-noise
        :float sigma: related to action noise std
        :float ou_noise: initialize OU-noise
        :float dt: derivative
        """
        self.theta = theta
        self.sigma = sigma
        self.ou_noise = ou_noise
        self.dt = dt
        self.size = size

    def __call__(self) -> float:
        """
        output a OU-noise

        :return array ou_noise: a noise generated by Ornstein-Uhlenbeck Process
        """
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        self.ou_noise -= self.theta * self.ou_noise * self.dt + noise
        return self.ou_noise