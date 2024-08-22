import numpy as np
import torch
import numpy.random as rd
from AgentBase_SAC import AgentBase_SAC
from Net import ActorSAC, CriticTwin, ShareSPG, CriticMultiple

class AgentSAC(AgentBase_SAC):
    def __init__(self):
        AgentBase_SAC.__init__(self)
        self.ClassCri = CriticTwin
        self.ClassAct = ActorSAC
        self.if_use_cri_target = True
        self.if_use_act_target = False

        self.alpha_log = None
        self.alpha_optim = None
        self.target_entropy = None
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        """
        Explict call ``self.init()`` to overwrite the ``self.object`` in ``__init__()`` for multiprocessing.
        """
        AgentBase_SAC.init(self, net_dim=net_dim, state_dim=state_dim, action_dim=action_dim,
                       reward_scale=reward_scale, gamma=gamma,
                       learning_rate=learning_rate, if_per_or_gae=if_per_or_gae,
                       env_num=env_num, gpu_id=gpu_id, )

        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=learning_rate)
        self.target_entropy = np.log(action_dim)

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select actions given an array of states.

        :param state: an array of states in a shape (batch_size, state_dim, ).
        :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        state = state.to(self.device)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            actions = self.act.get_action(state)
        else:
            actions = self.act(state)
        return actions.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
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
        alpha = None
        for _ in range(int(buffer.now_len * repeat_times / batch_size)):
            alpha = self.alpha_log.exp()

            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.optim_update(self.cri_optim, obj_critic)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.optim_update(self.alpha_optim, obj_alpha)

            '''objective of actor'''
            obj_actor = -(self.cri(state, action_pg) + logprob * alpha).mean()
            self.optim_update(self.act_optim, obj_actor)
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item(), obj_actor.item(), alpha.item()

    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state

    # def __init__(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
    #              learning_rate=1e-4, alpha=0.2, auto_entropy=True, target_entropy=None, **kwargs):
    #     super().__init__(net_dim, state_dim, action_dim, reward_scale, gamma, learning_rate, **kwargs)
    #
    #     # SAC-specific hyperparameters
    #     self.alpha = alpha
    #     self.auto_entropy = auto_entropy
    #     self.target_entropy = -action_dim if target_entropy is None else target_entropy
    #
    #     self.cri_optim = Adam(self.cri.parameters(), lr=learning_rate)
    #     self.act_optim = Adam(self.act.parameters(), lr=learning_rate)
    #     self.criterion = nn.MSELoss()
    #
    #     self.target_entropy = -np.prod(self.action_dim) if target_entropy is None else target_entropy
    #     self.alpha_log = torch.tensor(np.log(alpha), requires_grad=True, device=self.device, dtype=torch.float32)
    #
    #     # create target entropy tensor for entropy target regularization
    #     self.target_entropy = torch.tensor(self.target_entropy, device=self.device, dtype=torch.float32)

    # def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
    #     buffer.update_now_len()
    #
    #     if buffer.now_len < batch_size:
    #         return
    #
    #     update_a = 0
    #     if self.auto_entropy is True:
    #         self.alpha = np.exp(self.alpha_log.item())
    #         self.alpha_log.data += self.alpha_lr * (self.target_entropy - self.alpha)
    #         self.alpha = self.alpha.clip(1e-8, 2 ** 16)
    #         self.alpha_tensor = torch.tensor(self.alpha, device=self.device, requires_grad=False)
    #
    #     with torch.no_grad():
    #         reward, mask, action, state, next_s = buffer.sample_all()
    #
    #         next_q_label, next_log_prob = self.cri_target.get__q1_q2_log_prob(next_s, self.act_target(next_s))
    #         next_q_label = (next_q_label - self.alpha_tensor * next_log_prob).mean(dim=0)
    #         q_label = reward + mask * self.gamma * next_q_label
    #
    #     """Critic loss: q"""
    #     q1, q2 = self.cri.get__q1_q2(state, action)  # twin critics
    #     critic_loss = self.criterion(q1, q_label) + self.criterion(q2, q_label)
    #
    #     self.cri_optim.zero_grad()
    #     critic_loss.backward()
    #     clip_grad_norm_(self.cri.parameters(), max_norm=self.clip_grad_norm)
    #     self.cri_optim.step()
    #     self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
    #
    #     update_a += 1
    #     if update_a >= repeat_times:
    #         update_a = 0
    #
    #         '''actor loss: action'''
    #         action_pg, log_prob = self.act.get__a__log_prob(state)
    #         q_value_pg = torch.min(*self.cri_target.get__q1_q2(state, action_pg)).mean()
    #         actor_loss = (self.alpha_tensor * log_prob - q_value_pg).mean()
    #
    #         self.act_optim.zero_grad()
    #         actor_loss.backward()
    #         clip_grad_norm_(self.act.parameters(), max_norm=self.clip_grad_norm)
    #         self.act_optim.step()
    #
    #         self.soft_update(self.act_target, self.act, self.soft_update_tau)


