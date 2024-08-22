import numpy as np
import torch
from AgentBase import AgentBase
from AgentDDPG import AgentDDPG
from AgentSAC import AgentSAC
from Actor import Actor
from Critic import Critic


class AgentMADDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.batch_size = None
        self.n_actions = None
        self.n_states = None
        self.agent_id = None
        self.agents = None
        self.n_agents = None
        self.update_tau = None
        self.ClassAct = Actor
        self.ClassCri = Critic
        self.if_use_cri_target = True
        self.if_use_act_target = True
        self.repeat_times = 4

    def init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-5, if_per_or_gae=False, env_num=1, gpu_id=0, agent_type="DDPG"):
        """
        Initialize the MADDPG agent with the given parameters.
        """
        self.n_agents = 2
        # Create two agents: one using DDPG and the other using SAC
        self.agents = [AgentDDPG() if i == 0 else AgentSAC() for i in range(self.n_agents)]
        self.explore_env = self.explore_one_env
        self.if_off_policy = True
        self.agent_id = 0
        self.states = []
        self.learning_rate = learning_rate

        # Initialize the base agent
        AgentBase.init(self, net_dim=net_dim, state_dim=state_dim, action_dim=action_dim,
                       reward_scale=reward_scale, gamma=gamma,
                       learning_rate=learning_rate, if_per_or_gae=if_per_or_gae,
                       env_num=env_num, gpu_id=gpu_id, )

        # Initialize alpha for SAC
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=learning_rate)
        self.target_entropy = np.log(action_dim)

        # Initialize agents
        for i in range(self.n_agents):
            if i == 0:
                self.agents[i].init(net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_per=False, env_num=1)
            if i == 1:
                self.agents[i].init(net_dim, state_dim, action_dim)

        self.n_states = state_dim
        self.n_actions = action_dim
        self.batch_size = net_dim
        self.soft_update_tau = 2 ** -8

        self.agent1 = self.agents[0]
        self.agent2 = self.agents[1]
        self.device = torch.device(
            f"cuda:{self.agent_id}" if (torch.cuda.is_available() and (self.agent_id >= 0)) else "cpu")
        self.global_cri_optim = torch.optim.Adam(self.cri.parameters(), self.learning_rate * 0.1)
        self.agent1_act_optim = torch.optim.Adam(self.agent1.act.parameters(), self.learning_rate)
        self.agent2_act_optim = torch.optim.Adam(self.agent1.act.parameters(), self.learning_rate)

    def update_agent(self, rewards, dones, actions, state, next_obs):
        """
        Update both agents (DDPG and SAC) using the given experiences.
        """
        alpha = self.alpha_log.exp()
        state = state
        next_state = next_obs

        # Step 1: Update the critic of the SAC module
        with torch.no_grad():
            next_a, next_log_prob = self.agent2.act_target.get_action_logprob(next_state)
            next_q = torch.min(*self.cri_target.SAC_get_q1_q2(next_state.reshape(state.shape[0], -1), next_a))
            q_label = rewards + torch.tensor(self.gamma) * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.SAC_get_q1_q2(state.reshape(state.shape[0], -1), actions)
        obj_critic_SAC = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.

        # Step 2: Update the critic of the DDPG module
        with torch.no_grad():
            next_q = self.cri_target.DDPG_Q_value(next_state.reshape(state.shape[0], -1), self.act_target(next_state))
            q_label = rewards + torch.tensor(self.gamma) * next_q
        q_value = self.cri.DDPG_Q_value(state.reshape(state.shape[0], -1), actions)
        obj_critic_DDPG = self.agent1.criterion(q_value, q_label)

        # Perform the critic update
        self.optim_update(self.global_cri_optim, obj_critic_SAC + obj_critic_DDPG)
        if self.if_use_cri_target:
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        # Step 3: Update the actor of the agent trained with SAC
        action_pg, logprob = self.agent2.act.get_action_logprob(state)
        obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
        self.optim_update(self.alpha_optim, obj_alpha)

        obj_actor_of_agent_trained_with_sac = -(self.cri.SAC_Q_value(state.reshape(state.shape[0], -1), action_pg) + logprob * alpha).mean()
        self.optim_update(self.agent2_act_optim, obj_actor_of_agent_trained_with_sac)
        if self.if_use_act_target:
            self.soft_update(self.agent2.act_target, self.agent2.act, self.soft_update_tau)

        # Step 4: Update the actor of the agent trained with DDPG
        action_pg = self.agent1.act(state)
        obj_actor_of_agent_trained_with_ddpg = -self.cri.DDPG_Q_value(state.reshape(state.shape[0], -1), action_pg).mean()
        self.optim_update(self.agent1_act_optim, obj_actor_of_agent_trained_with_ddpg)
        if self.if_use_act_target:
            self.soft_update(self.agent1.act_target, self.agent1.act, self.soft_update_tau)

        pol_loss_agent1 = obj_actor_of_agent_trained_with_ddpg
        pol_loss_agent2 = obj_actor_of_agent_trained_with_sac
        vf_loss_agent1 = obj_critic_DDPG
        vf_loss_agent2 = obj_critic_SAC

        return pol_loss_agent1.item(), pol_loss_agent2.item(), vf_loss_agent1.item(), vf_loss_agent2.item()

    def update(self, buffer, batch_size, repeat_times):
        """
        Update the agents using samples from the buffer.
        """
        logging_tuple = None
        buffer.update_now_len()
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            rewards, dones, actions, state, next_obs = buffer.sample_batch(self.batch_size)
            logging_tuple = self.update_agent(rewards, dones, actions, state, next_obs)
        return logging_tuple

    def explore_one_env(self, env, target_step) -> list:
        """
        Explore the environment for a given number of steps.
        """
        traj_list = []
        states = self.states
        self.action_dim = 29

        action_weights = [0.75, 0.25]  # Weights for combining actions from different agents

        traj_state = []
        traj_other = []

        for _ in range(target_step):
            actions = []
            for i in range(self.n_agents):
                rand_weights = np.random.rand()
                state_tensor = torch.as_tensor(states[0], dtype=torch.float32)
                action = self.agents[i].select_actions(state_tensor.unsqueeze(0))
                actions.append(action)

            actions_tensor = torch.stack(actions)

            weighted_average_action = torch.sum(
                actions_tensor.squeeze(1) * torch.tensor(action_weights).unsqueeze(1),
                dim=0)

            action = weighted_average_action.numpy()
            next_s, reward, done, _ = env.step(action)

            other = torch.empty(2 + self.action_dim)
            other[0] = reward
            other[1] = done
            other[2:] = weighted_average_action

            state_tensor = torch.as_tensor(states[0], dtype=torch.float32)

            traj_state.append(state_tensor)
            traj_other.append(other)

            global_done = done
            if global_done:
                states = env.reset()[np.newaxis, :, :]
            else:
                states = next_s[np.newaxis, :, :, :]

        traj_list.append((torch.stack(traj_state), torch.stack(traj_other)))

        return traj_list

    def select_actions(self, states):
        """
        Select actions for each agent based on the given states.
        """
        actions = []
        for i in range(self.n_agents):
            action = self.agents[i].select_actions(states[i])
            actions.append(action)
        return actions

    def save_or_load_agent(self, cwd, if_save):
        """
        Save or load agent state.
        """
        for i in range(self.n_agents):
            self.agents[i].save_or_load_agent(cwd, if_save)

    def load_actor(self, cwd):
        """
        Load the actor networks.
        """
        for i in range(self.n_agents):
            self.agents[i].act.load_state_dict(torch.load(cwd + '/actor' + str(i) + '.pth', map_location='cpu'))

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the networks using the buffer.
        """
        buffer.update_now_len()
        self.batch_size = batch_size
        self.update_tau = soft_update_tau
        logging_tuple = self.update(buffer, batch_size, repeat_times)

        # self.update_all_agents()
        return logging_tuple
