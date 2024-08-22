import numpy as np
import torch.nn.functional as F
from elegantrl.agents import AgentBase
from elegantrl.agents.net import Actor, Critic
from ray.rllib.evaluation.rollout_worker import torch


class AgentPPO(AgentBase):
    def __init__(self):
        super(AgentPPO, self).__init__()
        self.if_on_policy = True
        self.ratio_clip_epsilon = 0.2
        self.lambda_entropy = 0.04

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, marl=True, n_agents=1, if_use_per=False,
             env_num=1, agent_id=0):
        self.device = torch.device(f"cuda:{agent_id}" if torch.cuda.is_available() else "cpu")

        # Actor network
        self.act = Actor(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optim = torch.optim.Adam(self.act.parameters(), lr=learning_rate)

        # Critic network
        self.cri = Critic(state_dim, net_dim).to(self.device)
        self.cri.train()
        self.cri_optim = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)

    def select_actions(self, state):
        states = torch.as_tensor((state,), device=self.device)
        action, _ = self.act.get_action(states)
        return action.cpu().data.numpy()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                old_value = self.cri(state).squeeze(1)
                old_log_prob = -(action - self.act(state)).pow(2) / (2 * self.act.a_std.pow(2) + 1e-5) - 0.5 * np.log(
                    2 * np.pi * self.act.a_std.pow(2))
                old_log_prob = old_log_prob.sum(1)

            for _ in range(4):  # learn actor
                log_prob = -(action - self.act(state)).pow(2) / (2 * self.act.a_std.pow(2) + 1e-5) - 0.5 * np.log(
                    2 * np.pi * self.act.a_std.pow(2))
                log_prob = log_prob.sum(1)

                ratio = (log_prob - old_log_prob.detach()).exp()
                surrogate1 = ratio * old_value
                surrogate2 = torch.clamp(ratio, 1 - self.ratio_clip_epsilon, 1 + self.ratio_clip_epsilon) * old_value

                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_entropy = (log_prob.exp() * log_prob).mean()  # policy entropy
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

                self.act_optim.zero_grad()
                obj_actor.backward()
                self.act_optim.step()

            value = self.cri(state).squeeze(1)
            obj_critic = F.smooth_l1_loss(value, old_value.detach())  # TD_error.pow(2).mean()
            self.cri_optim.zero_grad()
            obj_critic.backward()
            self.cri_optim.step()
