import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    The Critic class for Clipped Double DQN.
    This network estimates the Q-values for given states and actions.
    """

    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        state_dim = 5800  # Fixed state dimension

        # Determine the middle network structure
        if if_use_dn:
            nn_middle = DenseNet(mid_dim // 2)  # Use DenseNet if specified
            inp_dim = nn_middle.inp_dim
            out_dim = nn_middle.out_dim
        else:
            # Default network structure
            nn_middle = nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU()
            )
            inp_dim = mid_dim
            out_dim = mid_dim

        # Process actions to match state dimensions
        self.action_process = nn.Sequential(
            nn.Linear(action_dim, state_dim),
            nn.ReLU()
        )

        # Network to process (state, action) pairs
        self.net_sa = nn.Sequential(
            nn.Linear(action_dim + state_dim, inp_dim),
            nn.ReLU(),
            nn_middle
        )

        # Q-value networks for different algorithms
        self.net_q1 = nn.Sequential(
            nn.Linear(out_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, 1)
        )

        self.net_q2 = nn.Sequential(
            nn.Linear(out_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, 1)
        )

        self.net_q3 = nn.Sequential(
            nn.Linear(out_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, 1)
        )

    def forward(self, state, action):
        """
        Forward pass for computing the Q-value.

        :param state: Tensor representing the state input.
        :param action: Tensor representing the action input.
        :return: Tensor representing the Q-value computed by net_q1.
        """
        tmp = self.net_sa(torch.cat((state, action), dim=1))  # Concatenate state and action
        return self.net_q1(tmp)  # Compute Q-value using net_q1

    def DDPG_Q_value(self, state, action):
        """
        Compute Q-value for DDPG using net_q3.

        :param state: Tensor representing the state input.
        :param action: Tensor representing the action input.
        :return: Tensor representing the Q-value computed by net_q3.
        """
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q3(tmp)  # Compute Q-value using net_q3

    def SAC_Q_value(self, state, action):
        """
        Compute Q-value for SAC using net_q1.

        :param state: Tensor representing the state input.
        :param action: Tensor representing the action input.
        :return: Tensor representing the Q-value computed by net_q1.
        """
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # Compute Q-value using net_q1

    def SAC_get_q1_q2(self, state, action):
        """
        Compute two Q-values for SAC using net_q1 and net_q2.

        :param state: Tensor representing the state input.
        :param action: Tensor representing the action input.
        :return: Tuple of tensors representing the Q-values computed by net_q1 and net_q2.
        """
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # Compute Q-values using net_q1 and net_q2


class DenseNet(nn.Module):
    """
    A DenseNet class for creating a dense network with concatenated layers.
    """

    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(lay_dim * 1, lay_dim * 1),
            nn.Hardswish()
        )
        self.dense2 = nn.Sequential(
            nn.Linear(lay_dim * 2, lay_dim * 2),
            nn.Hardswish()
        )
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x1):
        """
        Forward pass for DenseNet.

        :param x1: Input tensor of shape (-1, lay_dim).
        :return: Tensor of shape (-1, lay_dim * 4) after passing through DenseNet layers.
        """
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)  # Concatenate input and output of dense1
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)  # Concatenate intermediate output and output of dense2
        return x3
