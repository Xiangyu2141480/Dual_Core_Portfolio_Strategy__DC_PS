import numpy as np
import torch
import torch.nn as nn

class Critic(nn.Module):
    """
    A simple Critic class.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state, action):
        """
        The forward function.

        :param state[np.array]: the input state.
        :param action[float]: the input action.
        :return: the output tensor.
        """
        return self.net(torch.cat((state, action), dim=1))  # Q value