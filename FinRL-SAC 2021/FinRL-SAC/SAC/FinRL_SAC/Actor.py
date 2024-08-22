import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    A simple Actor class.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        """
        The forward function.

        :param state[np.array]: the input state.
        :return: the output tensor.
        """
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state, action_std):
        """
        The forward function with Gaussian noise.

        :param state[np.array]: the input state.
        :param action_std[float]: the standard deviation of the Gaussian distribution.
        :return: the output tensor.
        """
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)