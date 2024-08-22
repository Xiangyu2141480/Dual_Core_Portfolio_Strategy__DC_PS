import numpy as np
import torch
import torch.nn as nn
import numpy as np
from numpy import *
from scipy import special
from scipy.integrate import quad
from scipy import optimize
from functools import partial
from scipy.linalg import sqrtm
import math
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.mid_dim = mid_dim
        self.state_dim = state_dim
        self.softmax = nn.Sequential(nn.Softmax(dim=1))
        self.fc = nn.Sequential(nn.Linear(10, mid_dim), nn.PReLU(),
                                nn.Linear(mid_dim, action_dim))
        self.Q_activation = nn.Sequential(nn.PReLU())
        self.fc1_1 = nn.Sequential(nn.Linear(action_dim, action_dim * 128), nn.LogSigmoid(),
                                   nn.Linear(action_dim * 128, action_dim))
        self.conv_EIIE_1 = nn.Sequential(nn.Conv2d(3, 2, kernel_size=(3, 1), stride=(1, 1), padding=(0, 0)),
                                       # (batch_size * 8 * 32 * 10)
                                       nn.ReLU(),
                                       nn.Conv2d(2, 20, kernel_size=(48, 1), stride=(1, 1), padding=(0, 0)),
                                       # (batch_size * 16 * 16 * 10)
                                       nn.ReLU())
        self.conv_EIIE_2 = nn.Sequential(nn.Conv2d(21, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                       # (batch_size * 32 * 8* 10)
                                       nn.Flatten())

    def extract_policy_network_input(self, state):
        state_network_input = state[:, :, -50:, :]
        last_portfolio_vector = state[:, 0:1, 0:1, :]
        # print("return vector in neural network:",relative_return_vector)
        # print("portolio_vector in neural network:",portolio_vector)
        return state_network_input, last_portfolio_vector

    def add_bias(self, EIIE_conv_output):
        H, W = EIIE_conv_output.shape
        softmax_input = torch.concat((torch.ones([H,1]).cuda(), EIIE_conv_output), dim=1)
        return softmax_input

    def forward(self, state, last_action = None): # torch.zeros([1, 1, 1, 29]).cuda()
        state = state.permute(0, 3, 1, 2)
        state_network_input, last_portfolio_vector = self.extract_policy_network_input(state)
        Conv_EIIE_out_1 = self.conv_EIIE_1(state_network_input) # self.conv_EIIE(torch.log2(state_network_input)*100)
        if last_action == None:
            last_action = last_portfolio_vector
        Conv_EIIE_out_1 = torch.cat([Conv_EIIE_out_1, last_action], dim=1)
        Conv_EIIE_out_2 = self.conv_EIIE_2(Conv_EIIE_out_1)
        Conv_EIIE_softmax_input = self.add_bias(Conv_EIIE_out_2)
        W = self.softmax(Conv_EIIE_softmax_input)
        return W  # 直接输出权重的时候使用

    def do_return_prediction_Q_value(self, state, prediction_value, if_training = True):
        state = state.permute(0, 3, 1, 2)
        state_network_input, BL_input = self.extract_policy_network_input(state)
        y_pred = self.Q_predict_via_FactorAE_module(torch.log2(state_network_input)*100, prediction_value)
        return y_pred