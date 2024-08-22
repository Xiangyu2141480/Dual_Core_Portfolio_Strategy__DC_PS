import numpy as np
import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.state_initual = state_dim
        self.commission_rate = 5e-4
        self.fc1 = nn.Sequential(nn.Linear(1280, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 128), nn.Hardswish(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.Hardswish(),
                                 nn.Linear(32, action_dim))
        self.fc2 = nn.Sequential(nn.Linear(action_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1))
        self.conv = nn.Sequential(nn.Conv2d(1, 8, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
                                  # (batch_size * 8 * 32 * 10)
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(8, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
                                  # (batch_size * 16 * 16 * 10)
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
                                  # (batch_size * 32 * 8* 10)
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                                  # (batch_size * 32 * 4 * 10)
                                  nn.Flatten())

    def __pure_pc(self, last_portfolio_weights, future_return_tensor, portfolio_weights):
        c = self.commission_rate
        future_return_tensor = torch.transpose(future_return_tensor, 2, 1)

        # print("1", (last_portfolio_weights * (1 + future_return_tensor))[:, :, 0].shape)
        # print("2", torch.sum((last_portfolio_weights * (1 + future_return_tensor))[:, :, 0], dim = 1).shape)
        future_return_tensor = 1 + future_return_tensor
        # print("future_return_tensor", future_return_tensor.shape)
        future_return_tensor = torch.concat((torch.ones((32, 1, 1)).cuda(), future_return_tensor), dim = 1)
        #print("future_return_tensor", future_return_tensor.shape)
        #print("last_portfolio_weights", last_portfolio_weights.shape)
        #print("(1 - torch.sum(last_portfolio_weights, dim = 1, keep_dim = True)", torch.sum(last_portfolio_weights, dim = 1, keepdim = True).shape)

        last_portfolio_weights = torch.concat(((1 - torch.sum(last_portfolio_weights, dim = 1, keepdim = True)), last_portfolio_weights), dim = 1)
        w_t = (last_portfolio_weights * (future_return_tensor))[:, :, 0] / torch.sum((last_portfolio_weights * (future_return_tensor))[:, :, 0], dim = 1, keepdim=True) # rebalanced
        w_t1 = portfolio_weights
        #print("交易量估计1", torch.sum(torch.abs(portfolio_weights - last_portfolio_weights[:, :, 0]), dim = 1))
        # print("w_t", w_t.shape)
        # print("w_t1", w_t1.shape)
        # print("torch.sum(torch.abs(w_t1 - w_t[:, :, 0]))", torch.sum(torch.abs(w_t1 - w_t[:, :, 0]), dim = 1).shape)
        mu = 1 - torch.sum(torch.abs(w_t1 - w_t), dim = 1) * c
        #print("交易量估计2", torch.sum(torch.abs(w_t1 - w_t), dim = 1))
            #1 - tf.reduce_sum(input_tensor=tf.abs(w_t1[:, 1:]-w_t[:, 1:]), axis=1)*c
        """
        mu = 1-3*c+c**2

        def recurse(mu0):
            factor1 = 1/(1 - c*w_t1[:, 0])
            if isinstance(mu0, float):
                mu0 = mu0
            else:
                mu0 = mu0[:, None]
            factor2 = 1 - c*w_t[:, 0] - (2*c - c**2)*tf.reduce_sum(
                tf.nn.relu(w_t[:, 1:] - mu0 * w_t1[:, 1:]), axis=1)
            return factor1*factor2

        for i in range(20):
            mu = recurse(mu)
        """
        return mu

    def reward_calculation(self, state, action, last_action):
        ######### 相对收益向量 ############
        future_return_tensor = (state[:, :, -1, :]) - 1
        # future_return_tensor_move_noise = torch.clamp((torch.abs(future_return_tensor)),min=0.0) * torch.sign(future_return_tensor)
        # relative_return_vector = torch.mean(future_return_tensor, axis=2)
        # relative_return_vector = torch.mean(torch.log2(state[:, :, -5:, :]) * 100, axis=2)
        portfolio_vetor_last = last_action # torch.transpose(state[:, :, 0, :], 2, 1)
        value_rate_after_commission = self.__pure_pc(portfolio_vetor_last, future_return_tensor, action)
        # print("action_critic", action.size())
        W = action[:, 1:]
        # return_vector_out = torch.log2(torch.bmm((torch.exp2(relative_return_vector) - 1), torch.unsqueeze(W, 2)).squeeze(2) + 1) - 0.001 * torch.sum(torch.abs(torch.unsqueeze(W, 2) - portfolio_vetor_last), axis=1)
        return_vector_out = torch.log((torch.bmm(future_return_tensor, torch.unsqueeze(W, 2))[:,:,0] + 1) * torch.unsqueeze(value_rate_after_commission, 1)) # - 0.001 * torch.sum(torch.abs(torch.unsqueeze(W, 2) - portfolio_vetor_last), axis=1)
        # print("torch.log((torch.bmm(future_return_tensor, torch.unsqueeze(W, 2))[:,:,0] + 1)", (torch.log((torch.bmm(future_return_tensor, torch.unsqueeze(W, 2))[:,:,0] + 1)).size()))
        # print("value_rate_after_commission", value_rate_after_commission.size())
        # print("return_vector_out", return_vector_out.size())
        # print("torch.unsqueeze(W, 2)",torch.unsqueeze(W, 2).size())
        ### return_vector_out = torch.bmm(relative_return_vector, torch.unsqueeze(W, 2)).squeeze(2)
        ### return_vector = torch.bmm(relative_return_vector, torch.unsqueeze(W, 2))
        ### H, W, C = return_vector.size()
        ### modified_return_vector = torch.cat([return_vector, torch.ones((H, W, C)).cuda() * (0.03)], axis=2)
        # modified_return_vector = torch.cat([return_vector, torch.ones((H, W, C)) * (0.1)], axis=2)
        ### final_modified_return_vector, b = torch.min(modified_return_vector, dim=2, keepdim=False, out=None)
        # print("final_modified_return_vector:",final_modified_return_vector)
        # adjusted_return_vector = torch.min(return_vector, 0.02)
        # return_vector = torch.bmm(relative_return_vector, torch.unsqueeze(W, 2))[:,:,0]
        # print("纯收益:",(torch.bmm(relative_return_vector, torch.unsqueeze(W, 2))[:,:,0]).sum())
        # print("手续费：",(0.002 * torch.mean(torch.abs(torch.unsqueeze(W, 2)-portfolio_vetor_last),axis=1)).sum())
        # print("实际收益:",return_vector.sum())
        return return_vector_out

        # Reward_list = torch.zeros(B)

    def modified_value(self, state, action):
        state = state[:, :, 6:, :]
        # print("state.shape",state.shape)
        relative_return_vector = torch.log2(state)  # bs/1/32/10
        return_mean = torch.mean(relative_return_vector, axis=2)  # bs/1/10
        last_return_vector = torch.mean(torch.log2(state[:, :, -5:, :]) * 100, axis=2)
        # print("last_return_vector",last_return_vector.size())
        # print("relative_return_vector",relative_return_vector[:,0,:,:].size())
        # print("return_mean",return_mean.size())
        # 计算方差协方差矩阵
        covariance_matrix = torch.bmm(torch.transpose((relative_return_vector[:, 0, :, :] - return_mean), 2, 1),
                                      (relative_return_vector[:, 0, :, :] - return_mean)) / (200-29-1)
        # 计算方差
        weights = torch.unsqueeze(action, 2)
        # print("weights",weights.size())
        # print("covariance_matrix",covariance_matrix.size())
        variance = torch.bmm(torch.bmm(torch.transpose(weights, 2, 1), covariance_matrix), weights)
        # 计算基于预期收益的校准值
        covariance_matrix_inv = torch.linalg.inv(covariance_matrix)
        mu = torch.bmm(torch.bmm(last_return_vector, covariance_matrix_inv), torch.transpose(last_return_vector, 2, 1))
        expected_return = 0.5
        Lamda = expected_return / mu
        # 计算校准之后的额风险值
        # modifed_variance = (1/(2*Lamda)) * variance
        modifed_variance = 10 * variance
        # print("Lamda:",Lamda)
        # print("risk modifed in net:",(1/(2*Lamda)))
        # print("variance in net:",variance)
        return modifed_variance[:, :, 0]


    def forward(self, state, action, last_action):
        state = state.permute(0, 3, 1, 2)
        state = state[:, 0:1, :, :]
        last_action = last_action[:, 1:].unsqueeze(2)
        reward = self.reward_calculation(state, action, last_action)
        objective_reward = reward
        return objective_reward

    def evaluate_function(self, state, action, last_action):
        state = state.permute(0, 3, 1, 2)
        state = state[:, 0:1, :, :]
        last_action = last_action[:, 1:].unsqueeze(2)
        reward = self.reward_calculation(state, action, last_action)
        modified_variance = self.modified_value(state, action)
        evaluating_value = reward - modified_variance
        return evaluating_value