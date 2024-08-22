import gym
import numpy as np
from numpy import random as rd
from scipy.linalg import *
import torch
import torch.nn as nn


class StockTradingEnv(gym.Env):
    def __init__(
            self,
            config,
            initial_account=1e6,
            gamma=0.99,
            turbulence_thresh=99,
            min_stock_rate=0.1,
            max_stock=29,
            initial_capital=1e7,
            buy_cost_pct=5e-4,
            sell_cost_pct=5e-4,
            reward_scaling=2 ** -11,
            initial_stocks=None,
            window_size=56
    ):
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        turbulence_ary = config["turbulence_array"]
        OCLC_ary = config["OCLC_array"]
        close_in_OCLC_array = config["close_in_OCLC_array"]
        if_train = config["if_train"]
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary
        self.OCLC_ary = OCLC_ary.astype(np.float32)
        self.close_in_OCLC_array = close_in_OCLC_array.astype(np.float32)
        # print("self.price_ary", self.price_ary.shape)
        # print("self.OCLC_ary", self.OCLC_ary.shape)

        self.tech_ary = self.tech_ary * 2 ** -7
        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (
                self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2 ** -5
        ).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.horizen_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = "StockEnv"
        self.window_size = window_size
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.state_dim = stock_dim + self.tech_ary.shape[1]
        self.state_dim1 = self.price_ary.shape[1]
        self.state_dim2 = self.tech_ary.shape[1] / self.price_ary.shape[1]
        # self.state_dim2 =  self.tech_ary.shape[1]
        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim + 1
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 100000.0
        self.episode_return = 0.0
        self.episode_expected_risk = 0.0
        self.eposide_return_variance = 0.0
        self.return_risk_profit = 0.0
        self.return_without_commission = 0.0
        self.last_action = np.zeros(stock_dim)
        self.presentaion_weights = np.zeros(stock_dim)
        # self.return_modified_by_risk = 0

        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    def reset(self):
        self.day = self.window_size - 5
        price = self.price_ary[self.day]

        if self.if_train:
            # 随机起始资产 self.stocks = (self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)).astype(np.float32)
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            # 随机起始资产 self.amount = (self.initial_capital * rd.uniform(0.95, 1.05) - (self.stocks * price).sum())
            self.amount = (self.initial_capital - (self.stocks * price).sum())
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        self.horizen_reward = 0.0
        self.episode_expected_risk = 0.0
        self.eposide_return_variance = 0.0
        self.return_risk_profit = 0.0
        self.return_without_commission = 0.0
        # print("price:",len(price[:]))
        self.last_action = np.zeros(len(price[:]))
        self.presentaion_weights = np.zeros(len(price[:]))
        # self.return_modified_by_risk = 0
        # print("self.stocks:",self.stocks)
        # print("self.amount:",self.amount)
        # print("self.initial_total_asset:",self.initial_total_asset)
        return self.get_state(self.last_action)  # state

    def collecting(self, Date, window):
        # self.day = Date
        information = self.price_ary[Date - window + 1:Date + 1]
        return information

    def collecting_all_history(self, Date):
        information = self.price_ary[:Date + 1]
        return information

    def beginning(self):
        self.day = self.window_size - 5
        price = self.price_ary[self.day]
        self.presentaion_weights = np.zeros(len(price[:]))
        if self.if_train:
            # 随机起始资产 self.stocks = (self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)).astype(np.float32)
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            # 随机起始资产 self.amount = (self.initial_capital * rd.uniform(0.95, 1.05) - (self.stocks * price).sum())
            self.amount = (self.initial_capital - (self.stocks * price).sum())
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        self.horizen_reward = 0.0
        self.episode_expected_risk = 0.0
        self.eposide_return_variance = 0.0
        self.return_risk_profit = 0.0
        self.return_without_commission = 0.0
        return self.day, self.initial_total_asset, self.horizen_reward, self.eposide_return_variance

    def backtest_step(self, portfolio_weights):
        # self.day = Date
        # print(self.day)
        price = self.price_ary[self.day]

        return_vector = np.log2(self.price_ary[self.day] / self.price_ary[self.day - 1])
        # weights_test = (return_vector==return_vector.max(axis = 0, keepdims = 1)).astype(float)
        # print(return_vector)
        # print(return_vector)
        W = portfolio_weights
        total_amount = self.total_asset
        # invested_asset = self.initial_total_asset
        # actions_share = (invested_asset * W) / price
        actions_share = (total_amount * W) / price
        actions_share = actions_share - self.stocks
        actions = actions_share.astype(int)
        self.day += 1
        Date = self.day
        price_end = self.price_ary[self.day]
        self.stocks_cool_down += 1

        if self.turbulence_bool[self.day] == 0:
            min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[index] > 0:  # Sell only if current asset is > 0
                    # sell_num_shares = min(self.stocks[index], -actions[index])
                    sell_num_shares = -actions[index]
                    self.stocks[index] -= sell_num_shares
                    self.amount += (price[index] * sell_num_shares * (1 - self.sell_cost_pct))
                    # self.amount += (price[index] * sell_num_shares )
                    self.stocks_cool_down[index] = 0
            for index in np.where(actions > min_action)[0]:  # buy_index:
                if (
                        price[index] > 0
                ):  # Buy only if the price is > 0 (no missing data in this particular date)
                    # buy_num_shares = min(self.amount // price[index], actions[index])
                    buy_num_shares = actions[index]
                    self.stocks[index] += buy_num_shares
                    self.amount -= (price[index] * buy_num_shares * (1 + self.buy_cost_pct))
                    # self.amount -= (price[index] * buy_num_shares)
                    self.stocks_cool_down[index] = 0

        else:  # sell all when turbulence
            self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cool_down[:] = 0

        if self.amount <= 0:
            self.amount = self.amount * (1 + (0.03/250))

        for index in np.where(self.stocks < 0)[0]:  # sell_index:
            if price[index] > 0:  # Sell only if current asset is > 0
                    # sell_num_shares = min(self.stocks[index], -actions[index])
                self.amount -= (np.abs(price[index] * self.stocks[index]) * (0.03 / 250))

        total_asset = self.amount + (self.stocks * price_end).sum()

       #  historical_return_vector = self.price_ary[self.day - self.window_size+5:self.day] / self.price_ary[self.day-self.window_size-4:self.day - 1] - 1
        historical_return_vector = self.price_ary[self.day - self.window_size + 6:self.day + 1] / self.price_ary[self.day - self.window_size + 5:self.day] - 1
        log_x = (np.log(historical_return_vector + 1))
        # print(historical_return_vector.shape)
        Convariance_matrix = np.cov(log_x, rowvar=False)
        weights = W

        self.episode_expected_risk += self.variance(W, Convariance_matrix) * 10
        reward = np.log(total_asset / self.total_asset) - self.variance(W, Convariance_matrix)[0, 0]
        self.total_asset = total_asset
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        self.horizen_reward += reward
        done = self.day + 51 >= self.max_step
        """
        Experiment 2021
                   self.day + 51
        Experiment 2023
                  self.day + 49
        """
        if done:
            horizen_reward = self.horizen_reward
            self.episode_return = np.log2(self.total_asset / self.initial_total_asset)
            self.eposide_return_variance = self.episode_expected_risk
            self.return_risk_profit = self.episode_return - self.eposide_return_variance
            # print(np.log2(self.total_asset/self.initial_total_asset))
        return np.log(self.total_asset / self.initial_total_asset), self.horizen_reward, self.total_asset, Date, done

    # 失败测试
    def tranform_network_to_vector1(self, Q):
        # torch.nn
        price = self.price_ary[self.day]
        historical_price1 = self.price_ary[self.day - self.window_size:self.day]
        historical_return_vector = self.price_ary[self.day - self.window_size:self.day] / self.price_ary[
                                                                                          self.day - self.window_size - 1:self.day - 1] - 1
        lamda = 1
        Tau = 1
        #################################################################################
        log_x = (np.log2(historical_return_vector + 1)) * 100
        # log_x = historical_return_vector*100
        # print("log_x:",log_x)
        Convariance_matrix = np.cov(log_x, rowvar=False)
        # print("Convariance_matrix:",Convariance_matrix)
        P = np.identity(log_x.shape[1])
        Tau_Convariance_matrix = Tau * Convariance_matrix
        mu4 = np.dot(np.dot(P, Tau * Convariance_matrix), P.T)
        Omega = np.diag(np.diag(mu4))
        Pai = np.mean(log_x[-1:, :], axis=0)
        ###################################################################
        P = np.matrix(P)
        Tau_Convariance_matrix = np.matrix(Tau_Convariance_matrix)
        Omega = np.matrix(Omega)
        Pai = np.matrix(Pai).T
        # print("Pai:",Pai)
        ####################################################################
        Tau_Convariance_matrix_inv = inv(Tau_Convariance_matrix)
        Tau_Convariance_matrix_inv = torch.as_tensor(Tau_Convariance_matrix_inv, dtype=torch.float32)

        inv_Omega = inv(Omega)
        inv_Omega = torch.as_tensor(inv_Omega, dtype=torch.float32)

        P = torch.as_tensor(P, dtype=torch.float32)
        Pai = torch.as_tensor(Pai, dtype=torch.float32)

        mu1 = inv(Tau_Convariance_matrix_inv + P.T * inv(Omega) * P)
        mu1 = torch.as_tensor(mu1, dtype=torch.float32)

        # Q = np.matrix(Q).T
        # Q = torch.as_tensor(Q, dtype=torch.float32)
        ###################################################################
        mu2 = torch.mm(Tau_Convariance_matrix_inv, Pai) + torch.mm(torch.mm(P, inv_Omega), Q)
        # mu2 = Tau_Convariance_matrix_inv * Pai + P * inv_Omega * Q
        mu3 = torch.mm(mu1, mu2)
        # print("expected return:",mu3)
        W = torch.mm(Tau_Convariance_matrix_inv, mu3) / Lamda
        return W

    def variance(self, actions, Convariance_matrix):
        Convariance_matrix = np.matrix(Convariance_matrix)
        actions = np.matrix(actions)
        variance = actions * Convariance_matrix * actions.T
        # print("variance:",variance)
        return variance

    def modified_calculation(self, r, Convariance_matrix, weights):
        R = np.matrix(r)
        R_abs = np.absolute(R)
        Convariance_matrix = np.matrix(Convariance_matrix)
        Weights = np.matrix(weights)
        e = np.matrix(np.ones((1, 10)))
        equal_vector = np.matrix(np.ones((1, 10))) / 10
        # Excepted_return = r * Weights.T + 0.03
        Excepted_return = 0.5
        # print("Excepted return:",Excepted_return)
        Alpha = e * inv(Convariance_matrix) * e.T
        # print("Alpha:",Alpha)
        Beta = e * inv(Convariance_matrix) * R.T
        # print("Beta:",Beta)
        Gamma = R * inv(Convariance_matrix) * R.T
        Delta = Alpha * Gamma - Beta * Beta
        # print("Delta:",Delta)
        Lamda = (Gamma - Beta * Excepted_return) / Delta
        # Mu = (Alpha * Excepted_return - Beta)/Delta
        Mu = Excepted_return / Gamma
        # print("Mu",Mu)
        # ("Part1:",Alpha * Excepted_return)
        # print("Part2:",Beta)
        # Modified_value = 1/(2 * Mu)
        # print("Modified value:",1/(2 * Mu))
        return Mu[0, 0]

    # 主要使用的公式
    def allocation_function(self, historical_return_vector, lamda, Tau, Q, total_amount, stocks_hold, price):
        log_x = (np.log2(historical_return_vector + 1)) * 100
        # log_x = historical_return_vector*100
        # print("log_x:",log_x)
        Convariance_matrix = np.cov(log_x, rowvar=False)
        # print("Convariance_matrix:",Convariance_matrix)
        P = np.identity(log_x.shape[1])
        Tau_Convariance_matrix = Tau * Convariance_matrix
        mu3 = np.dot(np.dot(P, Tau * Convariance_matrix), P.T)
        Omega = np.diag(np.diag(mu3))
        Pai = np.mean(log_x[-3:, :], axis=0)
        ###################################################################
        P = np.matrix(P)
        Tau_Convariance_matrix = np.matrix(Tau_Convariance_matrix)
        Omega = np.matrix(Omega)
        Pai = np.matrix(Pai).T
        # print("Pai:",Pai)
        ####################################################################
        Tau_Convariance_matrix_inv = inv(Tau_Convariance_matrix)
        mu1 = inv(Tau_Convariance_matrix_inv + P.T * inv(Omega) * P)
        Q = np.matrix(Q).T
        ###################################################################
        mu2 = Tau_Convariance_matrix_inv * Pai + P * inv(Omega) * Q
        mu3 = mu1 * mu2
        # print("expected return:",mu3)
        W = Tau_Convariance_matrix_inv * mu3 / lamda
        # print("W:",W)
        ######################################################################
        W_realize = np.array(W)[:, 0]
        asset_allocation_amount = total_amount * W_realize
        # print("asset_allocation_amount:",asset_allocation_amount)
        asset_allocation_shares = asset_allocation_amount / price
        # print("asset_allocation_shares:",asset_allocation_shares)
        asset_reallocation_shares = asset_allocation_shares - stocks_hold

        ######################################################################
        # print("price:",price)
        # print("number_of_share:",asset_reallocation_shares )
        return asset_reallocation_shares, W_realize

    def pure_reward_calculation(self, r, weights):
        R = np.matrix(r)
        R_abs = np.absolute(R)
        Weights = np.matrix(weights)
        return_without_commission = np.log2((np.exp2(R) - 1) * Weights.T + 1)
        # print("return_without_commission",return_without_commission)
        e = np.matrix(np.ones((1, 10)))
        return return_without_commission[0, 0]

    def covariance_matrix_calculation(self, r):
        # print("r", np.size(r, axis =1))
        mean = np.mean(r, axis = 0, keepdims = True)
        # print("mean", mean.shape)
        covariance_matrix =  (np.matrix(r - mean).T * np.matrix(r - mean))/(np.size(r, axis=0) - np.size(r, axis = 1) - 1)
        # print("covariance_matrix", covariance_matrix.shape)
        # print("(np.size(r, axis=0) - np.size(r, axis = 1) - 1)", (np.size(r, axis=0) - np.size(r, axis = 1) - 1))
        return covariance_matrix

    def step(self, actions):
        # actions = (actions * self.max_stock).astype(int)
        # print("actions:",actions)
        actions = actions[1:]
        price = self.price_ary[self.day]
        # print("Day:",self.day)
        # print("price:",price)
        # print("type:",price.type)
        ############################################################################################################################
        ## 神经网络输出的为Q值的时候使用
        # historical_price1 = self.price_ary[self.day-self.window_size:self.day]
        # historical_return_vector = self.price_ary[self.day-self.window_size:self.day]/self.price_ary[self.day-self.window_size-1:self.day-1]-1
        # lamda = 1
        # Tau = 1
        # Q = actions * 10
        ## print("Q:",Q)
        # total_amount = self.total_asset
        ## print("total_amount:",total_amount)
        # stocks_hold = self.stocks
        ## print("stocks_hold:",stocks_hold)
        # actions_share, W = self.allocation_function(historical_return_vector, lamda, Tau, Q, total_amount, stocks_hold, price)
        # actions = actions_share.astype(int)
        ## print("actions:",actions)
        ########################################################################################################################
        # 神经网络输出portfolio weights的情况使用
        # print("historical_return_vector:",historical_return_vector)
        # print("actions:",actions)
        W = actions
        # print("action:", W)
        total_amount = self.total_asset
        invested_amount = total_amount # self.initial_total_asset
        actions_share = (invested_amount * actions) / price
        actions_share = actions_share - self.stocks
        actions = actions_share.astype(int)
        ######################################################################################################################
        self.day += 1
        price_end = self.price_ary[self.day]
        self.stocks_cool_down += 1

        if self.turbulence_bool[self.day] == 0:
            min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[index] > 0:  # Sell only if current asset is > 0
                    # sell_num_shares = min(self.stocks[index], -actions[index])
                    sell_num_shares = -actions[index]
                    self.stocks[index] -= sell_num_shares
                    self.amount += (price[index] * sell_num_shares * (1 - self.sell_cost_pct))
                    # print("self.sell_cost_pct",self.sell_cost_pct)
                    # self.amount += (price[index] * sell_num_shares )
                    self.stocks_cool_down[index] = 0
            for index in np.where(actions > min_action)[0]:  # buy_index:
                if (
                        price[index] > 0
                ):  # Buy only if the price is > 0 (no missing data in this particular date)
                    # buy_num_shares = min(self.amount // price[index], actions[index])
                    buy_num_shares = actions[index]
                    self.stocks[index] += buy_num_shares
                    self.amount -= (price[index] * buy_num_shares * (1 + self.buy_cost_pct))
                    # self.amount -= (price[index] * buy_num_shares)
                    self.stocks_cool_down[index] = 0
            amount = self.amount
            # print("self.amount1", self.amount)
            ###########################################################################################
        else:  # sell all when turbulence
            self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cool_down[:] = 0
        if self.amount <= 0:
            self.amount = self.amount * (1 + (0.03/250))

        for index in np.where(self.stocks < 0)[0]:  # sell_index:
            if price[index] > 0:  # Sell only if current asset is > 0
                    # sell_num_shares = min(self.stocks[index], -actions[index])
                self.amount -= (np.abs(price[index] * self.stocks[index]) * (0.03 * 5 / 250))
                    # print("self.sell_cost_pct",self.sell_cost_pct)
        # print("self.amount", (amount - self.amount)/self.amount)
        # 上一期的投资组合同时作为state的一部分
        # 更新最新的投资组合向量
        self.last_action = W
        portfolio_weights_before_allocation = self.stocks * self.price_ary[self.day]/np.sum(self.stocks * self.price_ary[self.day])
        state, state_EIIE, future_return_based_on_closing = self.get_state(self.last_action)
        # print("last_action env",self.last_action )
        # print("W",W)
        # self.last_action = W
        # print("self.last_action ",self.last_action )
        # print("self.stocks:",self.stocks)
        total_asset = self.amount + (self.stocks * price_end).sum()
        # self.total_asset = total_asset
        # print("total_asset:",total_asset)
        ####################################################################################
        # reward = (total_asset - self.total_asset) * self.reward_scaling
        # reward = (total_asset-self.total_asset) /self.total_asset
        # reward = np.log2(total_asset/self.total_asset)
        # print("reward:",reward)
        # print("self.reward_scaling:",self.reward_scaling)
        # self.total_asset = total_asset

        # self.gamma_reward = self.gamma_reward * self.gamma + reward
        # self.horizen_reward+= reward
        # print("horizen_reward:",self.horizen_reward)
        ##############################################################################
        historical_return_vector = self.price_ary[self.day - self.window_size + 7:self.day + 1] / self.price_ary[self.day - self.window_size + 6:self.day]
        # log_x = (np.log2(historical_return_vector+1))
        log_x = np.log(historical_return_vector)
        # log_x_mean = np.mean(log_x, axis=0)
        log_x_covariance_matrix = self.covariance_matrix_calculation(log_x)

        r = np.mean(log_x[-1:, :], axis=0)
        # print("Return vector env:", r)
        Convariance_matrix = np.cov(log_x, rowvar=False)



        weights = W

        # Mu = self.modified_calculation(r, Convariance_matrix, weights)
        # Modified_value = (1 / (2 * Mu))
        # print("Modified_value:",Modified_value)
        # print("actions:",W)
        # print("Convariance_matrix:",Convariance_matrix)
        # self.episode_expected_risk+= (Modified_value * self.variance(W, Convariance_matrix))[0,0]
        self.episode_expected_risk += 10 * self.variance(W, log_x_covariance_matrix)[0, 0]
        # print("Modified risk:",(Modified_value * self.variance(W, Convariance_matrix))[0,0])
        # print("Accumulated Modified risk:",self.episode_expected_risk)
        modification_value = np.sum(np.abs(actions * price))/invested_amount
        # print("variance1", self.variance(W, Convariance_matrix))
        # print("variance2", self.variance(W, log_x_covariance_matrix))
        ################################################################################

        reward = np.log(total_asset / self.total_asset)
        '''
        if total_asset >= 1000:
            ### reward = (((total_asset - self.total_asset + self.initial_total_asset) / self.initial_total_asset) - 1) - 50 * self.variance(W, Convariance_matrix)[0, 0]
            ### reward = (np.log2((total_asset - self.total_asset + self.initial_total_asset) / self.initial_total_asset)) * (1/5) - 10 * self.variance(W, Convariance_matrix)[0, 0] - 0.001 * modification_value
            reward = (np.log((total_asset - self.total_asset + self.initial_total_asset) / self.initial_total_asset)) - (10 * self.variance(W, log_x_covariance_matrix)[0, 0]) - 0.001 * modification_value
            # reward = reward * (1e-2)
        else:
            ### reward = ((1000 / self.initial_total_asset) - 1)  - 50 * self.variance(W, Convariance_matrix)[0, 0]
            ###reward = (np.log2(1000 / self.initial_total_asset)) * (1/5) - 10 * self.variance(W, Convariance_matrix)[0, 0] - 0.001 * modification_value
            reward = (np.log(1000 / self.initial_total_asset)) - (10 * self.variance(W, log_x_covariance_matrix)[0, 0]) - 0.001 * modification_value
            # reward = 0
        '''
        # reward = np.log2(total_asset/self.total_asset)
        # print("reward:",reward)
        # reward = np.log2(total_asset/self.total_asset)
        # print("Logreturn:",np.log2(total_asset/self.total_asset))
        # print("Objective function:",reward)
        # reward = (total_asset-self.total_asset)*reward_scaling
        # print("Accumulated reward:",np.log2(total_asset/self.total_asset))
        # reward = np.log2(total_asset/self.total_asset) * self.reward_scaling
        self.total_asset = total_asset
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        self.horizen_reward += reward
        pure_reward = self.pure_reward_calculation(r, weights)
        # print("pure_reward:",pure_reward)
        # print("========================================")
        self.return_without_commission += pure_reward
        ##################################################################################################
        # self.return_modified_by_risk = np.log2(self.total_asset / self.initial_total_asset) - self.episode_expected_risk
        # self.return_risk_profit = np.log2(self.total_asset / self.initial_total_asset)  - self.episode_expected_risk
        ##################################################################################################
        done = self.day + 2 >= self.max_step
        # print("===================================================================")
        # print("self.day:",self.day)
        # print("self.max_step:",self.max_step)
        # print("done:",done)
        if done:
            # reward = self.gamma_reward
            horizen_reward = self.horizen_reward
            # print("horizen_reward:",horizen_reward )
            # 相对收益
            # self.episode_return = (self.total_asset-self.initial_total_asset) / self.initial_total_asset
            # 对数收益
            self.episode_return = np.log(self.total_asset / self.initial_total_asset)
            # print("episode_return:",self.episode_return)
            # # print("total_asset:",self.total_asset)
            # print("initial_total_asset:",self.initial_total_asset)
            # expected risk
            self.eposide_return_variance = self.episode_expected_risk
            # logreturn-expected risk
            # self.return_risk_profit = np.log2(self.episode_return+1) - self.eposide_return_variance
            # self.return_risk_profit = self.episode_return - self.eposide_return_variance
            # return_risk_profit此时用于描述不考虑手续费的累计收益情况
            self.return_risk_profit = self.return_without_commission
            # print("return_risk_profit:",self.return_risk_profit)
        return state, state_EIIE, future_return_based_on_closing, reward, done, dict()

    def DRL_agent_step(self, actions, modified = False):
        # actions = (actions * self.max_stock).astype(int)
        actions = actions[1:]
        # print("actions:",actions)
        price = self.price_ary[self.day]
        # print("Day:",self.day)
        # print("price:",price)
        # print("type:",price.type)
        ############################################################################################################################
        ## 神经网络输出的为Q值的时候使用
        # historical_price1 = self.price_ary[self.day-self.window_size:self.day]
        # historical_return_vector = self.price_ary[self.day-self.window_size:self.day]/self.price_ary[self.day-self.window_size-1:self.day-1]-1
        # lamda = 1
        # Tau = 1
        # Q = actions * 10
        ## print("Q:",Q)
        # total_amount = self.total_asset
        ## print("total_amount:",total_amount)
        # stocks_hold = self.stocks
        ## print("stocks_hold:",stocks_hold)
        # actions_share, W = self.allocation_function(historical_return_vector, lamda, Tau, Q, total_amount, stocks_hold, price)
        # actions = actions_share.astype(int)
        ## print("actions:",actions)
        ########################################################################################################################
        # 神经网络输出portfolio weights的情况使用
        # print("historical_return_vector:",historical_return_vector)
        # print("actions:",actions)
        W = actions
        # print("action3:", W)

        total_amount = self.total_asset

        # invested_asset = self.initial_total_asset
        '''
        if total_amount >= self.initial_total_asset:
            invested_asset = total_amount
        else:
            invested_asset = self.initial_total_asset        
        '''

        invested_asset = total_amount # self.initial_total_asset
        if modified == True:
            actions_share = (invested_asset * actions) / price
        if modified == False:
            actions_share = self.stocks
        # actions_share = (total_amount * actions) / price
        actions_share = actions_share - self.stocks
        actions = actions_share.astype(int)
        ######################################################################################################################
        self.day += 1
        price_end = self.price_ary[self.day]
        self.stocks_cool_down += 1

        if self.turbulence_bool[self.day] == 0:
            min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[index] > 0:  # Sell only if current asset is > 0
                    # sell_num_shares = min(self.stocks[index], -actions[index])
                    sell_num_shares = -actions[index]
                    self.stocks[index] -= sell_num_shares
                    self.amount += (price[index] * sell_num_shares * (1 - self.sell_cost_pct))
                    # print("self.sell_cost_pct",self.sell_cost_pct)
                    # self.amount += (price[index] * sell_num_shares )
                    self.stocks_cool_down[index] = 0
            for index in np.where(actions > min_action)[0]:  # buy_index:
                if (
                        price[index] > 0
                ):  # Buy only if the price is > 0 (no missing data in this particular date)
                    # buy_num_shares = min(self.amount // price[index], actions[index])
                    buy_num_shares = actions[index]
                    self.stocks[index] += buy_num_shares
                    self.amount -= (price[index] * buy_num_shares * (1 + self.buy_cost_pct))
                    # self.amount -= (price[index] * buy_num_shares)
                    self.stocks_cool_down[index] = 0

        else:  # sell all when turbulence
            self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cool_down[:] = 0

        if self.amount <= 0:
            self.amount = self.amount * (1 + (0.03/250))

        # print("做空成本不做考虑")
        for index in np.where(self.stocks < 0)[0]:  # sell_index:
            if price[index] > 0:  # Sell only if current asset is > 0
                    # sell_num_shares = min(self.stocks[index], -actions[index])
                self.amount -= (np.abs(price[index] * self.stocks[index]) * (0.03 / 250))

        # 上一期的投资组合同时作为state的一部分
        # 更新最新的投资组合向量
        # print("self.stocks",self.stocks)
        # print()
        self.presentaion_weights = (self.stocks * self.price_ary[self.day])/(self.amount + (self.stocks * price_end).sum())
        # print("self.presentaion_weights", self.presentaion_weights)
        #  print("===========================================================")
        self.last_action = W
        state, EIIE_input, future_return_based_on_closing = self.get_state(self.last_action)
        # print("last_action env",self.last_action )
        # print("W",W)
        # self.last_action = W
        # print("self.last_action ",self.last_action )
        # print("self.stocks:",self.stocks)
        total_asset = self.amount + (self.stocks * price_end).sum()
        # print("total_asset:",total_asset)
        ####################################################################################
        # reward = (total_asset - self.total_asset) * self.reward_scaling
        # reward = (total_asset-self.total_asset) /self.total_asset
        # reward = np.log2(total_asset/self.total_asset)
        # print("reward:",reward)
        # print("self.reward_scaling:",self.reward_scaling)
        # self.total_asset = total_asset
        # self.gamma_reward = self.gamma_reward * self.gamma + reward
        # self.horizen_reward+= reward
        # print("horizen_reward:",self.horizen_reward)
        ##############################################################################
        #print("self.day ", self.day)
        #print("self.price_ary[self.day - self.window_size + 2:self.day + 1]", self.price_ary[self.day - self.window_size + 2:self.day + 1].shape)
        #print("self.price_ary[self.day - self.window_size + 1:self.day]",self.price_ary[self.day - self.window_size + 1:self.day].shape)
        historical_return_vector = self.price_ary[self.day - self.window_size + 7:self.day + 1] / self.price_ary[self.day - self.window_size + 6:self.day]
        # log_x = (np.log2(historical_return_vector+1))
        log_x = np.log2(historical_return_vector)
        r = np.mean(log_x[-1:, :], axis=0)
        # print("Return vector env:", r)
        Convariance_matrix = np.cov(log_x, rowvar=False)
        weights = W
        # Mu = self.modified_calculation(r, Convariance_matrix, weights)
        # Modified_value = (1 / (2 * Mu))
        # print("Modified_value:",Modified_value)
        # print("actions:",W)
        # print("Convariance_matrix:",Convariance_matrix)
        # self.episode_expected_risk+= (Modified_value * self.variance(W, Convariance_matrix))[0,0]
        self.episode_expected_risk += 50 * self.variance(W, Convariance_matrix)[0, 0]
        # print("Modified risk:",(Modified_value * self.variance(W, Convariance_matrix))[0,0])
        # print("Accumulated Modified risk:",self.episode_expected_risk)
        # print("variance",self.variance(W, Convariance_matrix))
        ################################################################################
        reward = np.log2(total_asset / self.total_asset)
        # reward = np.log2(total_asset/self.total_asset)
        # print("reward:",reward)
        # reward = np.log2(total_asset/self.total_asset)
        # print("Logreturn:",np.log2(total_asset/self.total_asset))
        # print("Objective function:",reward)
        # reward = (total_asset-self.total_asset)*reward_scaling
        # print("Accumulated reward:",np.log2(total_asset/self.total_asset))
        # reward = np.log2(total_asset/self.total_asset) * self.reward_scaling
        self.total_asset = total_asset
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        self.horizen_reward += reward
        pure_reward = self.pure_reward_calculation(r, weights)
        # print("pure_reward:",pure_reward)
        # print("========================================")
        self.return_without_commission += pure_reward
        ##################################################################################################
        # self.return_modified_by_risk = np.log2(self.total_asset / self.initial_total_asset) - self.episode_expected_risk
        # self.return_risk_profit = np.log2(self.total_asset / self.initial_total_asset)  - self.episode_expected_risk
        ##################################################################################################
        done = self.day + 51 >= self.max_step
        """
        Experiment 1
                   self.day + 51
        Experiment 2
                  self.day + 49
        """
        # print("===================================================================")
        # print("self.day:",self.day)
        # print("self.max_step:",self.max_step)
        # print("done:",done)
        if done:
            # reward = self.gamma_reward
            horizen_reward = self.horizen_reward
            # print("horizen_reward:",horizen_reward )
            # 相对收益
            # self.episode_return = (self.total_asset-self.initial_total_asset) / self.initial_total_asset
            # 对数收益
            self.episode_return = np.log2(self.total_asset / self.initial_total_asset)
            # print("episode_return:",self.episode_return)
            # # print("total_asset:",self.total_asset)
            # print("initial_total_asset:",self.initial_total_asset)
            # expected risk
            self.eposide_return_variance = self.episode_expected_risk
            # logreturn-expected risk
            # self.return_risk_profit = np.log2(self.episode_return+1) - self.eposide_return_variance
            # self.return_risk_profit = self.episode_return - self.eposide_return_variance
            # return_risk_profit此时用于描述不考虑手续费的累计收益情况
            self.return_risk_profit = self.return_without_commission
            # print("return_risk_profit:",self.return_risk_profit)
        return state, EIIE_input, future_return_based_on_closing, reward, done, dict()

    def get_state(self, last_action):
        amount = np.array(self.amount * (2 ** -12), dtype=np.float32)
        scale = np.array(2 ** -6, dtype=np.float32)
        ################################################################################################
        # state_2d_p = np.hstack((self.price_ary[self.day - self.window_size + 3:self.day + 2]))
        state_2d_p = self.price_ary[self.day - self.window_size + 7:self.day + 1]
        state_3d_p = state_2d_p[:, :, np.newaxis]
        # state_close_p = np.hstack((self.price_ary[self.day - self.window_size + 2:self.day + 1]))
        state_close_p = self.price_ary[self.day - self.window_size + 6:self.day]
        state_close_reshape_p = state_close_p[:, :, np.newaxis]
        state_3d_normalization_p = state_3d_p / state_close_reshape_p
        # print("state_3d_normalization_p",state_3d_normalization_p.shape)
        # print("last_action",last_action.shape)
        # print("state_3d_normalization_p",state_3d_normalization_p.shape)
        # print("last_action[np.newaxis,: ,np.newaxis]", last_action[np.newaxis,: ,np.newaxis].shape)
        state = np.concatenate((last_action[np.newaxis,: ,np.newaxis], state_3d_normalization_p), axis=0)
        # print("state",state.shape)
        # print("state_3d_p:",state_3d_p)
        ###############################################################################################
        # state_2d = np.hstack((self.price_ary[self.day-self.window_size+1:self.day],self.tech_ary[self.day-self.window_size+1:self.day]))
        # state_3d = state_2d.reshape((33,10,9))
        # state_close = np.hstack((self.price_ary[self.day-self.window_size-1:self.day-1],self.tech_ary[self.day-self.window_size-1:self.day-1]))
        # state_close_reshape = state_close.reshape((33,10,9))
        # state.astype(np.float32)
        # state_3d_normalization = state_3d/state_close_reshape
        # print("state_3d_normalization:", state_3d_normalization)
        # print("state_3d:", state_3d)
        OCLC_ary_state_2d = self.OCLC_ary[self.day - self.window_size + 7:self.day + 1]
        OCLC_ary_state_3d = OCLC_ary_state_2d.reshape(50, 29, 3)
        close_in_OCLC_ary_state_2d = self.close_in_OCLC_array[self.day]
        # print("close_in_OCLC_ary_state_2d", self.close_in_OCLC_array.shape)
        # print("self.OCLC_ary", self.OCLC_ary.shape)
        close_OCLC_ary_state_3d = close_in_OCLC_ary_state_2d[np.newaxis, :, np.newaxis]
        EIIE_input = OCLC_ary_state_3d/close_OCLC_ary_state_3d
        last_action = np.concatenate((last_action[np.newaxis, :, np.newaxis], np.zeros([1, 29, 2])), axis=2)
        # print("last_action", last_action.shape)
        # print("EIIE_input", EIIE_input.shape)
        state_EIIE = np.concatenate((last_action, EIIE_input), axis=0)
        # print("OCLC_ary_state_2d", OCLC_ary_state_2d.shape)
        ##############################################################################################
        future_return_based_on_closing = self.close_in_OCLC_array[self.day + 1]/self.close_in_OCLC_array[self.day]
        future_return_based_on_closing = future_return_based_on_closing[np.newaxis, :, np.newaxis]
        # state_2d_tech = self.tech_ary[self.day - self.window_size + 7:self.day + 6]
        # # print("state_2d_tech", state_2d_tech.shape)
        # state_3d_tech = state_2d_tech.reshape(205, 29, 5)
        # state_2d_tech_last = self.tech_ary[self.day - self.window_size + 6:self.day + 5]
        # # print("state_2d_tech", state_2d_tech.shape)
        # state_3d_tech_last = state_2d_tech_last.reshape(205, 29, 5)
        # state_3d_tech_normalized = state_3d_tech/state_3d_tech_last
        # # print("state_3d_tech_normalized",state_3d_tech_normalized)
        # state_3d_tech = np.concatenate((torch.zeros([1,29,5]),state_3d_tech_normalized), axis=0)
        # # print("state_3d_tech", state_3d_tech.shape)
        # state_with_tech = np.concatenate((state, state_3d_tech), axis=2)
        # print("state_with_tech",state_with_tech.shape)
        return state, state_EIIE, future_return_based_on_closing

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
