from typing import Optional, Union, List
import gym
# from gym.core import RenderFrame
from numpy import random as rd
import numpy as np


class StockTradingEnv(gym.Env):
    # def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
    #     pass

    def __init__(
            self,
            config,
            initial_account=1e6,
            gamma=0.99,
            turbulence_thresh=99,
            min_stock_rate=0.1,
            max_stock=1e2,
            initial_capital=1e6,
            buy_cost_pct=5e-4,
            sell_cost_pct=5e-4,
            reward_scaling=2 ** -11,
            initial_stocks=None,
    ):

        self.window_size = 200
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        turbulence_ary = config["turbulence_array"]
        if_train = config["if_train"]
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary

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
        self.day = self.window_size-5
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = "StockEnv"
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.stock_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    # def variance_calculation(self, actions, covariance_matrix):
    #     actions = np.array(actions)  # 转换为NumPy数组
    #
    #     # 计算协方差矩阵的特征值和特征向量
    #     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    #
    #     # 选择特征值最大的一个特征向量
    #     max_eigenvalue_index = np.argmax(eigenvalues)
    #     max_eigenvector = eigenvectors[:, max_eigenvalue_index]
    #
    #     # 计算降维后的方差（将协方差矩阵投影到特征向量上）
    #     variance = np.dot(max_eigenvector, np.dot(covariance_matrix, max_eigenvector))
    #
    #     return float(variance)  # 返回方差值的标量

    def variance_calculation(self, actions, Convariance_matrix):
        Convariance_matrix = np.matrix(Convariance_matrix)
        actions = np.matrix(actions)
        # print("actions", actions.shape)
        # print("Convariance_matrix", Convariance_matrix.shape)
        variance = actions * Convariance_matrix * actions.T
        # print("variance:", variance)
        return variance

    def reset(self):
        self.day = self.window_size
        price = self.price_ary[self.day]

        if self.if_train:
            self.stocks = (
                    self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = (
                    self.initial_capital * rd.uniform(0.95, 1.05)
                    - (self.stocks * price).sum()
            )
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price)  # state

    def beginning(self):
        self.day = self.window_size
        price = self.price_ary[self.day]
        if self.if_train:
            # 随机起始资产 self.stocks = (self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)).astype(
            # np.float32)
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            # 随机起始资产 self.amount = (self.initial_capital * rd.uniform(0.95, 1.05) - (self.stocks * price).sum())
            self.amount_test = (self.initial_capital - (self.stocks * price).sum())
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount_test = self.initial_capital
        self.total_asset_test = self.amount_test + (self.stocks * price).sum()
        self.initial_total_asset_test = self.total_asset_test
        self.gamma_reward_test = 0.0
        self.total_return_test = 0.0
        self.episode_return_test = 0.0
        return self.day, self.initial_total_asset_test, self.episode_return_test

    def step(self, actions):
        action_softmax = actions
        # actions = (actions * self.max_stock).astype(int)
        """根据投资组合权重计算每个股票持有的数额"""
        number_of_stocks = ((self.total_asset * actions) / self.price_ary[self.day]).astype(int)
        """每只股票当前需要买卖的数额"""
        actions = number_of_stocks - self.stocks
        price = self.price_ary[self.day]
        self.stocks_cool_down += 1
        """执行股票买卖"""
        if self.turbulence_bool[self.day] == 0:
            min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[index] > 0:  # Sell only if current asset is > 0
                    sell_num_shares = min(self.stocks[index], -actions[index])
                    self.stocks[index] -= sell_num_shares
                    self.amount += (
                            price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                    )
                    self.stocks_cool_down[index] = 0
            for index in np.where(actions > min_action)[0]:  # buy_index:
                if (
                        price[index] > 0
                ):  # Buy only if the price is > 0 (no missing data in this particular date)
                    buy_num_shares = min(self.amount // price[index], actions[index])
                    self.stocks[index] += buy_num_shares
                    self.amount -= (
                            price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                    )
                    self.stocks_cool_down[index] = 0

        else:  # sell all when turbulence
            self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cool_down[:] = 0
        '''进入下一个交易日'''
        self.day += 1
        """获得当前环境状态"""
        state = self.get_state(price)

        """计算总资产数额"""
        total_asset = self.amount + (self.stocks * self.price_ary[self.day]).sum()
        """资源投资组合的相对收益"""
        return_rate = (total_asset - self.total_asset) / self.total_asset
        """获得相对收益向量"""
        # returns = (price - price[0]) / price[0]
        """取得过去30个交易日的价格向量"""
        """相对价格向量"""
        price_array = self.price_ary[self.day - 30: self.day + 1] / self.price_ary[self.day - 30 - 1: self.day] - 1
        covariance_matrix = np.var(price_array)
        """对数收益价格向量"""
        price_array_log = np.log2(
            self.price_ary[self.day - 29: self.day + 1] / self.price_ary[self.day - 29 - 1: self.day])
        # print("price_array_log", price_array_log.shape)
        """计算方差写方差句子应该使用np.cov"""
        covariance_matrix = np.cov(price_array_log.T)
        # variance = np.var(price_array)
        """基于资产的变化计算reward"""
        # reward = (total_asset - self.total_asset) * self.reward_scaling
        # reward = np.log2(total_asset / self.total_asset) - variance
        """完成variance计算之后，公式self.variance的输出只依然是一个二维张量，需要去张量第一行第一列的值"""
        # variance = self.variance_calculation(actions, covariance_matrix)
        variance = self.variance_calculation(actions, covariance_matrix)[0, 0]

        reward = return_rate - 0.01 * variance
        # 或 reward = return_rate - lambda * variance,lambda为风险厌恶系数
        """
        基于对数收益计算reward
        reward = np.log2(total_asset / self.total_asset)
        基于相对收益计算reward
        reward = (total_asset / self.total_asset)-1
        基于收益和方差计算reward
        variance = self.variance_calculation(action_softmax, covariance_matrix)
        reward = np.log2(total_asset / self.total_asset) - variance
        # CAPM模型计算expected return
        rf = 0.03 # 无风险收益率 
        beta = 0.8 # 股票的β系数
        market_return = 0.05 # 市场收益率
        expected_return = rf + beta*(market_return - rf)  
        # 基于CAPM模型计算reward
        reward = np.log(total_asset/self.total_asset) - expected_return
        # 三因子模型计算expected return
        rf = 0.03 
        mkt_risk_premium = 0.05 - rf
        smb = 0.04 # 小市值股票收益率
        hml = 0.03 # 高账面市值比股票收益率
        expected_return = rf + beta*mkt_risk_premium + s*smb + h*hml
        # 基于三因子模型计算reward  
        reward = np.log(total_asset/self.total_asset) - expected_return
        """

        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = np.log2(total_asset / self.initial_total_asset)

        return state, reward, done, dict()

    def back_test_step(self, actions):
        action_softmax = actions
        # actions = (actions * self.max_stock).astype(int)
        """根据投资组合权重计算每个股票持有的数额"""
        number_of_stocks = ((self.total_asset * actions) / self.price_ary[self.day]).astype(int)
        """每只股票当前需要买卖的数额"""
        actions = number_of_stocks - self.stocks
        price = self.price_ary[self.day]
        self.stocks_cool_down += 1
        """执行股票买卖"""
        if self.turbulence_bool[self.day] == 0:
            min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[index] > 0:  # Sell only if current asset is > 0
                    sell_num_shares = min(self.stocks[index], -actions[index])
                    self.stocks[index] -= sell_num_shares
                    self.amount += (
                            price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                    )
                    self.stocks_cool_down[index] = 0
            for index in np.where(actions > min_action)[0]:  # buy_index:
                if (
                        price[index] > 0
                ):  # Buy only if the price is > 0 (no missing data in this particular date)
                    buy_num_shares = min(self.amount // price[index], actions[index])
                    self.stocks[index] += buy_num_shares
                    self.amount -= (
                            price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                    )
                    self.stocks_cool_down[index] = 0

        else:  # sell all when turbulence
            self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cool_down[:] = 0
        '''进入下一个交易日'''
        self.day += 1
        """获得当前环境状态"""
        state = self.get_state(price)
        # actions = (actions * self.max_stock).astype(int)
        #
        # self.day += 1
        # price = self.price_ary[self.day]
        # self.stocks_cool_down += 1
        #
        # if self.turbulence_bool[self.day] == 0:
        #     min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
        #     for index in np.where(actions < -min_action)[0]:  # sell_index:
        #         if price[index] > 0:  # Sell only if current asset is > 0
        #             sell_num_shares = min(self.stocks[index], -actions[index])
        #             self.stocks[index] -= sell_num_shares
        #             self.amount += (
        #                     price[index] * sell_num_shares * (1 - self.sell_cost_pct)
        #             )
        #             self.stocks_cool_down[index] = 0
        #     for index in np.where(actions > min_action)[0]:  # buy_index:
        #         if (
        #                 price[index] > 0
        #         ):  # Buy only if the price is > 0 (no missing data in this particular date)
        #             buy_num_shares = min(self.amount // price[index], actions[index])
        #             self.stocks[index] += buy_num_shares
        #             self.amount -= (
        #                     price[index] * buy_num_shares * (1 + self.buy_cost_pct)
        #             )
        #             self.stocks_cool_down[index] = 0
        #
        # else:  # sell all when turbulence
        #     self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
        #     self.stocks[:] = 0
        #     self.stocks_cool_down[:] = 0
        #
        # state = self.get_state(price)
        """计算总资产数额"""
        total_asset = self.amount + (self.stocks * self.price_ary[self.day]).sum()
        """资源投资组合的相对收益"""
        return_rate = (total_asset - self.total_asset) / self.total_asset
        """获得相对收益向量"""
        # returns = (price - price[0]) / price[0]
        """取得过去30个交易日的价格向量"""
        """相对价格向量"""
        price_array = self.price_ary[self.day - 30: self.day + 1] / self.price_ary[self.day - 30 - 1: self.day] - 1
        covariance_matrix = np.var(price_array)
        """对数收益价格向量"""
        price_array_log = np.log2(
            self.price_ary[self.day - 29: self.day + 1] / self.price_ary[self.day - 29 - 1: self.day])
        # print("price_array_log", price_array_log.shape)
        """计算方差写方差句子应该使用np.cov"""
        covariance_matrix = np.cov(price_array_log.T)
        # variance = np.var(price_array)
        """基于资产的变化计算reward"""
        # reward = (total_asset - self.total_asset) * self.reward_scaling
        # reward = np.log2(total_asset / self.total_asset) - variance
        """完成variance计算之后，公式self.variance的输出只依然是一个二维张量，需要去张量第一行第一列的值"""
        # variance = self.variance_calculation(actions, covariance_matrix)
        variance = self.variance_calculation(actions, covariance_matrix)[0, 0]

        reward = return_rate - 0.05 * variance
        # 或 reward = return_rate - lambda * variance,lambda为风险厌恶系数
        """
        基于对数收益计算reward
        reward = np.log2(total_asset / self.total_asset)
        基于相对收益计算reward
        reward = (total_asset / self.total_asset)-1
        基于收益和方差计算reward
        variance = self.variance_calculation(action_softmax, covariance_matrix)
        reward = np.log2(total_asset / self.total_asset) - variance
        # CAPM模型计算expected return
        rf = 0.03 # 无风险收益率 
        beta = 0.8 # 股票的β系数
        market_return = 0.05 # 市场收益率
        expected_return = rf + beta*(market_return - rf)  
        # 基于CAPM模型计算reward
        reward = np.log(total_asset/self.total_asset) - expected_return
        # 三因子模型计算expected return
        rf = 0.03 
        mkt_risk_premium = 0.05 - rf
        smb = 0.04 # 小市值股票收益率
        hml = 0.03 # 高账面市值比股票收益率
        expected_return = rf + beta*mkt_risk_premium + s*smb + h*hml
        # 基于三因子模型计算reward  
        reward = np.log(total_asset/self.total_asset) - expected_return
        """

        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day + 49 == self.max_step
        """
        Experiment 2021
                    self.day+51
        Experiment 2023
                    self.day+49
        """
        if done:
            reward = self.gamma_reward
            self.episode_return = np.log2(total_asset / self.initial_total_asset)
        ################################################################
        # total_asset = self.amount + (self.stocks * price).sum()
        # reward = (total_asset - self.total_asset) * self.reward_scaling
        # self.total_asset = total_asset
        #
        # self.gamma_reward = self.gamma_reward * self.gamma + reward
        # done = self.day + 8 >= self.max_step
        # if done:
        #     reward = self.gamma_reward
        #     self.episode_return = np.log2(total_asset / self.initial_total_asset)

        return state, reward, done, dict()

    def get_state(self, price):
        """过去的收益数据"""
        state_2d_price_historical1 = self.price_ary[self.day - self.window_size + 1:self.day]
        # print("state_2d_p", state_2d_price_historical1.shape)
        state_3d_price_historical1 = state_2d_price_historical1[:, :, np.newaxis]

        state_2d_price_historical2 = self.price_ary[self.day - self.window_size:self.day - 1]
        state_3d_price_historical2 = state_2d_price_historical2[:, :, np.newaxis]
        state_3d_return_historical = (state_3d_price_historical1 / state_3d_price_historical2 - 1) * 100
        """当前的收益数据"""
        state_2d_price_current = self.price_ary[self.day]
        # print("state_2d_price_current", state_2d_price_current.shape)
        # state_3d_price_current = state_2d_price_current.reshape(1, 29, 5)
        state_3d_price_current = state_2d_price_current[np.newaxis, :, np.newaxis]

        state_2d_price_current_before = self.price_ary[self.day - 1]
        # state_3d_price_current_before = state_2d_price_current_before.reshape(1, 29, 5)
        state_3d_price_current_before = state_2d_price_current_before[np.newaxis, :, np.newaxis]
        state_3d_return_current = (state_3d_price_current / state_3d_price_current_before - 1) * 100
        # print("state_3d_return_historical", state_3d_return_historical.shape)
        # print("state_3d_return_current", state_3d_return_current.shape)
        state_with_price_array = np.concatenate((state_3d_return_historical, state_3d_return_current), axis=0)

        return state_with_price_array


    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
