import numpy as np
import pandas as pd
DOW_30_TICKER_10 = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON"
]
DOW_30_TICKER = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WBA",
    "WMT",
    "DIS",
]
TRAIN_START_DATE = '2020-01-01'
TRAIN_END_DATE =  '2022-12-31'
TECHNICAL_INDICATORS_LIST = [
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "dx_30",
    "close_30_sma"
]
ERL_PARAMS = {
    "learning_rate": 1e-5,
    "batch_size": 128,
    "gamma": 0.9,
    "seed": 0,
    "net_dimension": 512,
    "target_step": 701,
    "eval_gap": 30
}

import ray
kwargs = {"start_date": TRAIN_START_DATE,
"end_date": TRAIN_END_DATE,
"ticker_list": DOW_30_TICKER,
"time_interval": '1D',
"technical_indicator_list": TECHNICAL_INDICATORS_LIST,
"drl_lib": "elegantrl",
"model_name": "ddpg",
"cwd": "./test_ddpg",
"erl_params": ERL_PARAMS,
"break_step": 3e5}

from DataProcessor import DataProcessor
DP = DataProcessor(data_source = "yahoofinance",**kwargs)

# 确定下载数据的超参数参数
# region Description
start_date= TRAIN_START_DATE
end_date= TRAIN_END_DATE
ticker_list=DOW_30_TICKER
data_source="yahoofinance"
time_interval="1D"
technical_indicator_list=TECHNICAL_INDICATORS_LIST
drl_lib="elegantrl"
# env=env,
model_name="ddpg"
cwd="./test_ddpg"
erl_params=ERL_PARAMS
break_step = 3e5
"""====================================================================================================="""


# 构建环境样例
from StockTradingEnv import StockTradingEnv
env = StockTradingEnv
env_config = np.load('dict_back_test_env_config.npy', allow_pickle=True).item()
env_instance_train = env(config=env_config)


##################################################
# from Main import *
from DataProcessor import DataProcessor
# 回测参数确定
TEST_START_DATE = '2022-10-20'
TEST_END_DATE = '2023-12-31'
DOW_30_TICKER_10 = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON"]
DOW_30_TICKER = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WBA",
    "WMT",
    "DIS",
]
TECHNICAL_INDICATORS_LIST = [
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "dx_30",
    "close_30_sma"]


###回测环境参量输出
# region Description
from StockTradingEnv import StockTradingEnv
env = StockTradingEnv
# 输出需要观察的环境参数
start_date=TEST_START_DATE
end_date=TEST_END_DATE
ticker_list=DOW_30_TICKER
data_source="yahoofinance"
time_interval="1D"
technical_indicator_list=TECHNICAL_INDICATORS_LIST
drl_lib="elegantrl"
env=env
model_name="ddpg"
cwd="./test_ddpg"
net_dimension=512
###
# endregion

import ray
kwargs ={"start_date": TEST_START_DATE,
         "end_date": TEST_END_DATE,
         "ticker_list": DOW_30_TICKER,
         "time_interval": "1D",
         "technical_indicator_list": TECHNICAL_INDICATORS_LIST,
         "drl_lib": "elegantrl",
         "env": env,
         "model_name": "ddpg",
         "cwd": "./test_ddpg",
         "net_dimension": 512}
"""====================================================================================================="""

env_test_config = np.load('dict_back_test_env_test_config.npy', allow_pickle=True).item()

env_instance_test = env(config=env_test_config)
net_dimension = kwargs.get("net_dimension", 2 ** 8)
print("net_dimension:",net_dimension)
print("model_name:",model_name)
cwd = kwargs.get("cwd", "./" + str(model_name))
print("cwd:",cwd)
# print("price_array_test: ", len(price_array_test))

#################################################################################################
#################################################################################################
from Model import DRLAgent
DRLAgent_erl = DRLAgent

if drl_lib == "elegantrl":
    episode_total_assets_DDPG, episode_returns_DDPG, episode_returns_modified_by_risk_DDPG, episode_total_assets_DDPG_melt, episode_returns_DDPG_melt, episode_returns_modified_by_risk_DDPG_melt = DRLAgent_erl.DRL_prediction(
        model_name = model_name,
        cwd = cwd,
        net_dimension = net_dimension,
        environment = env_instance_test,
        environment_add = env_instance_train)
print("episode_returns_DDPG_melt", episode_returns_DDPG_melt)
print("episode_returns_DDPG_melt_length", len(episode_returns_DDPG_melt))

episode_total_assets_DDPG_test_plot = np.array(episode_total_assets_DDPG)
episode_returns_DDPG_test = np.array(episode_returns_DDPG)
episode_returns_modified_by_risk_DDPG_test = np.array(episode_returns_modified_by_risk_DDPG)

episode_total_assets_DDPG_test_plot_melt = np.array(episode_total_assets_DDPG_melt)
episode_returns_DDPG_test_melt = np.array(episode_returns_DDPG_melt)
episode_returns_modified_by_risk_DDPG_test_melt = np.array(episode_returns_modified_by_risk_DDPG_melt)

episode_total_assets_DDPG_train, episode_returns_DDPG_train, episode_returns_modified_by_risk_DDPG_train, episode_total_assets_DDPG_train_melt, episode_returns_DDPG_train_melt, episode_returns_modified_by_risk_DDPG_train_melt = DRLAgent_erl.DRL_prediction(
        model_name = model_name,
        cwd = cwd,
        net_dimension = net_dimension,
        environment = env_instance_train,
        environment_add = env_instance_test)
episode_total_assets_DDPG_train_plot = np.array(episode_total_assets_DDPG_train)
episode_returns_DDPG_train = np.array(episode_returns_DDPG_train)
episode_returns_modified_by_risk_DDPG_train = np.array(episode_returns_modified_by_risk_DDPG_train)
# episode_total_assets_DDPG_train_melt, episode_returns_DDPG_train_melt, episode_returns_modified_by_risk_DDPG_train_melt
episode_total_assets_DDPG_train_plot_melt = np.array(episode_total_assets_DDPG_train_melt)
episode_returns_DDPG_train_melt = np.array(episode_returns_DDPG_train_melt)
episode_returns_modified_by_risk_DDPG_train_melt = np.array(episode_returns_modified_by_risk_DDPG_train_melt)

print("================1. Step in the training set===================")
# region Description
env_Date = StockTradingEnv
environment_instance_test_Date = env_Date(config=env_config)
environment_Date = environment_instance_test_Date
number_of_stocks = 29
Date_train = list()
Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_Date.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
Date_train.append(Date - 51)
# initial_portfolio_weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_Date.max_step):
    # make portfolio decision
    portfolio_weights = last_portfolio_weights
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_Date.backtest_step(portfolio_weights)
    # print(Date-201, "|  ", eposide_total_asset)
    last_portfolio_weights = portfolio_weights
    Date_train.append(Date - 51)
    if done:
        print("Date_train", Date_train)
        print("Test Finished!")
        break

# print("Date_train", len(Date_train))
# print("episode_returns_DDPG_train",len(episode_returns_DDPG_train))

# Draw plot
# 由于agent的策略频率为周频率，需要将收益变化序列的日期序列进行对齐
print("len(episode_returns_DDPG_train)", len(episode_returns_DDPG_train))
if len(Date_train) >= len(episode_returns_DDPG_train):
    episode_returns_DDPG_train = np.concatenate((episode_returns_DDPG_train, episode_returns_DDPG_train[-1] * np.ones(len(Date_train)-len(episode_returns_DDPG_train))), axis = 0)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
save_title='accumulated reward in training set'
plt.figure(figsize=(10, 10), dpi=3000)
# fig = plt.figure(dpi=100,constrained_layout=True) #类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
# gs = GridSpec(2, 2, figure=fig) #GridSpec将fiure分为3行3列，每行三个axes，gs为一个matplotlib.gridspec.GridSpec对象，可灵活的切片figure
fig, axs = plt.subplots(2, figsize=(15, 10), dpi=100)
# axs[0]
ax000 = axs[0]
ax000.cla()
color1 = 'darkcyan'
ax000.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax000.tick_params(axis='x', labelsize=12)
color000 = 'lightcoral'
ax000.set_ylabel('Episode Return', fontsize=15)
ax000.plot(Date_train, episode_returns_DDPG_train, label='logreturn', color=color000, lw=1.5)
# ax00.plot(Date_train, episode_returns_modified_by_risk_DDPG_train, label='logreturn modified risk', color=color000, lw=1.5)
# ax000.legend(loc='center left')
ax000.grid()

# axs[1]
ax0001 = axs[1]
ax0001.cla()
daily_return_train = np.log2(episode_total_assets_DDPG_train_plot[1:]/episode_total_assets_DDPG_train_plot[:-1])
# print("daily_return_train:",daily_return_train)
# bins = range(-30, 30, 1000)
cm = plt.cm.get_cmap('Greens')
# n, bins, patches = ax0001.hist(daily_return_train*100, 500, density=True, cumulative=-1, color='green', histtype='bar', label='logreturn modified risk')
n, bins, patches = ax0001.hist(daily_return_train, 500, density=True, color='green', histtype='bar', label='logreturn')
ax0001.set_xlim(-30, 30)
for c, p in zip(n, patches):
    plt.setp(p, 'facecolor', cm(1-c))
'''
ax0001.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax0001.tick_params(axis='x', labelsize=12)
color000 = 'lightcoral'
ax0001.set_ylabel('Number of days', fontsize=15)
ax0001.hist(daily_return_train, bins, label='Daily return', color=color000)
# ax0001.hist(daily_return_train, 1, label='Daily return')
# ax00.plot(Date_train, episode_returns_modified_by_risk_DDPG_train, label='logreturn modified risk', color=color000, lw=1.5)
'''
ax0001.legend(loc='center left')
ax0001.grid()


plt.title(save_title, y=13.5, verticalalignment='bottom')
# plt.title(save_title,verticalalignment='bottom')
fig_name = 'performance_in_training_set.jpg'
plt.savefig(f"{cwd}/{fig_name}")
plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
# plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()
# endregion

print("================2. Step in the test set===================")
# region Description
env_Date = StockTradingEnv
environment_instance_test_Date = env_Date(config=env_test_config)
environment_Date = environment_instance_test_Date
number_of_stocks = 29
Date_test = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_Date.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
Date_test.append(Date - 51)
# initial_portfolio_weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_Date.max_step):
    # make portfolio decision
    portfolio_weights = last_portfolio_weights
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_Date.backtest_step(portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)
    last_portfolio_weights = portfolio_weights
    Date_test.append(Date - 51)
    if done:
        print("Test Finished!")
        # print("eposide_total_asset",eposide_total_asset)
        # print("eposide_return:",eposide_return)
        print("Date_test", Date_test)
        break

# Draw plot
# print("=============================Performance of EIIE agent===================================")
# print("len(episode_returns_DDPG_test", len(episode_returns_DDPG_test))
# print("len(Date_test)", len(Date_test))
if len(Date_test) >= len(episode_returns_DDPG_test):
    episode_returns_DDPG_test = np.concatenate((episode_returns_DDPG_test, episode_returns_DDPG_test[-1] * np.ones(len(Date_test)-len(episode_returns_DDPG_test))), axis = 0)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
save_title='accumulated reward in test set'
plt.figure(figsize=(10, 10), dpi=3000)
# fig = plt.figure(dpi=100,constrained_layout=True) #类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
# gs = GridSpec(2, 2, figure=fig) #GridSpec将fiure分为3行3列，每行三个axes，gs为一个matplotlib.gridspec.GridSpec对象，可灵活的切片figure
fig, axs = plt.subplots(2, figsize=(15, 10), dpi=100)

# axs[0]
ax001 = axs[0]
ax001.cla()
color1 = 'darkcyan'
ax001.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax001.tick_params(axis='x', labelsize=12)
color000 = 'lightcoral'
ax001.set_ylabel('Episode Return', fontsize=15)
ax001.plot(Date_test, episode_returns_DDPG_test, label='logreturn', color=color000, lw=1.5)
# ax00.plot(Date_train, episode_returns_modified_by_risk_DDPG_train, label='logreturn modified risk', color=color000, lw=1.5)
# ax001.legend(loc='center left')
ax001.grid()
plt.title(save_title, y=13.5, verticalalignment='bottom')

# axs[1]
ax0011 = axs[1]
ax0011.cla()
daily_return_test = np.log(episode_total_assets_DDPG_test_plot[1:]/episode_total_assets_DDPG_test_plot[:-1])
# print("daily_return_train:",daily_return_train)
# bins = range(-30, 30, 1000)
cm = plt.cm.get_cmap('Greens')
# n, bins, patches = ax0001.hist(daily_return_train*100, 500, density=True, cumulative=-1, color='green', histtype='bar', label='logreturn modified risk')
n, bins, patches = ax0011.hist(daily_return_test, 60, density=True, color='green', histtype='bar', label='logreturn')
ax0011.set_xlim(-30, 30)
for c, p in zip(n, patches):
    plt.setp(p, 'facecolor', cm(1-c))

ax0011.legend(loc='center left')
ax0011.grid()

# plt.title(save_title,verticalalignment='bottom')
fig_name = 'performance_in_test_set.jpg'
plt.savefig(f"{cwd}/{fig_name}")
plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
# plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()
# endregion

#######################################################################################
def generate_history_matrix(information):
    inputs = information.T
    # print(inputs.shape)
    # inputs = np.concatenate([np.ones([1, inputs.shape[1]]), inputs], axis=0)
    inputs = inputs[:, 1:] / inputs[:, :-1]
    return inputs
####################################################################################
# # region Description
# from CRP import CRP
# env_CRP = StockTradingEnv
# environment_instance_test_CRP = env_CRP(config=env_test_config)
# environment_CRP = environment_instance_test_CRP
# window_size = 50
# number_of_stocks = 29
# ###############################
# agent = CRP()
# ###################################
# eposide_return_modified_by_risk_list_Average_CRP = list()
# eposide_total_asset1_list_Average_CRP = list()
# eposdie_return_list_Average_CRP = list()
#
# Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_CRP.beginning()
# # print("eposide_total_asset:", eposide_total_asset)
# # print("eposide_return:", eposide_return)
# eposdie_return_list_Average_CRP.append(eposide_return)
# eposide_return_modified_by_risk_list_Average_CRP.append(eposide_return_modified_by_risk)
# eposide_total_asset1_list_Average_CRP.append(eposide_total_asset)
# initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
# last_portfolio_weights = initial_portfolio_weights
# for i in range(environment_CRP.max_step):
#     information = generate_history_matrix(environment_CRP.collecting(Date, window_size))
#     information = information[np.newaxis, :, :]
#
#     # make portfolio decision
#
#     portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
#     eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_CRP.backtest_step(
#         portfolio_weights)
#     # print(Date-32, "|  ", eposide_total_asset)
#
#     last_portfolio_weights = portfolio_weights
#     eposdie_return_list_Average_CRP.append(eposide_return)
#     eposide_return_modified_by_risk_list_Average_CRP.append(eposide_return_modified_by_risk)
#     eposide_total_asset1_list_Average_CRP.append(eposide_total_asset)
#     if done:
#         # print("Test Finished!")
#         # print("eposide_total_asset:", eposide_total_asset)
#         # print("eposide_return:", eposide_return)
#         break
#######################################################################################

print("================3. Traditional strategies--CRP in the backtest====================")
# region Description
from CRP import CRP
env_CRP = StockTradingEnv
environment_instance_test_CRP = env_CRP(config=env_test_config)
environment_CRP = environment_instance_test_CRP
window_size = 50
number_of_stocks = 29
###############################
agent = CRP()
###################################
eposide_return_modified_by_risk_list_Average_CRP = list()
eposide_total_asset1_list_Average_CRP = list()
eposdie_return_list_Average_CRP = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_CRP.beginning()
eposdie_return_list_Average_CRP.append(eposide_return)
eposide_return_modified_by_risk_list_Average_CRP.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_CRP.append(eposide_total_asset)
initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_CRP.max_step):
    information = generate_history_matrix(environment_CRP.collecting(Date, window_size))
    information = information[np.newaxis, :, :]
    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_CRP.backtest_step(
        portfolio_weights)
    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_CRP.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_CRP.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_CRP.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset1_list_Average_CRP", eposide_total_asset1_list_Average_CRP)
        print("length of eposide_total_asset1_list_Average_CRP", len(eposide_total_asset1_list_Average_CRP))
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

"""

"""
# 运行对比策略
'''
from BK import BK
策略1： BK
from CRP import CRP
策略2： CRP
from ONS import ONS
策略3： ONS
from OLMAR import OLMAR
策略4： OLMAR
from UP import UP 
策略5： UP
from ANTICOR import ANTICOR1
from PAMR import PAMR
from CORNK import CORNK
from M0 import M0
from RMR import RMR
from CWMR import CWMR_STD
from EG import EG
from SP import SP
from UBAH import UBAH
from WHAMR import WMAMR
全局最优策略
from BCRP import BCRP
from BEST import BEST
'''

"""
def generate_history_matrix(information):
    inputs = information.T
    # print(inputs.shape)
    # inputs = np.concatenate([np.ones([1, inputs.shape[1]]), inputs], axis=0)
    inputs = inputs[:, 1:] / inputs[:, :-1]
    return inputs


print("================1. Traditional strategies--BK in the backtest====================")
# region Description
from BK import BK
env_BK = StockTradingEnv
environment_instance_test_BK = env_BK(config=env_test_config)
environment_BK = environment_instance_test_BK
window_size = 200
number_of_stocks = 29
agent = BK()
eposdie_return_list_Average_BK = list()
eposide_return_modified_by_risk_list_Average_BK = list()
eposide_total_asset1_list_Average_BK = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_BK.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_BK.append(eposide_return)
eposide_return_modified_by_risk_list_Average_BK.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_BK.append(eposide_total_asset)

initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_BK.max_step):
    information = generate_history_matrix(environment_BK.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights(BK):", portfolio_weights)
    # print("stock_value_weights(BK):", portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_BK.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_BK.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_BK.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_BK.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================2. Traditional strategies--CRP in the backtest====================")
# region Description
from CRP import CRP
env_CRP = StockTradingEnv
environment_instance_test_CRP = env_CRP(config=env_test_config)
environment_CRP = environment_instance_test_CRP
window_size = 200
number_of_stocks = 29
###############################
agent = CRP()
###################################
eposide_return_modified_by_risk_list_Average_CRP = list()
eposide_total_asset1_list_Average_CRP = list()
eposdie_return_list_Average_CRP = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_CRP.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:", eposide_return)
eposdie_return_list_Average_CRP.append(eposide_return)
eposide_return_modified_by_risk_list_Average_CRP.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_CRP.append(eposide_total_asset)
initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_CRP.max_step):
    information = generate_history_matrix(environment_CRP.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_CRP.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_CRP.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_CRP.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_CRP.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================3. Traditional strategies--ONS in the backtest====================")
# region Description
from ONS import ONS
env_ONS = StockTradingEnv
environment_instance_test_ONS = env_ONS(config=env_test_config)
environment_ONS = environment_instance_test_ONS
window_size = 200
number_of_stocks = 29
###############################
agent = ONS()
###################################
eposide_return_modified_by_risk_list_Average_ONS = list()
eposide_total_asset1_list_Average_ONS = list()
eposdie_return_list_Average_ONS = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_ONS.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_ONS.append(eposide_return)
eposide_return_modified_by_risk_list_Average_ONS.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_ONS.append(eposide_total_asset)
# initial_portfolio_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_portfolio_weights = np.ones(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_ONS.max_step):
    information = generate_history_matrix(environment_ONS.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision
    # print(information.shape)
    # print(last_portfolio_weights.shape)
    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_ONS.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_ONS.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_ONS.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_ONS.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================4. Traditional strategies--OLMAR in the backtest====================")
# region Description
from OLMAR import OLMAR
env_OLMAR = StockTradingEnv
environment_instance_test_OLMAR = env_OLMAR(config=env_test_config)
environment_OLMAR = environment_instance_test_OLMAR
window_size = 200
number_of_stocks = 29
###############################
agent = OLMAR()
###################################
eposide_return_modified_by_risk_list_Average_OLMAR = list()
eposide_total_asset1_list_Average_OLMAR = list()
eposdie_return_list_Average_OLMAR = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_OLMAR.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_OLMAR.append(eposide_return)
eposide_return_modified_by_risk_list_Average_OLMAR.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_OLMAR.append(eposide_total_asset)
# initial_portfolio_weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_OLMAR.max_step):
    information = generate_history_matrix(environment_OLMAR.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_OLMAR.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_OLMAR.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_OLMAR.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_OLMAR.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================5. Traditional strategies--UP in the backtest====================")
# region Description
from UP import UP
env_UP = StockTradingEnv
environment_instance_test_UP = env_UP(config=env_test_config)
environment_UP = environment_instance_test_UP
window_size = 200
number_of_stocks = 29
###############################
agent = UP()
###################################
eposide_return_modified_by_risk_list_Average_UP = list()
eposide_total_asset1_list_Average_UP = list()
eposdie_return_list_Average_UP = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_UP.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_UP.append(eposide_return)
eposide_return_modified_by_risk_list_Average_UP.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_UP.append(eposide_total_asset)
# initial_portfolio_weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_UP.max_step):
    information = generate_history_matrix(environment_UP.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_UP.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_UP.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_UP.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_UP.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================6. Traditional strategies--Anticor in the backtest====================")
# region Description
from ANTICOR import ANTICOR1
env_Anticor = StockTradingEnv
environment_instance_test_Anticor = env_Anticor(config=env_test_config)
environment_Anticor = environment_instance_test_Anticor
window_size = 200
number_of_stocks = 29
###############################
agent = ANTICOR1()
###################################
eposide_return_modified_by_risk_list_Average_Anticor = list()
eposide_total_asset1_list_Average_Anticor = list()
eposdie_return_list_Average_Anticor = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_Anticor.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_Anticor.append(eposide_return)
eposide_return_modified_by_risk_list_Average_Anticor.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_Anticor.append(eposide_total_asset)
# initial_portfolio_weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_Anticor.max_step):
    information = generate_history_matrix(environment_Anticor.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_Anticor.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_Anticor.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_Anticor.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_Anticor.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================7. Traditional strategies--PAMA in the backtest====================")
# region Description
from PAMR import PAMR
env_PAMR = StockTradingEnv
environment_instance_test_PAMR = env_PAMR(config=env_test_config)
environment_PAMR = environment_instance_test_PAMR
window_size = 200
number_of_stocks = 29
###############################
agent = PAMR()
###################################
eposide_return_modified_by_risk_list_Average_PAMR = list()
eposide_total_asset1_list_Average_PAMR = list()
eposdie_return_list_Average_PAMR = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_PAMR.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_PAMR.append(eposide_return)
eposide_return_modified_by_risk_list_Average_PAMR.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_PAMR.append(eposide_total_asset)
# initial_portfolio_weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_PAMR.max_step):
    information = generate_history_matrix(environment_PAMR.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_PAMR.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_PAMR.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_PAMR.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_PAMR.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================8. Traditional strategies--Cornk in the backtest====================")
# region Description
from CORNK import CORNK
env_CORNK = StockTradingEnv
environment_instance_test_CORNK = env_CORNK(config=env_test_config)
environment_CORNK = environment_instance_test_CORNK
window_size = 200
number_of_stocks = 29
###############################
agent = CORNK()
###################################
eposide_return_modified_by_risk_list_Average_CORNK = list()
eposide_total_asset1_list_Average_CORNK = list()
eposdie_return_list_Average_CORNK = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_CORNK.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_CORNK.append(eposide_return)
eposide_return_modified_by_risk_list_Average_CORNK.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_CORNK.append(eposide_total_asset)
# initial_portfolio_weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_CORNK.max_step):
    information = generate_history_matrix(environment_CORNK.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_CORNK.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_CORNK.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_CORNK.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_CORNK.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================9. Traditional strategies--MO in the backtest====================")
# region Description
from M0 import M0
env_M0 = StockTradingEnv
environment_instance_test_M0 = env_M0(config=env_test_config)
environment_M0 = environment_instance_test_M0
window_size = 200
number_of_stocks = 29
eposdie_return_list_Average_M0 = list()
###############################
agent = M0()
###################################
eposide_return_modified_by_risk_list_Average_M0 = list()
eposide_total_asset1_list_Average_M0 = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_M0.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_M0.append(eposide_return)
eposide_return_modified_by_risk_list_Average_M0.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_M0.append(eposide_total_asset)
# initial_portfolio_weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_M0.max_step):
    information = generate_history_matrix(environment_M0.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_M0.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_M0.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_M0.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_M0.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================10. Traditional strategies--RMR in the backtest====================")
# region Description
from RMR import RMR
env_RMR = StockTradingEnv
environment_instance_test_RMR = env_RMR(config=env_test_config)
environment_RMR = environment_instance_test_RMR
window_size = 200
number_of_stocks = 29
eposdie_return_list_Average_RMR = list()
###############################
agent = RMR()
###################################
eposide_return_modified_by_risk_list_Average_RMR = list()
eposide_total_asset1_list_Average_RMR = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_RMR.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_RMR.append(eposide_return)
eposide_return_modified_by_risk_list_Average_RMR.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_RMR.append(eposide_total_asset)
# initial_portfolio_weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_RMR.max_step):
    information = generate_history_matrix(environment_RMR.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_RMR.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_RMR.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_RMR.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_RMR.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================11. Traditional strategies--CWMR_STD in the backtest====================")
# region Description
from CWMR import CWMR_STD
env_CWMR = StockTradingEnv
environment_instance_test_CWMR = env_CWMR(config=env_test_config)
environment_CWMR = environment_instance_test_CWMR
window_size = 200
number_of_stocks = 29
###############################
agent = CWMR_STD()
###################################
eposide_return_modified_by_risk_list_Average_CWMR = list()
eposide_total_asset1_list_Average_CWMR = list()
eposdie_return_list_Average_CWMR = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_CWMR.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_CWMR.append(eposide_return)
eposide_return_modified_by_risk_list_Average_CWMR.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_CWMR.append(eposide_total_asset)
# initial_portfolio_weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_portfolio_weights = np.ones(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_CWMR.max_step):
    information = generate_history_matrix(environment_CWMR.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_CWMR.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_CWMR.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_CWMR.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_CWMR.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================12. Traditional strategies--EG in the backtest====================")
# region Description
from EG import EG
env_EG = StockTradingEnv
environment_instance_test_EG = env_EG(config=env_test_config)
environment_EG = environment_instance_test_EG
window_size = 200
number_of_stocks = 29
###############################
agent = EG()
###################################
eposide_return_modified_by_risk_list_Average_EG = list()
eposide_total_asset1_list_Average_EG = list()
eposdie_return_list_Average_EG = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_EG.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_EG.append(eposide_return)
eposide_return_modified_by_risk_list_Average_EG.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_EG.append(eposide_total_asset)
# initial_portfolio_weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_EG.max_step):
    information = generate_history_matrix(environment_EG.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_EG.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)
    # print("============================================================")
    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_EG.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_EG.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_EG.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================13. Traditional strategies--SP in the backtest====================")
# region Description
from SP import SP
env_SP = StockTradingEnv
environment_instance_test_SP = env_SP(config=env_test_config)
environment_SP = environment_instance_test_SP
window_size = 32
number_of_stocks = 29
eposdie_return_list_Average_SP = list()
###############################
agent = SP()
###################################
eposide_return_modified_by_risk_list_Average_SP = list()
eposide_total_asset1_list_Average_SP = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_SP.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_SP.append(eposide_return)
eposide_return_modified_by_risk_list_Average_SP.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_SP.append(eposide_total_asset)
# initial_portfolio_weights = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
initial_portfolio_weights  = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_SP.max_step):
    information = generate_history_matrix(environment_SP.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_SP.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_SP.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_SP.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_SP.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================14. Traditional strategies--UBAH in the backtest====================")
# region Description
from UBAH import UBAH
env_UBAH = StockTradingEnv
environment_instance_test_UBAH = env_UBAH(config=env_test_config)
environment_UBAH = environment_instance_test_UBAH
window_size = 200
number_of_stocks = 29
###############################
agent = UBAH()
###################################
eposide_return_modified_by_risk_list_Average_UBAH = list()
eposide_total_asset1_list_Average_UBAH = list()
eposdie_return_list_Average_UBAH = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_UBAH.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_UBAH.append(eposide_return)
eposide_return_modified_by_risk_list_Average_UBAH.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_UBAH.append(eposide_total_asset)
# initial_portfolio_weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_UBAH.max_step):
    ### information = generate_history_matrix(environment_UBAH.collecting(Date, window_size))
    information = environment_UBAH.collecting(Date, window_size).T
    information = information[np.newaxis, :, :]
    # print(information)
    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights)
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_UBAH.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_UBAH.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_UBAH.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_UBAH.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================15. Traditional strategies--WMAMR in the backtest====================")
# region Description
from WHAMR import WMAMR
env_WMAMR = StockTradingEnv
environment_instance_test_WMAMR = env_WMAMR(config=env_test_config)
environment_WMAMR = environment_instance_test_WMAMR
window_size = 32
number_of_stocks = 29
###############################
agent = WMAMR()
###################################
eposide_return_modified_by_risk_list_Average_WMAMR = list()
eposide_total_asset1_list_Average_WMAMR = list()
eposdie_return_list_Average_WMAMR = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_WMAMR.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_WMAMR.append(eposide_return)
eposide_return_modified_by_risk_list_Average_WMAMR.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_WMAMR.append(eposide_total_asset)
# initial_portfolio_weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_portfolio_weights = np.ones(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_WMAMR.max_step):
    information = generate_history_matrix(environment_WMAMR.collecting(Date, window_size))
    information = information[np.newaxis, :, :]
    # print(information)
    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights)
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_WMAMR.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_WMAMR.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_WMAMR.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_WMAMR.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

"""



'''
print("================2.1  Mean-Variance--Robust Bayesian average algorithm procedure in the backtest======================================")
# region Description
from Robust_Bayesian_Portfolio_Choices import Robust_Bayesian_Portfolio_Choices
env_Robust_Bayesian = StockTradingEnv
environment_instance_test_Robust_Bayesian = env_Robust_Bayesian(config=env_test_config)
environment_Robust_Bayesian = environment_instance_test_Robust_Bayesian
window_size = 32
number_of_stocks = 10
agent = Robust_Bayesian_Portfolio_Choices()
eposdie_return_list_Average_Robust_Bayesian = list()
eposide_return_modified_by_risk_list_Average_Robust_Bayesian = list()
eposide_total_asset1_list_Average_Robust_Bayesian = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_Robust_Bayesian.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)

eposdie_return_list_Average_Robust_Bayesian.append(eposide_return)
eposide_return_modified_by_risk_list_Average_Robust_Bayesian.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_Robust_Bayesian.append(eposide_total_asset)

initial_portfolio_weights = np.zeros(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_Robust_Bayesian.max_step):
    information = generate_history_matrix(environment_Robust_Bayesian.collecting_all_history(Date))
    information = information[np.newaxis, :, :]

    # make portfolio decision
    if i <= 0:
        portfolio_weights = agent.decide_by_history_first_step(information, last_portfolio_weights)
    if i > 0:
        print("======================================================")
        print("======================================================")
        portfolio_weights = agent.decide_by_history_continous(information, last_portfolio_weights)
    # print("portfolio_weights(BK):", portfolio_weights.shape)
    # print("stock_value_weights(BK):", portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_Robust_Bayesian.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights

    eposdie_return_list_Average_Robust_Bayesian.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_Jorion_Bayes.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_Robust_Bayesian.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
'''

# endregion
"""
print("================2.1  Mean-Variance--Jorion's Bayes-Stein procedure in the backtest======================================")
# region Description
from Jorion_Bayes_Stein_procedure import Jorion_Bayes_Stein
env_Jorion_Bayes = StockTradingEnv
environment_instance_test_Jorion_Bayes = env_Jorion_Bayes(config=env_test_config)
environment_Jorion_Bayes = environment_instance_test_Jorion_Bayes
window_size = 200
number_of_stocks = 10
agent = Jorion_Bayes_Stein()
eposdie_return_list_Average_Jorion_Bayes = list()
eposide_return_modified_by_risk_list_Average_Jorion_Bayes = list()
eposide_total_asset1_list_Average_Jorion_Bayes = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_Jorion_Bayes.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_Jorion_Bayes.append(eposide_return)
eposide_return_modified_by_risk_list_Average_Jorion_Bayes.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_Jorion_Bayes.append(eposide_total_asset)
initial_portfolio_weights = np.ones(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_Jorion_Bayes.max_step):
    information = generate_history_matrix(environment_Jorion_Bayes.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights(BK):", portfolio_weights.shape)
    # print("stock_value_weights(BK):", portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_Jorion_Bayes.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_Jorion_Bayes.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_Jorion_Bayes.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_Jorion_Bayes.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================2.2  Mean-Variance--Rolling exception procedure in the backtest=========================================")
# region Description
from Rolling_exoectation_mean_variance import Rolling_exceptation_in_mean_variance
env_Rolling_exceptation = StockTradingEnv
environment_instance_test_Rolling_exceptation = env_Rolling_exceptation(config=env_test_config)
environment_Rolling_exceptation = environment_instance_test_Rolling_exceptation
window_size = 200
number_of_stocks = 10
agent = Rolling_exceptation_in_mean_variance()
eposdie_return_list_Average_Rolling_exceptation = list()
eposide_return_modified_by_risk_list_Average_Rolling_exceptation = list()
eposide_total_asset1_list_Average_Rolling_exceptation = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_Rolling_exceptation.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_Rolling_exceptation.append(eposide_return)
eposide_return_modified_by_risk_list_Average_Rolling_exceptation.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_Rolling_exceptation.append(eposide_total_asset)
initial_portfolio_weights = np.ones(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_Jorion_Bayes.max_step):
    information = generate_history_matrix(environment_Rolling_exceptation.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights.shape)
    # print("stock_value_weights(BK):", portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_Rolling_exceptation.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_Rolling_exceptation.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_Rolling_exceptation.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_Rolling_exceptation.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================2.3  Mean-Variance--Rolling exceptionKan and Zhou's three fund rule procedure in the backtest===========")
# region Description
from Kan_and_zhou_three_fund import Kan_and_zhou_three_fund
env_Kan_and_zhou_three_fund = StockTradingEnv
environment_instance_test_Kan_and_zhou_three_fund = env_Kan_and_zhou_three_fund(config=env_test_config)
environment_Kan_and_zhou_three_fund = environment_instance_test_Kan_and_zhou_three_fund
window_size = 200
number_of_stocks = 10
agent = Kan_and_zhou_three_fund()
eposdie_return_list_Average_Kan_and_zhou_three_fund = list()
eposide_return_modified_by_risk_list_Average_Kan_and_zhou_three_fund = list()
eposide_total_asset1_list_Average_Kan_and_zhou_three_fund = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_Kan_and_zhou_three_fund.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_Kan_and_zhou_three_fund.append(eposide_return)
eposide_return_modified_by_risk_list_Average_Kan_and_zhou_three_fund.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_Kan_and_zhou_three_fund.append(eposide_total_asset)
initial_portfolio_weights = np.ones(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_Jorion_Bayes.max_step):
    information = generate_history_matrix(environment_Kan_and_zhou_three_fund.collecting(Date, window_size))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights.shape)
    # print("stock_value_weights(BK):", portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_Kan_and_zhou_three_fund.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_Kan_and_zhou_three_fund.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_Kan_and_zhou_three_fund.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_Kan_and_zhou_three_fund.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

"""


"""
eposdie_return_list_Average_Jorion_Bayes_plot = np.array(eposdie_return_list_Average_Jorion_Bayes)
eposdie_return_list_Average_Rolling_exceptation_plot = np.array(eposdie_return_list_Average_Rolling_exceptation)
eposdie_return_list_Average_Kan_and_zhou_three_fund_plot = np.array(eposdie_return_list_Average_Kan_and_zhou_three_fund)

eposide_return_modified_by_risk_list_Average_Jorion_Bayes_plot = np.array(eposide_return_modified_by_risk_list_Average_Jorion_Bayes)
eposide_return_modified_by_risk_list_Average_Rolling_exceptation_plot = np.array(eposide_return_modified_by_risk_list_Average_Rolling_exceptation)
eposide_return_modified_by_risk_list_Average_Kan_and_zhou_three_fund_plot = np.array(eposide_return_modified_by_risk_list_Average_Kan_and_zhou_three_fund)

eposide_total_asset1_list_Average_Jorion_Bayes_plot = np.array(eposide_total_asset1_list_Average_Jorion_Bayes)
eposide_total_asset1_list_Average_Rolling_exceptation_plot = np.array(eposide_total_asset1_list_Average_Rolling_exceptation)
eposide_total_asset1_list_Average_Kan_and_zhou_three_fund_plot = np.array(eposide_total_asset1_list_Average_Kan_and_zhou_three_fund)
"""

"""
print("================输出全部最优策略====================")
print("================16. Traditional strategies--BCRP in the backtest====================")
# region Description
from BCRP import BCRP
env_BCRP = StockTradingEnv
environment_instance_test_BCRP = env_BCRP(config=env_test_config)
environment_BCRP = environment_instance_test_BCRP
window_size = 32
number_of_stocks = 29
###############################
agent = BCRP()
###################################
eposide_return_modified_by_risk_list_Average_BCRP = list()
eposide_total_asset1_list_Average_BCRP = list()
eposdie_return_list_Average_BCRP = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_BCRP.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_BCRP.append(eposide_return)
eposide_return_modified_by_risk_list_Average_BCRP.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_BCRP.append(eposide_total_asset)
initial_portfolio_weights = np.ones(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_BCRP.max_step):
    information = generate_history_matrix(
        environment_BCRP.collecting(environment_BCRP.max_step, environment_BCRP.max_step))
    information = information[np.newaxis, :, :]

    # make portfolio decision

    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_BCRP.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_BCRP.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_BCRP.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_BCRP.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion

print("================17. Traditional strategies--Best in the backtest====================")
# region Description
from BEST import BEST
env_Best = StockTradingEnv
environment_instance_test_Best = env_Best(config=env_test_config)
environment_Best = environment_instance_test_Best
window_size = 32
number_of_stocks = 29
eposdie_return_list_Average_Best = list()
###############################
agent = BEST()
###################################
eposide_return_modified_by_risk_list_Average_Best = list()
eposide_total_asset1_list_Average_Best = list()

Date, eposide_total_asset, eposide_return, eposide_return_modified_by_risk = environment_Best.beginning()
# print("eposide_total_asset:", eposide_total_asset)
# print("eposide_return:",eposide_return)
eposdie_return_list_Average_Best.append(eposide_return)
eposide_return_modified_by_risk_list_Average_Best.append(eposide_return_modified_by_risk)
eposide_total_asset1_list_Average_Best.append(eposide_total_asset)
# initial_portfolio_weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_portfolio_weights = np.ones(number_of_stocks) / number_of_stocks
last_portfolio_weights = initial_portfolio_weights
for i in range(environment_Best.max_step):
    information = generate_history_matrix(
        environment_Best.collecting(environment_Best.max_step, environment_Best.max_step))
    information = information[np.newaxis, :, :]

    # make portfolio decision
    portfolio_weights = agent.decide_by_history(information, last_portfolio_weights)
    # print("portfolio_weights:", portfolio_weights)
    # print("stock_value_weights:" ,portfolio_weights.sum())
    # portfolio_weights  = np.array(portfolio_weights )
    # portfolio_weights = np.matrix(covariance)
    # print("expected_return:",expected_return.shape)
    # print("covariance_inverse :",covariance_inverse .shape)
    # portfolio_weights = information[1,:]/information[1,:].sum()
    # print(portfolio_weights[:,0])
    eposide_return, eposide_return_modified_by_risk, eposide_total_asset, Date, done = environment_Best.backtest_step(
        portfolio_weights)
    # print(Date-32, "|  ", eposide_total_asset)

    last_portfolio_weights = portfolio_weights
    eposdie_return_list_Average_Best.append(eposide_return)
    eposide_return_modified_by_risk_list_Average_Best.append(eposide_return_modified_by_risk)
    eposide_total_asset1_list_Average_Best.append(eposide_total_asset)
    if done:
        print("Test Finished!")
        print("eposide_total_asset:", eposide_total_asset)
        print("eposide_return:", eposide_return)
        break
# endregion


print("================策略表现总结====================")

print("================提取作图所需要的各个策略的累计收益，累计资产，累计目标函数的变化====================")
"""
eposdie_return_list_Average_CRP_plot = np.array(eposdie_return_list_Average_CRP)
# region Description
"""
eposdie_return_list_Average_BK_plot = np.array(eposdie_return_list_Average_BK)

eposdie_return_list_Average_ONS_plot = np.array(eposdie_return_list_Average_ONS)
eposdie_return_list_Average_OLMAR_plot = np.array(eposdie_return_list_Average_OLMAR)
eposdie_return_list_Average_UP_plot = np.array(eposdie_return_list_Average_UP)
eposdie_return_list_Average_Anticor_plot = np.array(eposdie_return_list_Average_Anticor)
eposdie_return_list_Average_PAMR_plot = np.array(eposdie_return_list_Average_PAMR)
eposdie_return_list_Average_CORNK_plot = np.array(eposdie_return_list_Average_CORNK)
eposdie_return_list_Average_M0_plot = np.array(eposdie_return_list_Average_M0)
eposdie_return_list_Average_RMR_plot = np.array(eposdie_return_list_Average_RMR)
eposdie_return_list_Average_CWMR_plot = np.array(eposdie_return_list_Average_CWMR)
eposdie_return_list_Average_EG_plot = np.array(eposdie_return_list_Average_EG)
eposdie_return_list_Average_SP_plot = np.array(eposdie_return_list_Average_SP)
eposdie_return_list_Average_UBAH_plot = np.array(eposdie_return_list_Average_UBAH)
eposdie_return_list_Average_WMAMR_plot = np.array(eposdie_return_list_Average_WMAMR)
eposdie_return_list_Average_BCRP_plot = np.array(eposdie_return_list_Average_BCRP)
eposdie_return_list_Average_Best_plot = np.array(eposdie_return_list_Average_Best)
"""
Date_test_plot = np.array(Date_test)
# eposide_return_modified_by_risk_list_Average_CRP_plot = np.array(eposide_return_modified_by_risk_list_Average_CRP)
eposide_total_asset1_list_Average_CRP_plot = np.array(eposide_total_asset1_list_Average_CRP)
"""
eposide_return_modified_by_risk_list_Average_BK_plot = np.array(eposide_return_modified_by_risk_list_Average_BK)

eposide_return_modified_by_risk_list_Average_ONS_plot = np.array(eposide_return_modified_by_risk_list_Average_ONS)
eposide_return_modified_by_risk_list_Average_OLMAR_plot = np.array(eposide_return_modified_by_risk_list_Average_OLMAR)
eposide_return_modified_by_risk_list_Average_UP_plot = np.array(eposide_return_modified_by_risk_list_Average_UP)
eposide_return_modified_by_risk_list_Average_Anticor_plot = np.array(eposide_return_modified_by_risk_list_Average_Anticor)
eposide_return_modified_by_risk_list_Average_PAMR_plot = np.array(eposide_return_modified_by_risk_list_Average_PAMR)
eposide_return_modified_by_risk_list_Average_CORNK_plot = np.array(eposide_return_modified_by_risk_list_Average_CORNK)
eposide_return_modified_by_risk_list_Average_M0_plot = np.array(eposide_return_modified_by_risk_list_Average_M0)
eposide_return_modified_by_risk_list_Average_RMR_plot = np.array(eposide_return_modified_by_risk_list_Average_RMR)
eposide_return_modified_by_risk_list_Average_CWMR_plot = np.array(eposide_return_modified_by_risk_list_Average_CWMR)
eposide_return_modified_by_risk_list_Average_EG_plot = np.array(eposide_return_modified_by_risk_list_Average_EG)
eposide_return_modified_by_risk_list_Average_SP_plot = np.array(eposide_return_modified_by_risk_list_Average_SP)
eposide_return_modified_by_risk_list_Average_UBAH_plot = np.array(eposide_return_modified_by_risk_list_Average_UBAH)
eposide_return_modified_by_risk_list_Average_WMAMR_plot = np.array(eposide_return_modified_by_risk_list_Average_WMAMR)
eposide_return_modified_by_risk_list_Average_BCRP_plot = np.array(eposide_return_modified_by_risk_list_Average_BCRP)
eposide_return_modified_by_risk_list_Average_Best_plot = np.array(eposide_return_modified_by_risk_list_Average_Best)

eposide_total_asset1_list_Average_BK_plot = np.array(eposide_total_asset1_list_Average_BK)

eposide_total_asset1_list_Average_ONS_plot = np.array(eposide_total_asset1_list_Average_ONS)
eposide_total_asset1_list_Average_OLMAR_plot = np.array(eposide_total_asset1_list_Average_OLMAR)
eposide_total_asset1_list_Average_UP_plot = np.array(eposide_total_asset1_list_Average_UP)
eposide_total_asset1_list_Average_Anticor_plot = np.array(eposide_total_asset1_list_Average_Anticor)
eposide_total_asset1_list_Average_PAMR_plot = np.array(eposide_total_asset1_list_Average_PAMR)
eposide_total_asset1_list_Average_CORNK_plot = np.array(eposide_total_asset1_list_Average_CORNK)
eposide_total_asset1_list_Average_M0_plot = np.array(eposide_total_asset1_list_Average_M0)
eposide_total_asset1_list_Average_RMR_plot = np.array(eposide_total_asset1_list_Average_RMR)
eposide_total_asset1_list_Average_CWMR_plot = np.array(eposide_total_asset1_list_Average_CWMR)
eposide_total_asset1_list_Average_EG_plot = np.array(eposide_total_asset1_list_Average_EG)
eposide_total_asset1_list_Average_SP_plot = np.array(eposide_total_asset1_list_Average_SP)
eposide_total_asset1_list_Average_UBAH_plot = np.array(eposide_total_asset1_list_Average_UBAH)
eposide_total_asset1_list_Average_WMAMR_plot = np.array(eposide_total_asset1_list_Average_WMAMR)
eposide_total_asset1_list_Average_BCRP_plot = np.array(eposide_total_asset1_list_Average_BCRP)
eposide_total_asset1_list_Average_Best_plot = np.array(eposide_total_asset1_list_Average_Best)
"""


episode_returns_modified_by_risk_DDPG_plot = np.array(episode_returns_modified_by_risk_DDPG)
episode_returns_DDPG_plot = np.array(episode_returns_DDPG)
episode_total_assets_DDPG_plot = np.array(episode_total_assets_DDPG)

# episode_total_assets_DDPG_melt, episode_returns_DDPG_melt, episode_returns_modified_by_risk_DDPG_melt
episode_returns_modified_by_risk_melt_DDPG_plot = np.array(episode_returns_modified_by_risk_DDPG_melt)

episode_returns_DDPG_melt_plot = np.array(episode_returns_DDPG_melt)
# print("episode_returns_DDPG_melt_plot", episode_returns_DDPG_melt_plot)
episode_total_assets_DDPG_melt_plot = np.array(episode_total_assets_DDPG_melt)
# endregion


# print("================对累计收益和累计目标函数作图====================")
# # region Description
# recorder=None
# cwd= "./test_ddpg"
# save_title='performance in back test'
# fig_name='plot_backtest_curve.jpg'
#
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 10), dpi=3000)
# # fig = plt.figure(dpi=100,constrained_layout=True) #类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
# # gs = GridSpec(2, 2, figure=fig) #GridSpec将fiure分为3行3列，每行三个axes，gs为一个matplotlib.gridspec.GridSpec对象，可灵活的切片figure
# fig, axs = plt.subplots(2, figsize=(15, 10), dpi=100)
#
# # axs[0]
# if len(Date_test_plot) >= len(episode_returns_DDPG_plot):
#     episode_returns_DDPG_plot = np.concatenate((episode_returns_DDPG_plot, episode_returns_DDPG_plot[-1] * np.ones(len(Date_test_plot)-len(episode_returns_DDPG_plot))), axis = 0)
###########################################################################
print("================对累计收益和累计目标函数作图====================")
# region Description
recorder=None
cwd= "./test_ddpg"
save_title='performance in back test'
fig_name='plot_backtest_curve.jpg'

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10), dpi=3000)
# fig = plt.figure(dpi=100,constrained_layout=True) #类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
# gs = GridSpec(2, 2, figure=fig) #GridSpec将fiure分为3行3列，每行三个axes，gs为一个matplotlib.gridspec.GridSpec对象，可灵活的切片figure
fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=100)

# axs[0]
if len(Date_test_plot) >= len(episode_returns_DDPG_plot):
    episode_returns_DDPG_plot = np.concatenate((episode_returns_DDPG_plot, episode_returns_DDPG_plot[-1] * np.ones(len(Date_test_plot)-len(episode_returns_DDPG_plot))), axis = 0)
#######################################################
ax00 = axs[0, 0]
ax00.cla()

color1 = 'darkcyan'

ax00.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax00.tick_params(axis='x', labelsize=12)
title00 = "(a)"
ax00.set_title(title00,fontsize = 15, loc = "left")

color000 = 'lightcoral'
color001 = 'peru'  #
color002 = 'mediumturquoise'
color003 = 'sandybrown'
color004 = 'gold'
color005 = 'gray'  #
color006 = 'y'
color007 = 'orange'  #
color008 = 'g'
color009 = 'mediumaquamarine'  #
color0010 = 'thistle'  #
color0011 = 'slateblue'
color0012 = 'violet'
color0013 = 'plum'
color0014 = 'teal'
color0015 = 'slategray'
color0016 = 'peru'
color0017 = 'darkred'
color0018 = 'k'
color0019 = 'r'
ax00.set_ylabel('Episode Return', fontsize=15)
ax00.set_xlabel('Trading day', fontsize=15)

def plot_line_data_transform(value):
    # print("value", value)
    risk_free_rate_value = np.log2(0 + 0.03 * (1/250))
    value = np.concatenate([np.ones(1) * risk_free_rate_value, np.array(value).reshape(1)])
    return value

def plot_line_data_transform_risk(value):
    # print("value", value)
    # risk_free_rate_value = np.log2(1 + 0.03 * (1/250))
    value = np.concatenate([np.zeros(1) , np.array(value).reshape(1)])
    return value

'''
eposdie_return_list_Average_Jorion_Bayes_plot = np.array(eposdie_return_list_Average_Jorion_Bayes)
eposdie_return_list_Average_Rolling_exceptation_plot = np.array(eposdie_return_list_Average_Rolling_exceptation)
eposdie_return_list_Average_Kan_and_zhou_three_fund_plot = np.array(eposdie_return_list_Average_Kan_and_zhou_three_fund)
'''

ax00.plot(Date_test_plot, episode_returns_DDPG_plot, label='HDRL', color=color0018, lw=1.5)
ax00.plot(Date_test_plot, eposdie_return_list_Average_CRP_plot, label='CRP', color=color001, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, episode_returns_DDPG_melt_plot, label='BDA', color=color0018, lw=1.5)
"""
ax00.plot(Date_test_plot, eposdie_return_list_Average_BK_plot, label='BK', color=color000, lw=0.7, ls="-.")

ax00.plot(Date_test_plot, eposdie_return_list_Average_ONS_plot, label='ONS', color=color002, lw=0.7, ls="-.")
ax00.plot(Date_test_plot, eposdie_return_list_Average_OLMAR_plot, label='OLMAR', color=color003, lw=0.7, ls="-.")
ax00.plot(Date_test_plot, eposdie_return_list_Average_UP_plot, label='UP', color=color004, lw=0.7, ls="-.")
ax00.plot(Date_test_plot, eposdie_return_list_Average_ONS_plot, label='ONS', color=color005, lw=0.7, ls="-.")
ax00.plot(Date_test_plot, eposdie_return_list_Average_Anticor_plot, label='Anticor', color=color006, lw=0.7, ls="-.")
ax00.plot(Date_test_plot, eposdie_return_list_Average_PAMR_plot, label='PAMR', color=color007, lw=0.7, ls="-.")
ax00.plot(Date_test_plot, eposdie_return_list_Average_CORNK_plot, label='CORNK', color=color008, lw=0.7, ls="-.")
ax00.plot(Date_test_plot, eposdie_return_list_Average_M0_plot, label='M0', color=color009, lw=0.7, ls="-.")
ax00.plot(Date_test_plot, eposdie_return_list_Average_RMR_plot, label='RMR', color=color0010, lw=0.7, ls="-.")
ax00.plot(Date_test_plot, eposdie_return_list_Average_CWMR_plot, label='CWMR', color=color0011, lw=0.7, ls="-.")
ax00.plot(Date_test_plot, eposdie_return_list_Average_EG_plot, label='EG', color=color0012, lw=0.7, ls="-.")
### ax00.plot(Date_test_plot, eposdie_return_list_Average_SP_plot, label='SP', color=color0013, lw=0.7, ls="-.")
ax00.plot(Date_test_plot, eposdie_return_list_Average_UBAH_plot, label='UBAH', color=color0014, lw=0.7, ls="-.")
ax00.plot(Date_test_plot, eposdie_return_list_Average_WMAMR_plot, label='WMAMR', color=color0015, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_BCRP_plot, label='BCRP', color=color0016, lw=0.7, ls ="-."))
# ax00.plot(Date_test_plot, eposdie_return_list_Average_Best_plot, label='Best', color=color0017, lw=0.7, ls ="-."))
ax00.plot(Date_test_plot, eposdie_return_list_Average_Jorion_Bayes_plot, label='JB', color=color0016, lw=0.7, ls="-.")
ax00.plot(Date_test_plot, eposdie_return_list_Average_Rolling_exceptation_plot, label='RE', color=color0017, lw=0.7, ls="-.")
ax00.plot(Date_test_plot, eposdie_return_list_Average_Kan_and_zhou_three_fund_plot, label='KZTF', color=color0019, lw=0.7, ls="-.")
"""

# ax00.legend(loc='center left')
ax00.legend(loc='upper left',ncol=5)
ax00.grid()

###############################################################################################################
###############################################################################################################
ax10 = axs[0, 1]
ax10.cla()

color1 = 'darkcyan'

ax10.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax10.tick_params(axis='x', labelsize=12)
title10 = "(b)"
ax10.set_title(title10,fontsize = 15, loc = "left")

color000 = 'lightcoral'
color001 = 'peru'  #
color002 = 'mediumturquoise'
color003 = 'sandybrown'
color004 = 'gold'
color005 = 'gray'  #
color006 = 'y'
color007 = 'orange'  #
color008 = 'g'
color009 = 'mediumaquamarine'  #
color0010 = 'thistle'  #
color0011 = 'slateblue'
color0012 = 'violet'
color0013 = 'plum'
color0014 = 'teal'
color0015 = 'slategray'
color0016 = 'peru'
color0017 = 'darkred'
color0018 = 'k'
color0019 = 'r'
# ax10.set_ylabel('Episode Return', fontsize=15)
# ax10.set_xlabel('Trading day', fontsize=15)

strategies = ['EIIE' , 'CRP']
              # 'BK', 'ONS',
              # 'OLMAR', 'UP', 'Anticor',
              # 'PAMR', 'CORNK','M0', 'RMR',
              # 'CWMR', 'EG', 'UBAH',
              # 'WMAMR', 'JB', 'RE', 'KZTF'
episode_return = [episode_returns_DDPG_plot, eposdie_return_list_Average_CRP_plot]

                  # eposdie_return_list_Average_BK_plot, , eposdie_return_list_Average_ONS_plot,
                  # eposdie_return_list_Average_OLMAR_plot, eposdie_return_list_Average_UP_plot, eposdie_return_list_Average_Anticor_plot,
                  # eposdie_return_list_Average_PAMR_plot, eposdie_return_list_Average_CORNK_plot, eposdie_return_list_Average_M0_plot, eposdie_return_list_Average_RMR_plot,
                  # eposdie_return_list_Average_CWMR_plot, eposdie_return_list_Average_EG_plot, eposdie_return_list_Average_UBAH_plot,
                  # eposdie_return_list_Average_WMAMR_plot, eposdie_return_list_Average_Jorion_Bayes_plot,
                  # eposdie_return_list_Average_Rolling_exceptation_plot, eposdie_return_list_Average_Kan_and_zhou_three_fund_plot
# eposdie_return_list_Average_SP_plot
ind = [x for x,_ in enumerate(strategies)]

def calculation_of_accumulated_return(episode_return):
    value1 = []
    value2 = []
    value3 = []
    for x in episode_return:
        value1.append(x[40])
        value2.append(x[80]-x[40])
        value3.append(x[120]-x[80])
    return value1, value2, value3

value1, value2, value3 = calculation_of_accumulated_return(episode_return)

ax10.bar(ind, value1, width=0.5, label='Accumulated return in 40 trading days', color='#CD853F')
ax10.bar(ind, value2, width=0.5, label='Accumulated return in 40-80 trading days', color='silver', bottom = np.array(value1))
ax10.bar(ind, value3, width=0.5, label='Accumulated return in 80-120 trading days', color='gold', bottom= np.array(value1) + np.array(value2))
'''
plt.bar(ind, golds, width=0.5, label='golds', color='gold', bottom=silvers+bronzes) 
plt.bar(ind, silvers, width=0.5, label='silvers', color='silver', bottom=bronzes) 
plt.bar(ind, bronzes, width=0.5, label='bronzes', color='#CD853F') 
ticks = ax.set_xticks([0,20,40,60]) # 设置刻度
labels = ax.set_xticklabels(['one','two','three','four'],rotation = 30,fontsize = 'small')
'''
# ax10.set_xticks(ind, strategies)
ax10.set_xticks(ind)
ax10.set_xticklabels(strategies, rotation = 45, fontsize = 'small')
ax10.set_ylabel("Accumulated return", fontsize=15)
# ax10.set_xlabel("Strategy")
# ax01.legend(loc="upper right")
# plt.title("")
ax10.legend(loc="upper right")
ax10.grid()
###############################################################################################################
# axs[1]
if len(Date_test_plot) >= len(episode_returns_modified_by_risk_DDPG_plot):
    episode_returns_modified_by_risk_DDPG_plot = np.concatenate((episode_returns_modified_by_risk_DDPG_plot, episode_returns_modified_by_risk_DDPG_plot[-1] * np.ones(len(Date_test_plot)-len(episode_returns_modified_by_risk_DDPG_plot))), axis = 0)
#######################################################
ax01 = axs[1, 0]
ax01.cla()

color1 = 'darkcyan'

ax01.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax01.tick_params(axis='x', labelsize=12)
ax01.set_ylabel('Daily Return', fontsize=15)
ax01.set_xlabel('Standard Deviation', fontsize=15)
title01 = "(c)"
ax01.set_title(title01,fontsize = 15, loc = "left")

'''
eposide_total_asset1_list_Average_Jorion_Bayes_plot = np.array(eposide_total_asset1_list_Average_Jorion_Bayes)
eposide_total_asset1_list_Average_Rolling_exceptation_plot = np.array(eposide_total_asset1_list_Average_Rolling_exceptation)
eposide_total_asset1_list_Average_Kan_and_zhou_three_fund_plot = np.array(eposide_total_asset1_list_Average_Kan_and_zhou_three_fund)
ax00.plot(Date_test_plot, episode_returns_DDPG_melt_plot, label='Mul-BDA', color=color0018, lw=1.5)
'''

ax01.plot(plot_line_data_transform_risk(np.std(np.log(episode_total_assets_DDPG_plot[1:]/episode_total_assets_DDPG_plot[:-1]))), plot_line_data_transform(np.mean(np.log(episode_total_assets_DDPG_plot[1:]/episode_total_assets_DDPG_plot[:-1]))), label='HDRL', color=color0018, lw=1.5, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log(eposide_total_asset1_list_Average_CRP_plot[1:]/eposide_total_asset1_list_Average_CRP_plot[:-1]))), plot_line_data_transform(np.mean(np.log(eposide_total_asset1_list_Average_CRP_plot[1:]/eposide_total_asset1_list_Average_CRP_plot[:-1]))), label='CRP', color=color001, lw=1, marker="o")
"""
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_BK_plot[1:]/eposide_total_asset1_list_Average_BK_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_BK_plot[1:]/eposide_total_asset1_list_Average_BK_plot[:-1]))), label='BK', color=color000, lw=1, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_ONS_plot[1:]/eposide_total_asset1_list_Average_ONS_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_ONS_plot[1:]/eposide_total_asset1_list_Average_ONS_plot[:-1]))), label='ONS', color=color002, lw=1, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_OLMAR_plot[1:]/eposide_total_asset1_list_Average_OLMAR_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_OLMAR_plot[1:]/eposide_total_asset1_list_Average_OLMAR_plot[:-1]))), label='OLMAR', color=color003, lw=1, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_UP_plot[1:]/eposide_total_asset1_list_Average_UP_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_UP_plot[1:]/eposide_total_asset1_list_Average_UP_plot[:-1]))), label='UP', color=color004, lw=1, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_ONS_plot[1:]/eposide_total_asset1_list_Average_ONS_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_ONS_plot[1:]/eposide_total_asset1_list_Average_ONS_plot[:-1]))), label='ONS', color=color005, lw=1, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_Anticor_plot[1:]/eposide_total_asset1_list_Average_Anticor_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_Anticor_plot[1:]/eposide_total_asset1_list_Average_Anticor_plot[:-1]))), label='Anticor', color=color006, lw=1, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_PAMR_plot[1:]/eposide_total_asset1_list_Average_PAMR_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_PAMR_plot[1:]/eposide_total_asset1_list_Average_PAMR_plot[:-1]))), label='PAMR', color=color007, lw=1, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_CORNK_plot[1:]/eposide_total_asset1_list_Average_CORNK_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_CORNK_plot[1:]/eposide_total_asset1_list_Average_CORNK_plot[:-1]))), label='CORNK', color=color008, lw=1, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_M0_plot[1:]/eposide_total_asset1_list_Average_M0_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_M0_plot[1:]/eposide_total_asset1_list_Average_M0_plot[:-1]))), label='M0', color=color009, lw=1, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_RMR_plot[1:]/eposide_total_asset1_list_Average_RMR_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_RMR_plot[1:]/eposide_total_asset1_list_Average_RMR_plot[:-1]))), label='RMR', color=color0010, lw=1, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_CWMR_plot[1:]/eposide_total_asset1_list_Average_CWMR_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_CWMR_plot[1:]/eposide_total_asset1_list_Average_CWMR_plot[:-1]))), label='CWMR', color=color0011, lw=1, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_EG_plot[1:]/eposide_total_asset1_list_Average_EG_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_EG_plot[1:]/eposide_total_asset1_list_Average_EG_plot[:-1]))), label='EG', color=color0012, lw=1, marker="o")
### ax01.plot(plot_line_data_transform(np.std(np.log2(eposide_total_asset1_list_Average_SP_plot[1:]/eposide_total_asset1_list_Average_SP_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_SP_plot[1:]/eposide_total_asset1_list_Average_SP_plot[:-1]))), label='SP', color=color0013, lw=1, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_UBAH_plot[1:]/eposide_total_asset1_list_Average_UBAH_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_UBAH_plot[1:]/eposide_total_asset1_list_Average_UBAH_plot[:-1]))), label='UBAH', color=color0014, lw=1, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_WMAMR_plot[1:]/eposide_total_asset1_list_Average_WMAMR_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_WMAMR_plot[1:]/eposide_total_asset1_list_Average_WMAMR_plot[:-1]))), label='WMAMR', color=color0015, lw=1, marker="o")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_BCRP_plot, label='BCRP', color=color0016, lw=0.7, ls ="-."))
# ax00.plot(Date_test_plot, eposdie_return_list_Average_Best_plot, label='Best', color=color0017, lw=0.7, ls ="-."))
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_Jorion_Bayes_plot[1:]/eposide_total_asset1_list_Average_Jorion_Bayes_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_Jorion_Bayes_plot[1:]/eposide_total_asset1_list_Average_Jorion_Bayes_plot[:-1]))), label='JB', color=color0016, lw=1, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_Rolling_exceptation_plot[1:]/eposide_total_asset1_list_Average_Rolling_exceptation_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_Rolling_exceptation_plot[1:]/eposide_total_asset1_list_Average_Rolling_exceptation_plot[:-1]))), label='RE', color=color0017, lw=1, marker="o")
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(eposide_total_asset1_list_Average_Kan_and_zhou_three_fund_plot[1:]/eposide_total_asset1_list_Average_Kan_and_zhou_three_fund_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_Kan_and_zhou_three_fund_plot[1:]/eposide_total_asset1_list_Average_Kan_and_zhou_three_fund_plot[:-1]))), label='KZTF', color=color0019, lw=1, marker="o")
"""

# ax01.legend(loc='center left')
ax01.legend(loc='upper left',ncol=5)
ax01.grid()

ax11 = axs[1, 1]
ax11.cla()

def sortino_ratio_denominator(daliy_return):
    return_negative = list()
    min_acceptable_return = np.log2(0 + 0.03)
    risk_free_rate = np.log2(0 + 0.03)
    for i in range(len(daliy_return)):
        if daliy_return[i] <= min_acceptable_return:
            # print("return",daliy_return[i])
            return_negative.append(daliy_return[i])
    return_negative = np.array(return_negative)
    # print("return_negative", return_negative)
    # sortino_ratio_denominator = np.var(return_negative)**(1/2)
    # sortino_ratio = (daliy_return.mean() - min_acceptable_return * (1 / 250)) / (
    #             ((np.sum((return_negative - risk_free_rate * (1 / 250)) ** 2)) / (len(return_negative) - 1)) ** (1 / 2))
    sortino_ratio_denominator = (
                ((np.sum((return_negative - risk_free_rate * (1 / 250)) ** 2)) / (len(return_negative))) ** (1 / 2))
    value = np.concatenate([np.zeros(1), np.array(sortino_ratio_denominator).reshape(1)])
    return value

color1 = 'darkcyan'

ax11.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax11.tick_params(axis='x', labelsize=12)
ax11.set_ylabel('Daily Return', fontsize=15)
ax11.set_xlabel('Lower partial standard deviation', fontsize=15)
title11 = "(d)"
ax11.set_title(title11,fontsize = 15, loc = "left")
# xaxis_label = "$(d))$"
# ax11.set_xlabel(xaxis_label,fontsize = 18,bbox = box)

ax11.plot(sortino_ratio_denominator(np.log(episode_total_assets_DDPG_plot[1:]/episode_total_assets_DDPG_plot[:-1])), plot_line_data_transform(np.mean(np.log2(episode_total_assets_DDPG_plot[1:]/episode_total_assets_DDPG_plot[:-1]))), label='HDRL', color=color0018, lw=1.5, marker="o")
ax11.plot(sortino_ratio_denominator(np.log(eposide_total_asset1_list_Average_CRP_plot[1:]/eposide_total_asset1_list_Average_CRP_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_CRP_plot[1:]/eposide_total_asset1_list_Average_CRP_plot[:-1]))), label='CRP', color=color001, lw=1, marker="o")
"""
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_BK_plot[1:]/eposide_total_asset1_list_Average_BK_plot[:-1])),plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_BK_plot[1:]/eposide_total_asset1_list_Average_BK_plot[:-1]))), label='BK', color=color000, lw=1, marker="o")
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_ONS_plot[1:]/eposide_total_asset1_list_Average_ONS_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_ONS_plot[1:]/eposide_total_asset1_list_Average_ONS_plot[:-1]))), label='ONS', color=color002, lw=1, marker="o")
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_OLMAR_plot[1:]/eposide_total_asset1_list_Average_OLMAR_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_OLMAR_plot[1:]/eposide_total_asset1_list_Average_OLMAR_plot[:-1]))),  label='OLMAR', color=color003, lw=1, marker="o")
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_UP_plot[1:]/eposide_total_asset1_list_Average_UP_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_UP_plot[1:]/eposide_total_asset1_list_Average_UP_plot[:-1]))), label='UP', color=color004, lw=1, marker="o")
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_ONS_plot[1:]/eposide_total_asset1_list_Average_ONS_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_ONS_plot[1:]/eposide_total_asset1_list_Average_ONS_plot[:-1]))), label='ONS', color=color005, lw=1, marker="o")
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_Anticor_plot[1:]/eposide_total_asset1_list_Average_Anticor_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_Anticor_plot[1:]/eposide_total_asset1_list_Average_Anticor_plot[:-1]))), label='Anticor', color=color006, lw=1, marker="o")
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_PAMR_plot[1:]/eposide_total_asset1_list_Average_PAMR_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_PAMR_plot[1:]/eposide_total_asset1_list_Average_PAMR_plot[:-1]))), label='PAMR', color=color007, lw=1, marker="o")
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_CORNK_plot[1:]/eposide_total_asset1_list_Average_CORNK_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_CORNK_plot[1:]/eposide_total_asset1_list_Average_CORNK_plot[:-1]))), label='CORNK', color=color008, lw=1, marker="o")
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_M0_plot[1:]/eposide_total_asset1_list_Average_M0_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_M0_plot[1:]/eposide_total_asset1_list_Average_M0_plot[:-1]))), label='M0', color=color009, lw=1, marker="o")
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_RMR_plot[1:]/eposide_total_asset1_list_Average_RMR_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_RMR_plot[1:]/eposide_total_asset1_list_Average_RMR_plot[:-1]))), label='RMR', color=color0010, lw=1, marker="o")
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_CWMR_plot[1:]/eposide_total_asset1_list_Average_CWMR_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_CWMR_plot[1:]/eposide_total_asset1_list_Average_CWMR_plot[:-1]))), label='CWMR', color=color0011, lw=1, marker="o")
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_EG_plot[1:]/eposide_total_asset1_list_Average_EG_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_EG_plot[1:]/eposide_total_asset1_list_Average_EG_plot[:-1]))), label='EG', color=color0012, lw=1, marker="o")
### ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_SP_plot[1:]/eposide_total_asset1_list_Average_SP_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_SP_plot[1:]/eposide_total_asset1_list_Average_SP_plot[:-1]))), label='SP', color=color0013, lw=1, marker="o")
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_UBAH_plot[1:]/eposide_total_asset1_list_Average_UBAH_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_UBAH_plot[1:]/eposide_total_asset1_list_Average_UBAH_plot[:-1]))), label='UBAH', color=color0014, lw=1, marker="o")
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_WMAMR_plot[1:]/eposide_total_asset1_list_Average_WMAMR_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_WMAMR_plot[1:]/eposide_total_asset1_list_Average_WMAMR_plot[:-1]))), label='WMAMR', color=color0015, lw=1, marker="o")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_BCRP_plot, label='BCRP', color=color0016, lw=0.7, ls ="-."))
# ax00.plot(Date_test_plot, eposdie_return_list_Average_Best_plot, label='Best', color=color0017, lw=0.7, ls ="-."))
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_Jorion_Bayes_plot[1:]/eposide_total_asset1_list_Average_Jorion_Bayes_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_Jorion_Bayes_plot[1:]/eposide_total_asset1_list_Average_Jorion_Bayes_plot[:-1]))), label='JB', color=color0016, lw=1, marker="o")
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_Rolling_exceptation_plot[1:]/eposide_total_asset1_list_Average_Rolling_exceptation_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_Rolling_exceptation_plot[1:]/eposide_total_asset1_list_Average_Rolling_exceptation_plot[:-1]))), label='RE', color=color0017, lw=1, marker="o")
ax11.plot(sortino_ratio_denominator(np.log2(eposide_total_asset1_list_Average_Kan_and_zhou_three_fund_plot[1:]/eposide_total_asset1_list_Average_Kan_and_zhou_three_fund_plot[:-1])), plot_line_data_transform(np.mean(np.log2(eposide_total_asset1_list_Average_Kan_and_zhou_three_fund_plot[1:]/eposide_total_asset1_list_Average_Kan_and_zhou_three_fund_plot[:-1]))), label='KZTF', color=color0019, lw=1, marker="o")
"""


# ax11.legend(loc='center left')
ax11.legend(loc='upper left',ncol=5)
ax11.grid()

plt.title(save_title, y=13.5, verticalalignment='bottom')
# plt.title(save_title,verticalalignment='bottom')
plt.savefig(f"{cwd}/{fig_name}")
plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`

#############################################################################################################################################
#############################################################################################################################################
"""将论文策略和FinRL中的强化学习智能体的表现金勋对比"""
# import os
# domain = os.path.abspath(r'FinRL_orginal_compared_strategies')
# info = "Episode_total_assets_of_all_FinRL_baseline_strategies_config.npy"
# info_FinRL = os.path.join(domain, info)
#
# info_FinRL = np.load(info_FinRL, allow_pickle=True).item()
#
# FinRL_A2C_episode_total_assets = info_FinRL["A2C_episode_total_assets"]
# FinRL_DDPG_episode_total_assets = info_FinRL["DDPG_episode_total_assets"]
# FinRL_PPO_episode_total_assets = info_FinRL["PPO_episode_total_assets"]
# FinRL_SAC_episode_total_assets = info_FinRL["SAC_episode_total_assets"]
# FinRL_TD3_episode_total_assets = info_FinRL["TD3_episode_total_assets"]


print("================对累计收益和累计目标函数作图-FinRL====================")
# region Description
recorder=None
cwd= "./test_ddpg"
save_title='performance in back test FinRL'
fig_name='plot_backtest_curve_FinRL.jpg'

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10), dpi=3000)
# fig = plt.figure(dpi=100,constrained_layout=True) #类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
# gs = GridSpec(2, 2, figure=fig) #GridSpec将fiure分为3行3列，每行三个axes，gs为一个matplotlib.gridspec.GridSpec对象，可灵活的切片figure
fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=100)


ax00 = axs[0, 0]
ax00.cla()

color1 = 'darkcyan'

ax00.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax00.tick_params(axis='x', labelsize=12)
title00 = "(a)"
ax00.set_title(title00,fontsize = 15, loc = "left")

color000 = 'lightcoral'
color001 = 'peru'  #
color002 = 'mediumturquoise'
color003 = 'sandybrown'
color004 = 'gold'
color005 = 'gray'  #
color006 = 'y'
color007 = 'orange'  #
color008 = 'g'
color009 = 'mediumaquamarine'  #
color0010 = 'thistle'  #
color0011 = 'slateblue'
color0012 = 'violet'
color0013 = 'plum'
color0014 = 'teal'
color0015 = 'slategray'
color0016 = 'peru'
color0017 = 'darkred'
color0018 = 'k'
color0019 = 'r'
ax00.set_ylabel('Episode Return', fontsize=15)
ax00.set_xlabel('Trading day', fontsize=15)

def plot_line_data_transform(value):
    # print("value", value)
    risk_free_rate_value = np.log2(0 + 0.03 * (1/250))
    value = np.concatenate([np.ones(1) * risk_free_rate_value, np.array(value).reshape(1)])
    return value

def plot_line_data_transform_risk(value):
    # print("value", value)
    # risk_free_rate_value = np.log2(0 + 0.03 * (1/250))
    value = np.concatenate([np.zeros(1) , np.array(value).reshape(1)])
    return value


# FinRL_A2C_return = np.log2(FinRL_A2C_episode_total_assets/FinRL_A2C_episode_total_assets[0])
# FinRL_DDPG_return = np.log2(FinRL_DDPG_episode_total_assets/FinRL_DDPG_episode_total_assets[0])
# FinRL_PPO_return = np.log2(FinRL_PPO_episode_total_assets/FinRL_PPO_episode_total_assets[0])
# FinRL_SAC_return = np.log2(FinRL_SAC_episode_total_assets/FinRL_SAC_episode_total_assets[0])
# FinRL_TD3_return = np.log2(FinRL_TD3_episode_total_assets/FinRL_TD3_episode_total_assets[0])

# ax00.plot(Date_test_plot, episode_returns_DDPG_plot, label='HDRL', color=color0018, lw=1.5)
# #print("episode_returns_DDPG_plot", episode_returns_DDPG_plot)
# ax00.plot(Date_test_plot, FinRL_A2C_return, label='FinRL-A2C', color=color000, lw=0.7, ls="-.")
# #print("FinRL_A2C_return", FinRL_A2C_return)
# ax00.plot(Date_test_plot, FinRL_DDPG_return, label='FinRL-DDPG', color=color001, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, FinRL_PPO_return, label='FinRL-PPO', color=color002, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, FinRL_SAC_return, label='FinRL-SAC', color=color003, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, FinRL_TD3_return, label='FinRL-TD3', color=color004, lw=0.7, ls="-.")

# ax00.legend(loc='center left')
ax00.legend(loc='upper left',ncol=3)
ax00.grid()

###############################################################################################################
###############################################################################################################
ax10 = axs[0, 1]
ax10.cla()

color1 = 'darkcyan'

ax10.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax10.tick_params(axis='x', labelsize=12)
title10 = "(b)"
ax10.set_title(title10,fontsize = 15, loc = "left")

color000 = 'lightcoral'
color001 = 'peru'  #
color002 = 'mediumturquoise'
color003 = 'sandybrown'
color004 = 'gold'
color005 = 'gray'  #
color006 = 'y'
color007 = 'orange'  #
color008 = 'g'
color009 = 'mediumaquamarine'  #
color0010 = 'thistle'  #
color0011 = 'slateblue'
color0012 = 'violet'
color0013 = 'plum'
color0014 = 'teal'
color0015 = 'slategray'
color0016 = 'peru'
color0017 = 'darkred'
color0018 = 'k'
color0019 = 'r'
# ax10.set_ylabel('Episode Return', fontsize=15)
# ax10.set_xlabel('Trading day', fontsize=15)

# strategies = ['Autofotmer', 'FinRL-A2C', 'FinRL-DDPG', 'FinRL-PPO', 'FinRL-SAC', 'FinRL-TD3']
# episode_return = [episode_returns_DDPG_plot, FinRL_A2C_return, FinRL_DDPG_return, FinRL_PPO_return, FinRL_SAC_return, FinRL_TD3_return]
# eposdie_return_list_Average_SP_plot
ind = [x for x,_ in enumerate(strategies)]

def calculation_of_accumulated_return(episode_return):
    value1 = []
    value2 = []
    value3 = []
    for x in episode_return:
        value1.append(x[40])
        value2.append(x[80]-x[40])
        value3.append(x[120]-x[80])
    return value1, value2, value3

value1, value2, value3 = calculation_of_accumulated_return(episode_return)

ax10.bar(ind, value1, width=0.5, label='Accumulated return in 40 trading days', color='#CD853F')
ax10.bar(ind, value2, width=0.5, label='Accumulated return in 40-80 trading days', color='silver', bottom = np.array(value1))
ax10.bar(ind, value3, width=0.5, label='Accumulated return in 80-120 trading days', color='gold', bottom= np.array(value1) + np.array(value2))
'''
plt.bar(ind, golds, width=0.5, label='golds', color='gold', bottom=silvers+bronzes) 
plt.bar(ind, silvers, width=0.5, label='silvers', color='silver', bottom=bronzes) 
plt.bar(ind, bronzes, width=0.5, label='bronzes', color='#CD853F') 
ticks = ax.set_xticks([0,20,40,60]) # 设置刻度
labels = ax.set_xticklabels(['one','two','three','four'],rotation = 30,fontsize = 'small')
'''
# ax10.set_xticks(ind, strategies)
ax10.set_xticks(ind)
ax10.set_xticklabels(strategies, rotation = 45, fontsize = 'small')
ax10.set_ylabel("Accumulated return", fontsize=15)
# ax10.set_xlabel("Strategy")
# ax01.legend(loc="upper right")
# plt.title("")
ax10.legend(loc="upper right")
ax10.grid()
###############################################################################################################
# axs[1]
if len(Date_test_plot) >= len(episode_returns_modified_by_risk_DDPG_plot):
    episode_returns_modified_by_risk_DDPG_plot = np.concatenate((episode_returns_modified_by_risk_DDPG_plot, episode_returns_modified_by_risk_DDPG_plot[-1] * np.ones(len(Date_test_plot)-len(episode_returns_modified_by_risk_DDPG_plot))), axis = 0)
#######################################################
ax01 = axs[1, 0]
ax01.cla()

color1 = 'darkcyan'

ax01.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax01.tick_params(axis='x', labelsize=12)
ax01.set_ylabel('Daily Return', fontsize=15)
ax01.set_xlabel('Standard Deviation', fontsize=15)
title01 = "(c)"
ax01.set_title(title01,fontsize = 15, loc = "left")

# episode_return = [episode_returns_DDPG_plot, FinRL_A2C_episode_total_assets, FinRL_DDPG_episode_total_assets, FinRL_PPO_episode_total_assets, FinRL_SAC_episode_total_assets, FinRL_TD3_episode_total_assets]
ax01.plot(plot_line_data_transform_risk(np.std(np.log2(episode_total_assets_DDPG_plot[1:]/episode_total_assets_DDPG_plot[:-1]))), plot_line_data_transform(np.mean(np.log2(episode_total_assets_DDPG_plot[1:]/episode_total_assets_DDPG_plot[:-1]))), label='HDRL', color=color0018, lw=1.5, marker="o")
# print("episode_total_assets_DDPG_plot", episode_total_assets_DDPG_plot)
# ax01.plot(plot_line_data_transform_risk(np.std(np.log2(FinRL_A2C_episode_total_assets[1:]/FinRL_A2C_episode_total_assets[:-1]))), plot_line_data_transform(np.mean(np.log2(FinRL_A2C_episode_total_assets[1:]/FinRL_A2C_episode_total_assets[:-1]))), label='FinRL-A2C', color=color000, lw=1, marker="o")
# # print("FinRL_A2C_episode_total_assets", FinRL_A2C_episode_total_assets)
# ax01.plot(plot_line_data_transform_risk(np.std(np.log2(FinRL_DDPG_episode_total_assets[1:]/FinRL_DDPG_episode_total_assets[:-1]))), plot_line_data_transform(np.mean(np.log2(FinRL_DDPG_episode_total_assets[1:]/FinRL_DDPG_episode_total_assets[:-1]))), label='FinRL-DDPG', color=color001, lw=1, marker="o")
# ax01.plot(plot_line_data_transform_risk(np.std(np.log2(FinRL_PPO_episode_total_assets[1:]/FinRL_PPO_episode_total_assets[:-1]))), plot_line_data_transform(np.mean(np.log2(FinRL_PPO_episode_total_assets[1:]/FinRL_PPO_episode_total_assets[:-1]))), label='FinRL-PPO', color=color002, lw=1, marker="o")
# ax01.plot(plot_line_data_transform_risk(np.std(np.log2(FinRL_SAC_episode_total_assets[1:]/FinRL_SAC_episode_total_assets[:-1]))), plot_line_data_transform(np.mean(np.log2(FinRL_SAC_episode_total_assets[1:]/FinRL_SAC_episode_total_assets[:-1]))), label='FinRL-SAC', color=color003, lw=1, marker="o")
# ax01.plot(plot_line_data_transform_risk(np.std(np.log2(FinRL_TD3_episode_total_assets[1:]/FinRL_TD3_episode_total_assets[:-1]))), plot_line_data_transform(np.mean(np.log2(FinRL_TD3_episode_total_assets[1:]/FinRL_TD3_episode_total_assets[:-1]))), label='FinRL-TD3', color=color004, lw=1, marker="o")
# ax01.legend(loc='center left')
ax01.legend(loc='upper left',ncol=3)
ax01.grid()

ax11 = axs[1, 1]
ax11.cla()

def sortino_ratio_denominator(daliy_return):
    return_negative = list()
    # min_acceptable_return = np.log2(1 + 0.03)
    risk_free_rate = np.log2(1)
    for i in range(len(daliy_return)):
        if daliy_return[i] <= 0:
            # print("return",daliy_return[i])
            return_negative.append(daliy_return[i])
    return_negative = np.array(return_negative)
    # print("return_negative", return_negative)
    # sortino_ratio_denominator = np.var(return_negative)**(1/2)
    # sortino_ratio = (daliy_return.mean() - min_acceptable_return * (1 / 250)) / (
    #             ((np.sum((return_negative - risk_free_rate * (1 / 250)) ** 2)) / (len(return_negative) - 1)) ** (1 / 2))
    sortino_ratio_denominator = (
                ((np.sum((return_negative - risk_free_rate * (1 / 250)) ** 2)) / (len(return_negative))) ** (1 / 2))
    value = np.concatenate([np.zeros(1), np.array(sortino_ratio_denominator).reshape(1)])
    return value

color1 = 'darkcyan'

ax11.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax11.tick_params(axis='x', labelsize=12)
ax11.set_ylabel('Daily Return', fontsize=15)
ax11.set_xlabel('Lower partial standard deviation', fontsize=15)
title11 = "(d)"
ax11.set_title(title11,fontsize = 15, loc = "left")
# xaxis_label = "$(d))$"
# ax11.set_xlabel(xaxis_label,fontsize = 18,bbox = box)

# episode_return = [episode_returns_DDPG_plot, FinRL_A2C_episode_total_assets, FinRL_DDPG_episode_total_assets, FinRL_PPO_episode_total_assets, FinRL_SAC_episode_total_assets, FinRL_TD3_episode_total_assets]
ax11.plot(sortino_ratio_denominator(np.log(episode_total_assets_DDPG_plot[1:]/episode_total_assets_DDPG_plot[:-1])), plot_line_data_transform(np.mean(np.log2(episode_total_assets_DDPG_plot[1:]/episode_total_assets_DDPG_plot[:-1]))), label='HDRL', color=color0018, lw=1.5, marker="o")
# ax11.plot(sortino_ratio_denominator(np.log2(FinRL_A2C_episode_total_assets[1:]/FinRL_A2C_episode_total_assets[:-1])),plot_line_data_transform(np.mean(np.log2(FinRL_A2C_episode_total_assets[1:]/FinRL_A2C_episode_total_assets[:-1]))), label='FinRL-A2C', color=color000, lw=1, marker="o")
# ax11.plot(sortino_ratio_denominator(np.log2(FinRL_DDPG_episode_total_assets[1:]/FinRL_DDPG_episode_total_assets[:-1])), plot_line_data_transform(np.mean(np.log2(FinRL_DDPG_episode_total_assets[1:]/FinRL_DDPG_episode_total_assets[:-1]))), label='FinRL-DDPG', color=color001, lw=1, marker="o")
# ax11.plot(sortino_ratio_denominator(np.log2(FinRL_PPO_episode_total_assets[1:]/FinRL_PPO_episode_total_assets[:-1])), plot_line_data_transform(np.mean(np.log2(FinRL_PPO_episode_total_assets[1:]/FinRL_PPO_episode_total_assets[:-1]))), label='FinRL-PPO', color=color002, lw=1, marker="o")
# ax11.plot(sortino_ratio_denominator(np.log2(FinRL_SAC_episode_total_assets[1:]/FinRL_SAC_episode_total_assets[:-1])), plot_line_data_transform(np.mean(np.log2(FinRL_SAC_episode_total_assets[1:]/FinRL_SAC_episode_total_assets[:-1]))),  label='FinRL-SAC', color=color003, lw=1, marker="o")
# ax11.plot(sortino_ratio_denominator(np.log2(FinRL_TD3_episode_total_assets[1:]/FinRL_TD3_episode_total_assets[:-1])), plot_line_data_transform(np.mean(np.log2(FinRL_TD3_episode_total_assets[1:]/FinRL_TD3_episode_total_assets[:-1]))), label='FinRL-TD3', color=color004, lw=1, marker="o")

# ax11.legend(loc='center left')
ax11.legend(loc='upper left',ncol=3)
ax11.grid()

plt.title(save_title, y=13.5, verticalalignment='bottom')
# plt.title(save_title,verticalalignment='bottom')
plt.savefig(f"{cwd}/{fig_name}")
plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`

#############################################################################################################################################
#############################################################################################################################################
# ax00 = axs[0]
# ax00.cla()
#
# color1 = 'darkcyan'
#
# ax00.tick_params(axis='y', labelcolor=color1, labelsize=12)
# ax00.tick_params(axis='x', labelsize=12)
#
# color000 = 'lightcoral'
color001 = 'peru'  #
# color002 = 'mediumturquoise'
# color003 = 'sandybrown'
# color004 = 'gold'
# color005 = 'gray'  #
# color006 = 'y'
# color007 = 'orange'  #
# color008 = 'g'
# color009 = 'mediumaquamarine'  #
# color0010 = 'thistle'  #
# color0011 = 'slateblue'
# color0012 = 'violet'
# color0013 = 'plum'
# color0014 = 'teal'
# color0015 = 'slategray'
# color0016 = 'peru'
# color0017 = 'darkred'
# color0018 = 'k'
# ax00.set_ylabel('Episode Return', fontsize=15)
#
# ax00.plot(Date_test_plot, eposdie_return_list_Average_BK_plot, label='BK', color=color000, lw=0.7, ls="-.")
ax00.plot(Date_test_plot, eposdie_return_list_Average_CRP_plot, label='CRP', color=color001, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_ONS_plot, label='ONS', color=color002, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_OLMAR_plot, label='OLMAR', color=color003, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_UP_plot, label='UP', color=color004, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_ONS_plot, label='ONS', color=color005, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_Anticor_plot, label='Anticor', color=color006, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_PAMR_plot, label='PAMR', color=color007, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_CORNK_plot, label='CORNK', color=color008, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_M0_plot, label='M0', color=color009, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_RMR_plot, label='RMR', color=color0010, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_CWMR_plot, label='CWMR', color=color0011, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_EG_plot, label='EG', color=color0012, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_SP_plot, label='SP', color=color0013, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_UBAH_plot, label='UBAH', color=color0014, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, eposdie_return_list_Average_WMAMR_plot, label='WMAMR', color=color0015, lw=0.7, ls="-.")
# ax00.plot(Date_test_plot, episode_returns_DDPG_plot, label='BDA', color=color0018, lw=1.5)
# ax00.plot(Date_test_plot, episode_returns_DDPG_melt_plot, label='Mul-BDA', color=color0018, lw=1.5)
# # ax00.plot(Date_test_plot, eposdie_return_list_Average_BCRP_plot, label='BCRP', color=color0016, lw=0.7, ls ="-."))
# # ax00.plot(Date_test_plot, eposdie_return_list_Average_Best_plot, label='Best', color=color0017, lw=0.7, ls ="-."))
# ax00.legend(loc='center left')
# ax00.grid()
#
# ##############################################################
# # axs[1]
# if len(Date_test_plot) >= len(episode_returns_modified_by_risk_DDPG_plot):
#     episode_returns_modified_by_risk_DDPG_plot = np.concatenate((episode_returns_modified_by_risk_DDPG_plot, episode_returns_modified_by_risk_DDPG_plot[-1] * np.ones(len(Date_test_plot)-len(episode_returns_modified_by_risk_DDPG_plot))), axis = 0)
# #######################################################
# ax01 = axs[1]
# ax01.cla()
#
# color1 = 'darkcyan'
#
# ax01.tick_params(axis='y', labelcolor=color1, labelsize=12)
# ax01.tick_params(axis='x', labelsize=12)
# ax01.set_ylabel('Episode Return modified by risk', fontsize=15)
#
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_BK_plot, label='BK', color=color000, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_CRP_plot, label='CRP', color=color001, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_ONS_plot, label='ONS', color=color002, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_OLMAR_plot, label='OLMAR', color=color003, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_UP_plot, label='UP', color=color004, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_ONS_plot, label='ONS', color=color005, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_Anticor_plot, label='Anticor', color=color006, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_PAMR_plot, label='PAMR', color=color007, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_CORNK_plot, label='CORNK', color=color008, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_M0_plot, label='M0', color=color009, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_RMR_plot, label='RMR', color=color0010, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_CWMR_plot, label='CWMR', color=color0011, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_EG_plot, label='EG', color=color0012, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_SP_plot, label='SP', color=color0013, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_UBAH_plot, label='UBAH', color=color0014, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, eposide_return_modified_by_risk_list_Average_WMAMR_plot, label='WMAMR', color=color0015, lw=0.7, ls="-.")
# ax01.plot(Date_test_plot, episode_returns_modified_by_risk_DDPG_plot, label='DDPG-BL', color=color0018, lw=1.5, )
# # ax00.plot(Date_test_plot, eposdie_return_list_Average_BCRP_plot, label='BCRP', color=color0016, lw=0.7, ls ="-."))
# # ax00.plot(Date_test_plot, eposdie_return_list_Average_Best_plot, label='Best', color=color0017, lw=0.7, ls ="-."))
# ax01.legend(loc='center left')
# ax01.grid()
#
# plt.title(save_title, y=13.5, verticalalignment='bottom')
# # plt.title(save_title,verticalalignment='bottom')
# plt.savefig(f"{cwd}/{fig_name}")
# plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
# # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()
# # endregion
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

print("=================输出各个策略的累计收益，方差，夏普比例====================")
# region Description
def positive_day_calculation(daliy_return):
    daliy_return[daliy_return <= 0] = 0
    daliy_return[daliy_return > 0] = 1
    positive_day = np.sum(daliy_return)
    return positive_day

def sortino_ratio_calculation(daliy_return):
    return_negative = list()
    """设置无风险收益/最小接受收益"""
    # risk_free_rate = np.array(0.03)
    # min_acceptable_return = np.array(0.03)
    risk_free_rate = np.log(np.array(0.00) + 1)
    min_acceptable_return = np.log(np.array(0.00) + 1)
    for i in range(len(daliy_return)):
        if daliy_return[i] <= min_acceptable_return * (1/250):
            # print("return",daliy_return[i])
            return_negative.append(daliy_return[i])
    return_negative = np.array(return_negative)
    # print("return_negative", return_negative)
    # sortino_ratio = daliy_return.mean()/(np.var(return_negative)**(1/2))
    sortino_ratio = (daliy_return.mean() - min_acceptable_return* (1/250))/ (((np.sum((return_negative - min_acceptable_return * (1/250))**2))/(len(return_negative)))** (1 / 2))
    low_partial_standard_deviation = (((np.sum((return_negative - min_acceptable_return * (1/250))**2))/(len(return_negative)))** (1 / 2))
    # print("low_partial_standard_deviation", (np.sum((return_negative - risk_free_rate * (1/250))**2)/(len(return_negative) - 1))**(1/2))
    # print("(len(return_negative) - 1))", (len(return_negative) - 1))
    # print("(return_negative - risk_free_rate * (1/250))", (return_negative - risk_free_rate * (1/250)))
    # print("(return_negative - risk_free_rate * (1/250))**2", (return_negative - risk_free_rate * (1/250))**2)
    return sortino_ratio, low_partial_standard_deviation

def mean_and_variance_calculation(eposide_returns, name):
    # print("eposide_returns",eposide_returns)
    print("eposide_returns",len(eposide_returns)-1)
    eposide_return = np.log(eposide_returns[-1]/eposide_returns[0])
    daliy_return = np.log(eposide_returns[1:]/eposide_returns[:-1])
    return_mean = daliy_return.mean()
    variance = np.var(daliy_return)
    # print("return_mean", str(return_mean))
    # print("rf", (0.033 * (1/250)))
    """设置无风险收益"""
    # risk_free_rate = np.array(0.03)
    risk_free_rate = np.log(np.array(0.00) + 1)
    sharpe_ratio = (return_mean - (risk_free_rate * (1/250)))/(variance**(1/2))
    sortino_ratio, low_partial_standard_deviation = sortino_ratio_calculation(daliy_return)
    positive_day = positive_day_calculation(daliy_return)

    # plt.scatter(return_mean, variance)
    # plt.show()
    print(f"{name:>7}     |"f"    {eposide_return:8.6f}      {return_mean:7.6f}         {variance**(1/2):7.6f}      {sharpe_ratio:7.6f}   {positive_day:7.6f}   {low_partial_standard_deviation:7.6f}   {sortino_ratio:7.6f}")
    # print(return_mean)
    # print(variance)
    # print(sharpe_ratio)
    ### return eposide_return, return_mean, variance, sharpe_ratio
    ### output_data_list = [name, "eposide_return*100", "return_mean*100", "variance**(1/2)", "sharpe_ratio", "positive_day", "low_partial_standard_deviation", "sortino_ratio"]
    ### output_data_list = [str(eposide_return*100), str(return_mean*100), str(variance**(1/2)), str(sharpe_ratio), str(positive_day), str(low_partial_standard_deviation), str(sortino_ratio)]
    output_data_list = np.array((eposide_return, return_mean, variance ** (1 / 2), sharpe_ratio, positive_day, low_partial_standard_deviation, sortino_ratio))
    # print("eposide_return*100", np.float(eposide_return*100))
    # print("output_data_list", output_data_list.reshape(1,7).shape)
    return eposide_return, return_mean, variance, sharpe_ratio, output_data_list.reshape(1,7)

print(f"{'Strategy':>8}{'AR':>12}{'DR':>14}{'Std':>18}{'SR':>12} {'PD':>12}{'LPSD':>12}{'SOR':>12}")
output_data_list_including_all_strategies = np.ones((2, 7, 1))
eposide_return_DDPG_BL, return_mean_DDPG_BL, variance_DDPG_BL, sharpe_ratio_DDPG_BL, output_data_list_DDPG_BL = mean_and_variance_calculation(episode_total_assets_DDPG_plot, "EIIE")
# output_data_list_including_all_strategies.append(output_data_list_DDPG_BL)
output_data_list_including_all_strategies[0, :,0 ] = output_data_list_DDPG_BL
eposide_return_DDPG_BL_melt, return_mean_DDPG_BL_melt, variance_DDPG_BL_melt, sharpe_ratio_DDPG_BL_melt, output_data_list_DDPG_BL_melt = mean_and_variance_calculation(episode_total_assets_DDPG_melt_plot, "EIIE")
# output_data_list_including_all_strategies.append(output_data_list_DDPG_BL_melt)
eposide_return_CRP, return_mean_CRP, variance_CRP, sharpe_ratio_CRP, output_data_list_CRP = mean_and_variance_calculation(eposide_total_asset1_list_Average_CRP_plot, "CRP")
# output_data_list_including_all_strategies.append(output_data_list_CRP)
output_data_list_including_all_strategies[1, :,0 ] = output_data_list_CRP
"""
output_data_list_including_all_strategies[1, :,0  ] = output_data_list_DDPG_BL_melt
eposide_return_BK, return_mean_BK, variance_BK, sharpe_ratio_BK, output_data_list_BK = mean_and_variance_calculation(eposide_total_asset1_list_Average_BK_plot, "BK")
# output_data_list_including_all_strategies.append(output_data_list_BK)
output_data_list_including_all_strategies[2, :,0 ] = output_data_list_BK

eposide_return_ONS, return_mean_ONS, variance_ONS, sharpe_ratio_ONS, output_data_list_ONS = mean_and_variance_calculation(eposide_total_asset1_list_Average_ONS_plot, "ONS")
# output_data_list_including_all_strategies.append(output_data_list_ONS)
output_data_list_including_all_strategies[4, :,0 ] = output_data_list_ONS
eposide_return_OLMAR, return_mean_OLMAR, variance_OLMAR, sharpe_ratio_OLMAR, output_data_list_OLMAR = mean_and_variance_calculation(eposide_total_asset1_list_Average_OLMAR_plot, "OLMAR")
# output_data_list_including_all_strategies.append(output_data_list_OLMAR)
output_data_list_including_all_strategies[5, :,0 ] = output_data_list_OLMAR
eposide_return_UP, return_mean_UP, variance_UP, sharpe_ratio_UP, output_data_list_UP = mean_and_variance_calculation(eposide_total_asset1_list_Average_UP_plot, "UP")
# output_data_list_including_all_strategies.append(output_data_list_UP)
output_data_list_including_all_strategies[6, :,0 ] = output_data_list_UP
eposide_return_Anticor, return_mean_Anticor, variance_Anticor, sharpe_ratio_Anticor, output_data_list_Anticor, = mean_and_variance_calculation(eposide_total_asset1_list_Average_Anticor_plot, "Anticor")
# output_data_list_including_all_strategies.append(output_data_list_Anticor)
output_data_list_including_all_strategies[7, :,0 ] = output_data_list_Anticor
eposide_return_PAMR, return_mean_PAMR, variance_PAMR, sharpe_ratio_PAMR, output_data_list_PAMR = mean_and_variance_calculation(eposide_total_asset1_list_Average_PAMR_plot, "PAMR")
# output_data_list_including_all_strategies.append(output_data_list_PAMR)
output_data_list_including_all_strategies[8, :,0 ] = output_data_list_PAMR
eposide_return_CORNK, return_mean_CORNK, variance_CORNK, sharpe_ratio_CORNK, output_data_list_CORNK = mean_and_variance_calculation(eposide_total_asset1_list_Average_CORNK_plot, "CORNK")
# output_data_list_including_all_strategies.append(output_data_list_CORNK)
output_data_list_including_all_strategies[9, :,0 ] = output_data_list_CORNK
eposide_return_M0, return_mean_M0, variance_M0, sharpe_ratio_M0, output_data_list_M0 = mean_and_variance_calculation(eposide_total_asset1_list_Average_M0_plot, "M0")
# output_data_list_including_all_strategies.append(output_data_list_M0)
output_data_list_including_all_strategies[10, :,0 ] = output_data_list_M0
eposide_return_RMR, return_mean_RMR, variance_RMR, sharpe_ratio_RMR, output_data_list_RMR = mean_and_variance_calculation(eposide_total_asset1_list_Average_RMR_plot, "RMR")
# output_data_list_including_all_strategies.append(output_data_list_RMR)
output_data_list_including_all_strategies[11, :,0  ] = output_data_list_RMR
eposide_return_CWMR, return_mean_CWMR, variance_CWMR, sharpe_ratio_CWMR, output_data_list_CWMR = mean_and_variance_calculation(eposide_total_asset1_list_Average_CWMR_plot, "CWMR")
# output_data_list_including_all_strategies.append(output_data_list_CWMR)
output_data_list_including_all_strategies[12, :,0  ] = output_data_list_CWMR
eposide_return_EG, return_mean_EG, variance_EG, sharpe_ratio_EG, output_data_list_EG = mean_and_variance_calculation(eposide_total_asset1_list_Average_EG_plot, "EG")
# output_data_list_including_all_strategies.append(output_data_list_EG)
output_data_list_including_all_strategies[13, :,0  ] = output_data_list_EG
eposide_return_SP, return_mean_SP, variance_SP, sharpe_ratio_SP, output_data_list_SP = mean_and_variance_calculation(eposide_total_asset1_list_Average_SP_plot, "SP")
# output_data_list_including_all_strategies.append(output_data_list_SP)
output_data_list_including_all_strategies[14, :,0  ] = output_data_list_SP
eposide_return_UBAH, return_mean_UBAH, variance_UBAH, sharpe_ratio_UBAH, output_data_list_UBAH = mean_and_variance_calculation(eposide_total_asset1_list_Average_UBAH_plot, "UBAH")
# output_data_list_including_all_strategies.append(output_data_list_UBAH)
output_data_list_including_all_strategies[15 , :,0 ] = output_data_list_UBAH
eposide_return_WMAMR, return_mean_WMAMR, variance_WMAMR, sharpe_ratio_WMAMR, output_data_list_WMAMR = mean_and_variance_calculation(eposide_total_asset1_list_Average_WMAMR_plot, "WMAMR")
# output_data_list_including_all_strategies.append(output_data_list_WMAMR)
output_data_list_including_all_strategies[16, :,0  ] = output_data_list_WMAMR
eposide_return_Jorion_Bayes_plot, return_mean_Jorion_Bayes_plot, variance_Jorion_Bayes_plot, sharpe_ratio_Jorion_Bayes_plot, output_data_list_Jorion_Bayes = mean_and_variance_calculation(eposide_total_asset1_list_Average_Jorion_Bayes_plot, "AJB")
# output_data_list_including_all_strategies.append(output_data_list_Jorion_Bayes)
output_data_list_including_all_strategies[17, :,0  ] = output_data_list_Jorion_Bayes
eposide_return_Average_Rolling_exceptation_plot, return_mean_Average_Rolling_exceptation_plot, variance_Average_Rolling_exceptation_plot, sharpe_ratio_Average_Rolling_exceptation_plot, output_data_list_Average_Rolling_exceptation_plot = mean_and_variance_calculation(eposide_total_asset1_list_Average_Rolling_exceptation_plot, "ARE")
# output_data_list_including_all_strategies.append(output_data_list_Average_Rolling_exceptation_plot)
output_data_list_including_all_strategies[18, :,0  ] = output_data_list_Average_Rolling_exceptation_plot
eposide_return_Average_Kan_and_zhou_three_fund_plot, return_mean_Average_Kan_and_zhou_three_fund_plot, variance_Average_Kan_and_zhou_three_fund_plot, sharpe_ratio_Average_Kan_and_zhou_three_fund_plot, output_data_list_Average_Kan_and_zhou_three_fund_plot = mean_and_variance_calculation(eposide_total_asset1_list_Average_Kan_and_zhou_three_fund_plot, "KAZTF")
# output_data_list_including_all_strategies.append(output_data_list_Average_Kan_and_zhou_three_fund_plot)
output_data_list_including_all_strategies[19, :,0  ] = output_data_list_Average_Kan_and_zhou_three_fund_plot
"""

column= ["AR", "DR", "Std", "SR", "PD", "LPSD", "SOR"]
# print("index", index.shape())
index  = ["EIIE", "CRP"]   #, "BDA", "BK", "ONS", "OLMAR", "UP", "Anticor", "PAMR", "CORNK", "M0", "RMR", "CWMR", "EG", "SP", "UBAH", "WMAMR", "AJB", "ARE", "KAZTF"
# print("column", column.shape())
# import csv #调用数据保存文件
import pandas as pd #用于数据输出
# #一个sheet
# list1=[1,2]
# list2=[3,4]
# list=[]
# list.append(list1)
# list.append(list2)
# print(list)
# column=['one','two'] #列表头名称
# test=pd.DataFrame(columns=column,data=list)#将数据放进表格
# test.to_csv('test.csv') #数据存入csv,存储位置及文件名称


print("output_data_list_including_all_strategies", output_data_list_including_all_strategies.shape)
Empirical_result = pd.DataFrame(columns=column, data = output_data_list_including_all_strategies[:, :, 0], index = index)
pd.set_option('display.max_columns', None)#所有列
pd.set_option('display.max_rows', None)#所有行
print(Empirical_result)
episode_total_assets_DDPG_melt_plot
# Empirical_result = pd.DataFrame(data = output_data_list_including_all_strategies[:, :, 0])
# Empirical_result.save('mpirical_reslut_in_the_back_test.xls')
Empirical_result.to_csv("Empirical_reslut_in_the_back_test.csv", sep="," ,encoding='utf-8')
# endregion

print("=================对各个策略的累计收益变化和收益分布分别作图====================")
def draw_plot(Date_test, eposdie_return_list, episode_total_assets_list, save_title= None, Fig_name = None, CWD = "./test_ddpg"):
    mpl.use('Agg')
    save_title = save_title
    plt.figure(figsize=(10, 10), dpi=3000)
    # fig = plt.figure(dpi=100,constrained_layout=True) #类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
    # gs = GridSpec(2, 2, figure=fig) #GridSpec将fiure分为3行3列，每行三个axes，gs为一个matplotlib.gridspec.GridSpec对象，可灵活的切片figure
    fig, axs = plt.subplots(2, figsize=(15, 10), dpi=100)

    # axs[0]
    ax001 = axs[0]
    ax001.cla()
    color1 = 'darkcyan'
    ax001.tick_params(axis='y', labelcolor=color1, labelsize=12)
    ax001.tick_params(axis='x', labelsize=12)
    color000 = 'lightcoral'
    ax001.set_ylabel('Episode Return', fontsize=15)
    ax001.plot(Date_test, eposdie_return_list, label='logreturn', color=color000, lw=1.5)
    # ax00.plot(Date_train, episode_returns_modified_by_risk_DDPG_train, label='logreturn modified risk', color=color000, lw=1.5)
    # ax001.legend(loc='center left')
    ax001.grid()
    plt.title(save_title, y=13.5, verticalalignment='bottom')

    # axs[1]
    ax0011 = axs[1]
    ax0011.cla()
    daily_return_test = np.log(episode_total_assets_list[1:] / episode_total_assets_list[:-1])
    # print("daily_return_train:",daily_return_train)
    # bins = range(-30, 30, 1000)
    cm = plt.cm.get_cmap('Greens')
    # n, bins, patches = ax0001.hist(daily_return_train*100, 500, density=True, cumulative=-1, color='green', histtype='bar', label='logreturn modified risk')
    n, bins, patches = ax0011.hist(daily_return_test * 100, 60, density=True, color='green', histtype='bar',
                                   label='logreturn')
    ax0011.set_xlim(-30, 30)
    for c, p in zip(n, patches):
        plt.setp(p, 'facecolor', cm(1 - c))

    ax0011.legend(loc='center left')
    ax0011.grid()
    Fig_name = Fig_name
    # plt.title(save_title,verticalalignment='bottom')
    plt.savefig(f"{CWD}/{Fig_name}")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()
    # endregion

draw_plot(Date_test, episode_returns_DDPG_plot, episode_total_assets_DDPG_plot, save_title= None, Fig_name = "performance_in_test_set(DDPG-BL)", CWD = "./test_ddpg")

"""
draw_plot(Date_test, eposdie_return_list_Average_BK_plot, eposide_total_asset1_list_Average_BK_plot, save_title= None, Fig_name = "performance_in_test_set(BK)", CWD = "./test_ddpg")
draw_plot(Date_test, eposdie_return_list_Average_CRP_plot, eposide_total_asset1_list_Average_CRP_plot, save_title= None, Fig_name = "performance_in_test_set(CRP)", CWD = "./test_ddpg")
draw_plot(Date_test, eposdie_return_list_Average_Best_plot, eposide_total_asset1_list_Average_Best_plot, save_title= None, Fig_name = "performance_in_test_set(Best)", CWD = "./test_ddpg")
"""



print("================对收益分布做箱线图====================")
# region Description
recorder=None
cwd= "./test_ddpg"
save_title='performance in back test'
fig_name='plot_backtest_boxplot.jpg'

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
box_data = list()
daily_return_DPG_test = np.log(episode_total_assets_DDPG_plot[1:] / episode_total_assets_DDPG_plot[:-1])
box_data.append(daily_return_DPG_test)
daily_return_CRP_test = np.log(eposide_total_asset1_list_Average_CRP_plot[1:] / eposide_total_asset1_list_Average_CRP_plot[:-1])
box_data.append(daily_return_CRP_test)
"""
daily_return_BK_test = np.log2(eposide_total_asset1_list_Average_BK_plot[1:] / eposide_total_asset1_list_Average_BK_plot[:-1])
box_data.append(daily_return_BK_test)

daily_return_ONS_test = np.log2(eposide_total_asset1_list_Average_ONS_plot[1:] / eposide_total_asset1_list_Average_ONS_plot[:-1])
box_data.append(daily_return_ONS_test)
daily_return_OLMAR_test = np.log2(eposide_total_asset1_list_Average_OLMAR_plot[1:] / eposide_total_asset1_list_Average_OLMAR_plot[:-1])
box_data.append(daily_return_OLMAR_test)
daily_return_UP_test = np.log2(eposide_total_asset1_list_Average_UP_plot[1:] / eposide_total_asset1_list_Average_UP_plot[:-1])
box_data.append(daily_return_UP_test)
daily_return_Anticor_test = np.log2(eposide_total_asset1_list_Average_Anticor_plot[1:] / eposide_total_asset1_list_Average_Anticor_plot[:-1])
box_data.append(daily_return_Anticor_test)
daily_return_PAMA_test = np.log2(eposide_total_asset1_list_Average_PAMR_plot[1:] / eposide_total_asset1_list_Average_PAMR_plot[:-1])
box_data.append(daily_return_PAMA_test)
daily_return_CORNK_test = np.log2(eposide_total_asset1_list_Average_CORNK_plot[1:] / eposide_total_asset1_list_Average_CORNK_plot[:-1])
box_data.append(daily_return_CORNK_test)
daily_return_M0_test = np.log2(eposide_total_asset1_list_Average_M0_plot[1:] / eposide_total_asset1_list_Average_M0_plot[:-1])
box_data.append(daily_return_M0_test)
daily_return_RMR_test = np.log2(eposide_total_asset1_list_Average_RMR_plot[1:] / eposide_total_asset1_list_Average_RMR_plot[:-1])
box_data.append(daily_return_RMR_test)
daily_return_CWMR_test = np.log2(eposide_total_asset1_list_Average_CWMR_plot[1:] / eposide_total_asset1_list_Average_CWMR_plot[:-1])
box_data.append(daily_return_CWMR_test)
daily_return_EG_test = np.log2(eposide_total_asset1_list_Average_EG_plot[1:] / eposide_total_asset1_list_Average_EG_plot[:-1])
box_data.append(daily_return_EG_test)
daily_return_SP_test = np.log2(eposide_total_asset1_list_Average_SP_plot[1:] / eposide_total_asset1_list_Average_SP_plot[:-1])
box_data.append(daily_return_SP_test)
daily_return_UBAH_test = np.log2(eposide_total_asset1_list_Average_UBAH_plot[1:] /eposide_total_asset1_list_Average_UBAH_plot [:-1])
box_data.append(daily_return_UBAH_test)
daily_return_WMAMR_test = np.log2(eposide_total_asset1_list_Average_WMAMR_plot[1:] / eposide_total_asset1_list_Average_WMAMR_plot[:-1])
box_data.append(daily_return_WMAMR_test)
daily_return_BCRP_test = np.log2(eposide_total_asset1_list_Average_BCRP_plot[1:] / eposide_total_asset1_list_Average_BCRP_plot[:-1])
box_data.append(daily_return_BCRP_test)
daily_return_Best_test = np.log2(eposide_total_asset1_list_Average_Best_plot[1:] / eposide_total_asset1_list_Average_Best_plot[:-1])
box_data.append(daily_return_Best_test)
"""


plt.figure(figsize=(10, 10), dpi=3000)
# fig = plt.figure(dpi=100,constrained_layout=True) #类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
# gs = GridSpec(2, 2, figure=fig) #GridSpec将fiure分为3行3列，每行三个axes，gs为一个matplotlib.gridspec.GridSpec对象，可灵活的切片figure
fig, axs = plt.subplots(1, figsize=(15, 10), dpi=100)
eposide_total_asset1_list_Average_CRP_plot = np.array(eposide_total_asset1_list_Average_CRP)
"""
eposide_total_asset1_list_Average_BK_plot = np.array(eposide_total_asset1_list_Average_BK)

eposide_total_asset1_list_Average_ONS_plot = np.array(eposide_total_asset1_list_Average_ONS)
eposide_total_asset1_list_Average_OLMAR_plot = np.array(eposide_total_asset1_list_Average_OLMAR)
eposide_total_asset1_list_Average_UP_plot = np.array(eposide_total_asset1_list_Average_UP)
eposide_total_asset1_list_Average_Anticor_plot = np.array(eposide_total_asset1_list_Average_Anticor)
eposide_total_asset1_list_Average_PAMR_plot = np.array(eposide_total_asset1_list_Average_PAMR)
eposide_total_asset1_list_Average_CORNK_plot = np.array(eposide_total_asset1_list_Average_CORNK)
eposide_total_asset1_list_Average_M0_plot = np.array(eposide_total_asset1_list_Average_M0)
eposide_total_asset1_list_Average_RMR_plot = np.array(eposide_total_asset1_list_Average_RMR)
eposide_total_asset1_list_Average_CWMR_plot = np.array(eposide_total_asset1_list_Average_CWMR)
eposide_total_asset1_list_Average_EG_plot = np.array(eposide_total_asset1_list_Average_EG)
eposide_total_asset1_list_Average_SP_plot = np.array(eposide_total_asset1_list_Average_SP)
eposide_total_asset1_list_Average_UBAH_plot = np.array(eposide_total_asset1_list_Average_UBAH)
eposide_total_asset1_list_Average_WMAMR_plot = np.array(eposide_total_asset1_list_Average_WMAMR)
eposide_total_asset1_list_Average_BCRP_plot = np.array(eposide_total_asset1_list_Average_BCRP)
eposide_total_asset1_list_Average_Best_plot = np.array(eposide_total_asset1_list_Average_Best)
"""


episode_returns_modified_by_risk_DDPG_plot = np.array(episode_returns_modified_by_risk_DDPG)
episode_returns_DDPG_plot = np.array(episode_returns_DDPG)
episode_total_assets_DDPG_plot = np.array(episode_total_assets_DDPG)
# axs[0]
ax00 = axs
ax00.cla()

ax00.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax00.tick_params(axis='x', labelsize=12)

ax00.set_ylabel('Episode Return', fontsize=15)

ax00.boxplot(box_data,0,'',showmeans=True, vert=True)
# ax00.boxplot(daily_return_BK_test,1,'',showmeans=True, vert=True)

ax00.legend(loc='center left')
ax00.grid()

###############################################################################################################


plt.title(save_title, y=13.5, verticalalignment='bottom')
# plt.title(save_title,verticalalignment='bottom')
plt.savefig(f"{cwd}/{fig_name}")
plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
# plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()
# endregion

episode_total_assets = episode_total_assets_DDPG_plot
episode_total_assets = np.array(episode_total_assets)
episode_total_assets_back_test_config = {"episode_total_assets": episode_total_assets}
np.save('episode_total_assets_back_test_config.npy', episode_total_assets_back_test_config)