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
TRAIN_START_DATE = '2018-01-01'
TRAIN_END_DATE =  '2020-12-31'
"""
Experiment1
TRAIN_START_DATE = '2018-01-01'
TRAIN_END_DATE =  '2020-12-31'

Experiment2
TRAIN_START_DATE = '2018-07-01'
TRAIN_END_DATE =  '2021-06-30'

Experiment3
TRAIN_START_DATE = '2019-01-01'
TRAIN_END_DATE =  '2021-12-31'

Experiment4
TRAIN_START_DATE = '2019-07-01'
TRAIN_END_DATE =  '2022-06-30'
"""
TECHNICAL_INDICATORS_LIST = [
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "dx_30",
    "close_30_sma"
]
ERL_PARAMS = {
    "learning_rate": 1e-8,
    "batch_size": 128,
    "gamma": 0.9,
    "seed": 0,
    "net_dimension": 512,
    "target_step": 500,
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
# endregion

# 下载数据
data = DP.download_data(ticker_list = ticker_list,
                        start_date = start_date,
                        end_date = end_date,
                        time_interval = DOW_30_TICKER)
# 数据清洗
data = DP.clean_data(data, time_interval="1D")
# 加入技术指标
data = DP.add_technical_indicator(data, technical_indicator_list)
if_vix=True
if if_vix:
    data = DP.add_vix(data, time_interval)

# 输出构建环境需要的数据矩阵
price_array, tech_array, turbulence_array, OCLC_array, close_in_OCLC_array = DP.df_to_array(data, if_vix)
"""====================================================================================================="""

# 确定环境参数
env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "OCLC_array": OCLC_array,
        "close_in_OCLC_array": close_in_OCLC_array,
        "if_train": False}

# 构建环境样例
# from StockTradingEnv import StockTradingEnv
# env = StockTradingEnv
# env_instance_train = env(config=env_config)


##################################################
# from Main import *
from DataProcessor import DataProcessor
# 回测参数确定
TEST_START_DATE = '2020-10-21'
TEST_END_DATE = '2021-12-31'
"""
Experiment 2021
TEST_START_DATE = '2020-10-21'
TEST_END_DATE = '2021-12-31'

Experiment 2023
TEST_START_DATE = '2022-10-20'
TEST_END_DATE = '2023-12-31'

"""
# TEST_END_DATE = '2023-01-06'
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
print("start_date:",start_date)
end_date=TEST_END_DATE
print("end_date:",end_date)
ticker_list=DOW_30_TICKER
print("ticker_list:",ticker_list)
data_source="yahoofinance"
print("data_source:",data_source)
time_interval="1D"
print("time_interval:",time_interval)
technical_indicator_list=TECHNICAL_INDICATORS_LIST
print("technical_indicator_list:",technical_indicator_list)
drl_lib="elegantrl"
print("drl_lib:",drl_lib)
env=env
print("env:",env)
model_name="ddpg"
print("model_name:",model_name)
cwd="./test_ddpg"
print("cwd:",cwd)
net_dimension=512
print("net_dimension:",net_dimension)
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
DP = DataProcessor(data_source = "yahoofinance", **kwargs)
data = DP.download_data(ticker_list, start_date, end_date, time_interval)
data = DP.clean_data(data, time_interval)
data = DP.add_technical_indicator(data, technical_indicator_list)
if_vix=True
if if_vix:
    data = DP.add_vix(data, time_interval)
price_array_test, tech_array_test, turbulence_array_test, OCLC_array_test, close_in_OCLC_array_test = DP.df_to_array(data, if_vix)
"""====================================================================================================="""
env_test_config = {"price_array": price_array_test,
                   "tech_array": tech_array_test,
                   "turbulence_array": turbulence_array_test,
                   "OCLC_array": OCLC_array_test,
                   "close_in_OCLC_array": close_in_OCLC_array_test,
                   "if_train": False}
"""======================================================================================================="""
##############################################################
np.save('dict_back_test_env_config.npy', env_config)
np.save('dict_back_test_env_test_config.npy', env_test_config)

env_config = np.load('dict_back_test_env_config.npy', allow_pickle=True).item()
env_test_config = np.load('dict_back_test_env_test_config.npy', allow_pickle=True).item()






