import numpy as np
import pandas as pd
import pytz
import yfinance as yf

# 确定所需哟啊的下载的数据（与对应的参数）
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
    "HON"
]
TRAIN_START_DATE = '2010-06-15'
TRAIN_END_DATE =  '2020-08-01'
TECHNICAL_INDICATORS_LIST = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]
ERL_PARAMS = {
    "learning_rate": 1e-8,
    "batch_size": 1024,
    "gamma": 0.9,
    "seed": 0,
    "net_dimension": 512,
    "target_step": 2516,
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
"break_step": 4e6}

from DataProcessor import DataProcessor
DP = DataProcessor(data_source = "yahoofinance",**kwargs)

# 确定下载数据的超参数参数
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
break_step = 4e6

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
price_array, tech_array, turbulence_array = DP.df_to_array(data, if_vix)



# 确定环境参数
env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": True}

# 构建环境样例
from StockTradingEnv import StockTradingEnv
env = StockTradingEnv
env_instance = env(config=env_config)

# 搭建回测环境
TEST_START_DATE = '2020-06-15'
TEST_END_DATE = '2021-03-01'
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
    "HON"]
TECHNICAL_INDICATORS_LIST = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma"]
# demo for elegantrl
start_date=TEST_START_DATE
#print("start_date:",start_date)
end_date=TEST_END_DATE
#print("end_date:",end_date)
ticker_list=DOW_30_TICKER
#print("ticker_list:",ticker_list)
data_source="yahoofinance"
#print("data_source:",data_source)
time_interval="1D"
#print("time_interval:",time_interval)
technical_indicator_list=TECHNICAL_INDICATORS_LIST
#print("technical_indicator_list:",technical_indicator_list)
drl_lib="elegantrl"
#print("drl_lib:",drl_lib)
env=env
#print("env:",env)
model_name="ddpg"
#print("model_name:",model_name)
cwd="./test_ddpg"
#print("cwd:",cwd)
net_dimension=512
#print("net_dimension:",net_dimension)

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
         "net_dimension": 128}
DP = DataProcessor(data_source = "yahoofinance", **kwargs)
data = DP.download_data(ticker_list, start_date, end_date, time_interval)
data = DP.clean_data(data, time_interval)
data = DP.add_technical_indicator(data, technical_indicator_list)
if_vix=True
if if_vix:
    data = DP.add_vix(data, time_interval)
price_array_test, tech_array_test, turbulence_array_test = DP.df_to_array(data, if_vix)
env_test_config = {"price_array": price_array_test,
                   "tech_array": tech_array_test,
                   "turbulence_array": turbulence_array_test,
                   "if_train": False}


# 确定DRL模型所需哟啊的参数
erl_params = ERL_PARAMS
break_step = kwargs.get("break_step", 4e6)
cwd = kwargs.get("cwd", "./" + str(model_name))

# 建立ageng
from Model import DRLAgent
DRLAgent_erl = DRLAgent
agent = DRLAgent_erl(env=env,
                     price_array=price_array,
                     tech_array=tech_array,
                     turbulence_array=turbulence_array,
                     price_array_test=price_array_test,
                     tech_array_test=tech_array_test,
                     turbulence_array_test=turbulence_array_test
                    )
# 建立模型
model = agent.get_model(model_name, model_kwargs=erl_params)
print("break step:", break_step)
# 训练模型
trained_model = agent.train_model(model = model,
                                  cwd = cwd,
                                  total_timesteps = break_step)