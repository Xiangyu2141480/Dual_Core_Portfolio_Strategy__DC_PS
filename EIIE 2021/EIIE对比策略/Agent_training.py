import numpy as np
import pandas as pd
import pytz
import yfinance as yf

import numpy as np
env_config = np.load('dict_env_config.npy', allow_pickle=True).item()
env_test_config = np.load('dict_env_test_config.npy', allow_pickle=True).item()

price_array = env_config["price_array"]
tech_array = env_config["tech_array"]
turbulence_array = env_config["turbulence_array"]
OCLC_array = env_config["OCLC_array"]
close_in_OCLC_array = env_config["close_in_OCLC_array"]


price_array_test = env_test_config["price_array"]
tech_array_test = env_test_config["tech_array"]
turbulence_array_test = env_test_config["turbulence_array"]
OCLC_array_test = env_test_config["OCLC_array"]
close_in_OCLC_array_test = env_test_config["close_in_OCLC_array"]

from StockTradingEnv import StockTradingEnv
env = StockTradingEnv
env_instance = env(config=env_config)

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
# "DOW"
TRAIN_START_DATE = '2018-01-01'
TRAIN_END_DATE = '2020-12-31'

TECHNICAL_INDICATORS_LIST = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30"
    "close_30_sma",
    "close_60_sma",
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
break_step = 3e5

# 搭建回测环境
TEST_START_DATE = '2020-10-21'
TEST_END_DATE = '2021-12-31'
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
#  "DOW"
TECHNICAL_INDICATORS_LIST = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma"]

# 对应参量的输出检验
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
##########################################################
# 确定DRL模型所需哟啊的参数
erl_params = ERL_PARAMS
break_step = kwargs.get("break_step", 3e5)
cwd = kwargs.get("cwd", "./" + str(model_name))

# 建立agent
from Model import DRLAgent
DRLAgent_erl = DRLAgent
agent = DRLAgent_erl(env=env,
                     price_array=price_array,
                     tech_array=tech_array,
                     turbulence_array=turbulence_array,
                     OCLC_array=OCLC_array,
                     close_in_OCLC_array=close_in_OCLC_array,
                     price_array_test=price_array_test,
                     tech_array_test=tech_array_test,
                     turbulence_array_test=turbulence_array_test,
                     OCLC_array_test=OCLC_array_test,
                     close_in_OCLC_array_test = close_in_OCLC_array_test,
                    )
# 建立模型
model = agent.get_model(model_name, model_kwargs=erl_params)
print("break step:", break_step)
# 训练模型
trained_model = agent.train_model(model = model,
                                  cwd = cwd,
                                  total_timesteps = break_step)






