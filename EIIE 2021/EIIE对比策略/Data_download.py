import numpy as np
import pandas as pd
import pytz
import yfinance as yf

# 确定所需哟啊的下载的数据（与对应的参数）
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
# "DOW",
SSE_50_TICKER = [
    "600000.XSHG",
    "600036.XSHG",
    "600104.XSHG",
    "600030.XSHG",
    "601628.XSHG",
    "601166.XSHG",
    "601318.XSHG",
    "601328.XSHG",
    "601088.XSHG",
    "601857.XSHG",
    "601601.XSHG",
    "601668.XSHG",
    "601288.XSHG",
    "601818.XSHG",
    "601989.XSHG",
    "601398.XSHG",
    "600048.XSHG",
    "600028.XSHG",
    "600050.XSHG",
    "600519.XSHG",
    "600016.XSHG",
    "600887.XSHG",
    "601688.XSHG",
    "601186.XSHG",
    "601988.XSHG",
    "601211.XSHG",
    "601336.XSHG",
    "600309.XSHG",
    "603993.XSHG",
    "600690.XSHG",
    "600276.XSHG",
    "600703.XSHG",
    "600585.XSHG",
    "603259.XSHG",
    "601888.XSHG",
    "601138.XSHG",
    "600196.XSHG",
    "601766.XSHG",
    "600340.XSHG",
    "601390.XSHG",
    "601939.XSHG",
    "601111.XSHG",
    "600029.XSHG",
    "600019.XSHG",
    "601229.XSHG",
    "601800.XSHG",
    "600547.XSHG",
    "601006.XSHG",
    "601360.XSHG",
    "600606.XSHG",
    "601319.XSHG",
    "600837.XSHG",
    "600031.XSHG",
    "601066.XSHG",
    "600009.XSHG",
    "601236.XSHG",
    "601012.XSHG",
    "600745.XSHG",
    "600588.XSHG",
    "601658.XSHG",
    "601816.XSHG",
    "603160.XSHG",
]
HSI_50_TICKER = [
    "0011.HK",
    "0005.HK",
    "0012.HK",
    "0006.HK",
    "0003.HK",
    "0016.HK",
    "0019.HK",
    "0002.HK",
    "0001.HK",
    "0267.HK",
    "0101.HK",
    "0941.HK",
    "0762.HK",
    "0066.HK",
    "0883.HK",
    "2388.HK",
    "0017.HK",
    "0083.HK",
    "0939.HK",
    "0388.HK",
    "0386.HK",
    "3988.HK",
    "2628.HK",
    "1398.HK",
    "2318.HK",
    "3328.HK",
    "0688.HK",
    "0857.HK",
    "1088.HK",
    "0700.HK",
    "0836.HK",
    "1109.HK",
    "1044.HK",
    "1299.HK",
    "0151.HK",
    "1928.HK",
    "0027.HK",
    "2319.HK",
    "0823.HK",
    "1113.HK",
    "1038.HK",
    "2018.HK",
    "0175.HK",
    "0288.HK",
    "1997.HK",
    "2007.HK",
    "2382.HK",
    "1093.HK",
    "1177.HK",
    "2313.HK",
]
TRAIN_START_DATE = '2018-01-01'
TRAIN_END_DATE = '2020-12-31'


TECHNICAL_INDICATORS_LIST = [
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "dx_30",
    "close_30_sma"]

ERL_PARAMS = {
    "learning_rate": 1e-5,
    "batch_size": 128,
    "gamma": 0.9,
    "seed": 0,
    "net_dimension": 512,
    "target_step": 552,
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
print("price_array", price_array.shape)
print("close_in_OCLC_array", close_in_OCLC_array.shape)

# 确定环境参数
env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "OCLC_array": OCLC_array,
        "close_in_OCLC_array": close_in_OCLC_array,
        "if_train": True}

# 构建环境样例
from StockTradingEnv import StockTradingEnv
env = StockTradingEnv
env_instance = env(config=env_config)

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
# "DOW",

TECHNICAL_INDICATORS_LIST = [
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "dx_30",
    "close_30_sma"]
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
price_array_test, tech_array_test, turbulence_array_test, OCLC_array_test, close_in_OCLC_array_test = DP.df_to_array(data, if_vix)
env_test_config = {"price_array": price_array_test,
                   "tech_array": tech_array_test,
                   "turbulence_array": turbulence_array_test,
                   "OCLC_array": OCLC_array_test,
                   "close_in_OCLC_array": close_in_OCLC_array_test,
                   "if_train": False}

##############################################################
np.save('dict_env_config.npy', env_config)
np.save('dict_env_test_config.npy', env_test_config)

env_config = np.load('dict_env_config.npy', allow_pickle=True).item()
env_test_config = np.load('dict_env_test_config.npy', allow_pickle=True).item()