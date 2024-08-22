import numpy as np
from StockTradingEnv import StockTradingEnv
import pandas as pd
from DataProcessor import DataProcessor
import warnings
warnings.filterwarnings("ignore")
TEST_START_DATE = '2023-01-01'
TEST_END_DATE = '2023-12-31'

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
    "DIS"
]

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

env = StockTradingEnv
# demo for elegantrl
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
print("DP",DP)
data = DP.download_data(ticker_list, start_date, end_date, time_interval)
data = DP.clean_data(data,time_interval)
print("data:",data)
data.to_csv("datatest.csv")
if_vix=True
#data = data.drop(columns='Unnamed: 0')
data = pd.read_csv("datatest.csv")
data = DP.add_technical_indicator(data, technical_indicator_list)

if if_vix:
    data = DP.add_vix(data,time_interval)

price_array_test, tech_array_test, turbulence_array_test = DP.df_to_array(data, if_vix)
print("data:",data)

env_back_test_config = {"price_array": price_array_test,
                   "tech_array": tech_array_test,
                   "turbulence_array": turbulence_array_test,
                   "if_train": False}
np.save('dict_back_test_env_test_config.npy', env_back_test_config)