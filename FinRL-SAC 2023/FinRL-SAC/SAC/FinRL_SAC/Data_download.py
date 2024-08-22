import numpy as np
import pandas as pd
import pytz
import yfinance as yf
import StockTradingEnv

"""
1
"""
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
TRAIN_START_DATE = '2020-01-01'
TRAIN_END_DATE = '2022-12-31'
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
    "learning_rate": 1e-5,
    "batch_size": 128,
    "gamma": 0.9,
    "seed": 0,
    "net_dimension": 512,
    "target_step": 1080,
    "eval_gap": 30
}
kwargs={
        "start_date":TRAIN_START_DATE,
        "end_date":TRAIN_END_DATE,
        "ticker_list":DOW_30_TICKER,
        "time_interval":"1D",
        "technical_indicator_list":TECHNICAL_INDICATORS_LIST,
        "drl_lib":"elegantrl",
        "if_vix":True,
        "model_name":"sac",
        "cwd":"./test_sac",
        "erl_params":ERL_PARAMS,
        "break_step":3e5,
}
start_date= TRAIN_START_DATE
end_date= TRAIN_END_DATE
ticker_list=DOW_30_TICKER
time_interval="1D"

env = StockTradingEnv

technical_indicator_list=TECHNICAL_INDICATORS_LIST
drl_lib="elegantrl"
if_vix=True
env=env
model_name="sac"
cwd="./test_sac"
erl_params=ERL_PARAMS
break_step = 3e5
"""
2
"""

# YahooFinance=YahooFinanceProcessor
from DataProcessor import DataProcessor
DP = DataProcessor(data_source = "yahoofinance", **kwargs)

data = DP.download_data(ticker_list=ticker_list,
                       start_date=start_date,
                       end_date=end_date,
                       time_interval=time_interval)

data = DP.clean_data(data,time_interval)
print("data:",data)
if_vix=True
if if_vix:
    data = DP.add_vix(data,time_interval)
data.to_csv("datatrain.csv")

data = pd.read_csv("datatrain.csv")
data = data.drop(columns='Unnamed: 0')
print("data:",data)

data = DP.add_technical_indicator(data, technical_indicator_list)
print("data:",data)

"""
3
"""
d_vix = data.pop('vix')
data['vix'] = d_vix
if_vix=True
price_array, tech_array, turbulence_array = DP.df_to_array(data, if_vix)
env_config = {"price_array": price_array,
                   "tech_array": tech_array,
                   "turbulence_array": turbulence_array,
                   "if_train": False}
##############################################################
np.save('dict_env_config.npy', env_config)
# np.save('dict_env_test_config.npy', env_test_config)

env_config = np.load('dict_env_config.npy', allow_pickle=True).item()
# env_test_config = np.load('dict_env_test_config.npy', allow_pickle=True).item()