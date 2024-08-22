import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from DRLAgent import DRLAgent

import numpy as np
env_config = np.load('dict_env_config.npy', allow_pickle=True).item()
# env_test_config = np.load('dict_env_test_config.npy', allow_pickle=True).item()

price_array = env_config["price_array"]
tech_array = env_config["tech_array"]
turbulence_array = env_config["turbulence_array"]


# price_array_test = env_test_config["price_array"]
# tech_array_test = env_test_config["tech_array"]
# turbulence_array_test = env_test_config["turbulence_array"]
from StockTradingEnv import StockTradingEnv
env = StockTradingEnv
env_instance = env(config=env_config)

# cwd = kwargs.get("cwd", "./" + str(model_name))
TRAIN_START_DATE = '2020-01-01'
TRAIN_END_DATE = '2022-12-31'

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
# drl_lib="elegantrl"
# model_name="ppo"
# cwd="./test_ppo"

break_step = kwargs.get("break_step", 1e6)
print("break_step",break_step)
erl_params = kwargs.get("erl_params")
print("erl_params",erl_params)
drl_lib = kwargs.get("drl_lib")
print("drl_lib",drl_lib)
model_name = kwargs.get("model_name")
print("model_name", model_name)
cwd = kwargs.get("cwd")
print("cwd", cwd)

DRLAgent_erl = DRLAgent
agent = DRLAgent_erl(env =env, price_array = price_array, tech_array = tech_array, turbulence_array = turbulence_array,)

model = agent.get_model(model_name, model_kwargs=erl_params)

print("model_name", model_name)
print("break_step", break_step)
trained_model = agent.train_model(
    model=model, cwd=cwd, total_timesteps=break_step)


