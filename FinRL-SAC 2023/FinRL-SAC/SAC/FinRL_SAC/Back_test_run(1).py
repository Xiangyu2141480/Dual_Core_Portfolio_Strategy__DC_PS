import numpy as np
from StockTradingEnv import StockTradingEnv
from DRLAgent import DRLAgent

env_back_test_config = np.load('dict_back_test_env_test_config.npy', allow_pickle=True).item()

price_array_test = env_back_test_config["price_array"]
tech_array_test = env_back_test_config["tech_array"]
turbulence_array_test = env_back_test_config["turbulence_array"]

env_test_config = {"price_array": price_array_test,
                   "tech_array": tech_array_test,
                   "turbulence_array": turbulence_array_test,
                   "if_train": False}

env = StockTradingEnv
env_instance_test = env(config=env_test_config)
TEST_START_DATE = '2021-01-01'
TEST_END_DATE = '2021-12-31'

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
model_name = "sac"
drl_lib = "elegantrl"
kwargs ={"start_date": TEST_START_DATE,
         "end_date": TEST_END_DATE,
         "ticker_list": DOW_30_TICKER,
         "time_interval": "1D",
         "technical_indicator_list": TECHNICAL_INDICATORS_LIST,
         "drl_lib": "elegantrl",
         "env": env,
         "model_name": "sac",
         "cwd": "./test_sac",
         "net_dimension": 512}

net_dimension = kwargs.get("net_dimension", 2 ** 8)
print("net_dimension:",net_dimension)
print("model_name:",model_name)
cwd = kwargs.get("cwd", "./" + str(model_name))
print("cwd:",cwd)
print("price_array: ", len(price_array_test))

DRLAgent_erl = DRLAgent
if drl_lib == "elegantrl":
    episode_total_assets = DRLAgent_erl.DRL_prediction(
        model_name=model_name,
        cwd=cwd,
        net_dimension=net_dimension,
        environment=env_instance_test)
print("episode_total_assets:",episode_total_assets)
print("episode_total_assets length:",len(episode_total_assets) - 1)


def positive_day_calculation(daliy_return):
    daliy_return[daliy_return <= 0] = 0
    daliy_return[daliy_return > 0] = 1
    positive_day = np.sum(daliy_return)
    return positive_day


# def sortino_ratio_calculation(daliy_return):
#     return_negative = list()
#     for i in range(len(daliy_return)):
#         if daliy_return[i] <= 0:
#             # print("return",daliy_return[i])
#             return_negative.append(daliy_return[i])
#     return_negative = np.array(return_negative)
#     # print("return_negative", return_negative)
#     sortino_ratio = daliy_return.mean()/(np.var(return_negative)**(1/2))
#     variance_negative = (np.var(return_negative) ** (1 / 2))
#     return sortino_ratio, variance_negative


# def mean_and_variance_calculation(eposide_returns, name):
#     print("eposide_returns",len(eposide_returns))
#     eposide_return = np.log2(eposide_returns[-1]/eposide_returns[0])
#     daliy_return = np.log2(eposide_returns[1:]/eposide_returns[:-1])
#     return_mean = daliy_return.mean()
#     variance = np.var(daliy_return)

#     sharpe_ratio = return_mean/(variance**(1/2))
#     sortino_ratio, variance_negative = sortino_ratio_calculation(daliy_return)
#     positive_day = positive_day_calculation(daliy_return)

#     # plt.scatter(return_mean, variance)
#     # plt.show()
#     print(f"{name:>7}     |"f"    {eposide_return*100:8.6f}        {return_mean*100:7.6f}                {(variance)**(1/2):7.6f}      {sharpe_ratio:7.6f}   {positive_day:7.6f}   {variance_negative:7.6f}   {sortino_ratio:7.6f}")
#     # print(return_mean)
#     # print(variance)
#     # print(sharpe_ratio)
#     return eposide_return, return_mean, variance, sharpe_ratio

def sortino_ratio_calculation(daliy_return):
    return_negative = list()
    """设置无风险收益/最小接受收益"""
    # risk_free_rate = np.array(0.03)
    # min_acceptable_return = np.array(0.03)
    risk_free_rate = 0.0
    min_acceptable_return = 0.0
    for i in range(len(daliy_return)):
        if daliy_return[i] <= min_acceptable_return * (1 / 250):
            # print("return",daliy_return[i])
            return_negative.append(daliy_return[i])
    return_negative = np.array(return_negative)
    # print("return_negative", return_negative)
    # sortino_ratio = daliy_return.mean()/(np.var(return_negative)**(1/2))
    sortino_ratio = (daliy_return.mean() - min_acceptable_return * (1 / 250)) / (
                ((np.sum((return_negative - min_acceptable_return * (1 / 250)) ** 2)) / (len(return_negative))) ** (1 / 2))
    low_partial_standard_deviation = (
                ((np.sum((return_negative - min_acceptable_return * (1 / 250)) ** 2)) / (len(return_negative) )) ** (1 / 2))
    # print("low_partial_standard_deviation", (np.sum((return_negative - risk_free_rate * (1/250))**2)/(len(return_negative) - 1))**(1/2))
    # print("(len(return_negative) - 1))", (len(return_negative) - 1))
    # print("(return_negative - risk_free_rate * (1/250))", (return_negative - risk_free_rate * (1/250)))
    # print("(return_negative - risk_free_rate * (1/250))**2", (return_negative - risk_free_rate * (1/250))**2)
    return sortino_ratio, low_partial_standard_deviation

def mean_and_variance_calculation(eposide_returns, name):
    # print("eposide_returns", len(eposide_returns))
    # print("eposide_returns", eposide_returns)
    risk_free_rate = 0.0
    eposide_return = np.log(eposide_returns[-1] / eposide_returns[0])
    daliy_return = np.log(eposide_returns[1:] / eposide_returns[:-1])
    # eposide_return = np.sum(eposide_returns)
    # daliy_return = eposide_returns
    return_mean = daliy_return.mean()
    variance = np.var(daliy_return)
    sharpe_ratio = (return_mean - risk_free_rate)/ (variance ** (1 / 2))
    sortino_ratio, low_partial_standard_deviation = sortino_ratio_calculation(daliy_return)
    positive_day = positive_day_calculation(daliy_return)

    # plt.scatter(return_mean, variance)
    # plt.show()
    Index = "Index"
    AR = "AR"
    DR = "DR"
    Std = "Std"
    SR = "SR"
    PD = "PD"
    LPSD = "LPSD"
    print("Algorithm name       AR              DR                 Std              SR           PD          LPSD       SOR")
    print(
        f"{name:>7}     |"f"    {eposide_return:8.6f}        {return_mean:7.6f}          {(variance) ** (1 / 2):7.6f}         {sharpe_ratio:7.6f}     {positive_day:7.6f}   {low_partial_standard_deviation:7.6f}   {sortino_ratio:7.6f}")
    output_data_list = np.array((eposide_return, return_mean, variance ** (1 / 2), sharpe_ratio,
                                 positive_day, low_partial_standard_deviation, sortino_ratio))
    # print(return_mean)
    # print(variance)
    # print(sharpe_ratio)
    return eposide_return, return_mean, variance, sharpe_ratio, output_data_list

episode_total_assets = np.array(episode_total_assets)
eposide_return_SAC_FinRL, return_mean_SAC_FinRL, variance_SAC_FinRL, sharpe_ratio_SAC_FinRL, output_data_list_FinRL = mean_and_variance_calculation(episode_total_assets, "DDPG-FinRL")

output_data_list_including_all_strategies = np.ones((1, 7, 1))

output_data_list_including_all_strategies[0, :,0 ] = output_data_list_FinRL

column= ["AR", "DR", "Std", "SR", "PD", "LPSD", "SOR"]
# print("index", index.shape())
index  = ["FinRL-SAC"]
import pandas as pd
Empirical_result = pd.DataFrame(columns=column, data = output_data_list_including_all_strategies[:, :, 0], index = index)
Empirical_result.to_csv("Empirical_reslut_in_the_back_test.csv", sep="," ,encoding='utf-8')

episode_total_assets = np.array(episode_total_assets)

episode_total_assets_back_test_config = {"episode_total_assets": episode_total_assets}
np.save('episode_total_assets_back_test_config.npy', episode_total_assets_back_test_config)