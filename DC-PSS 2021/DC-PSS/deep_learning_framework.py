# 导入必要的库
import pandas as pd
import numpy as np
from DataProcessor import DataProcessor
from Autoformer import Autoformer
from OtherModel import OtherModel

# 初始化数据处理器
data_processor = DataProcessor(data_source="yahoofinance")

# 下载和清洗数据
df_train = data_processor.download_data(ticker_list, start_date, end_date, time_interval)
df_train = data_processor.clean_data(df_train, time_interval)

# 添加技术指标和湍流
df_train = data_processor.add_technical_indicator(df_train, tech_indicator_list)
df_train = data_processor.add_turbulence(df_train)

# 将数据转换为数组
price_array, tech_array, turbulence_array = data_processor.df_to_array(df_train, if_vix)

# 初始化Autoformer模型和对比模型
autoformer = Autoformer(params)
other_model = OtherModel(params)

# 使用训练数据训练模型
autoformer.train(price_array, tech_array, turbulence_array)
other_model.train(price_array, tech_array, turbulence_array)

# 使用测试数据评估模型的性能
autoformer_score = autoformer.evaluate(test_data)
other_model_score = other_model.evaluate(test_data)

# 打印模型的性能
print("Autoformer Score: ", autoformer_score)
print("Other Model Score: ", other_model_score)

# 如果需要，进行预测
autoformer_predictions = autoformer.predict(new_data)
other_model_predictions = other_model.predict(new_data)

# 打印预测结果
print("Autoformer Predictions: ", autoformer_predictions)
print("Other Model Predictions: ", other_model_predictions)
