import pandas as pd
import numpy as np
import torch
from company_preprocess import FinancialDataProcessor as CompanyProcessor
from news_preprocess import NewsDataProcessor


def align_news_data(company_dates, news_dates, news_data):
    # 创建以公司交易日为索引的新DataFrame，初始化时指定数据类型为 float
    aligned_news = pd.DataFrame(0.0, index=company_dates, columns=range(news_data.shape[1]))
    for date in news_dates:
        if date in aligned_news.index:
            # 确保 news_data 的相应行也是浮点类型
            aligned_news.loc[date] = news_data[news_dates.get_loc(date)].astype(float)
    return aligned_news.values


def split_data(company_data, news_data, train_ratio=0.8):
    # 计算训练集大小
    total_samples = len(company_data)
    train_size = int(total_samples * train_ratio)
    # 分割数据集
    train_data = (company_data[:train_size], news_data[:train_size])
    test_data = (company_data[train_size:], news_data[train_size:])
    return train_data, test_data, train_size


def initialize_dataset(company_filepath, news_filepath):
    # 加载公司数据
    company_processor = CompanyProcessor(company_filepath)
    company_data = company_processor.create_time_series_dataset()
    # print(company_processor.data.index)
    company_dates = pd.to_datetime(company_processor.data.index)  # 获取公司数据的日期索引
    # 加载新闻数据
    news_processor = NewsDataProcessor(news_filepath)
    news_data = news_processor.create_time_series_dataset()
    news_dates = pd.to_datetime(news_processor.data.index)  # 获取新闻数据的日期索引
    # 对齐新闻数据
    aligned_news_data = align_news_data(company_dates, news_dates, news_data)
    # 分割数据为训练集和测试集
    train_data, test_data, train_size = split_data(company_data, aligned_news_data)
    return train_data, test_data, company_dates


def load_train_data(train_data, company_dates, day_start_index, window_width):
    # 假设 day_start_index 是从 0 开始的整数索引
    day_start = company_dates[day_start_index]  # 直接使用索引获取日期
    end_index = day_start_index + window_width

    company_features = train_data[0][day_start_index]  # 第0天的公司数据
    daily_news_features = train_data[1][day_start_index:end_index]  # 包括第0到第3天的数据
    target_company_features = train_data[0][end_index]  # 第4天的数据

    return company_features, daily_news_features, target_company_features


def load_test_data(test_data, company_dates, day_start_index, window_width):
    # 假设 day_start_index 是从 0 开始的整数索引
    day_start = company_dates[day_start_index]  # 直接使用索引获取日期
    end_index = day_start_index + window_width

    company_features = test_data[0][day_start_index]  # 第0天的公司数据
    daily_news_features = test_data[1][day_start_index:end_index]  # 包括第0到第3天的数据
    target_company_features = test_data[0][end_index]  # 第4天的数据

    return company_features, daily_news_features, target_company_features

# if __name__ == '__main__':
#    train_data, test_data, company_dates = initialize_dataset(COMPANY_FILEPATH, NEWS_FILEPATH)
#    day_start = '2023-01-03'
#    num_days = 5
#    company_features, daily_news_features, target_company_features = load_data(train_data, company_dates, day_start,
#                                                                               num_days)

#   print("第一天公司特征张量:", company_features)
#    print("每日新闻向量:", daily_news_features)
#    print("第五天公司特征向量:", target_company_features)
