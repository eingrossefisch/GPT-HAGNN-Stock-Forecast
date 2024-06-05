import pandas as pd
import numpy as np


class NewsDataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.load_and_preprocess_data()
        self.company_names = self.extract_company_names()

    def load_and_preprocess_data(self):
        # 加载Excel文件，设置第一列为时间索引
        data = pd.read_excel(self.filepath, header=0, index_col=0)
        return data

    def extract_company_names(self):
        # 提取公司名称并保持顺序
        company_names = list(self.data.columns)  # 直接从列名提取
        # print("公司名称列表:", company_names)
        return company_names

    def create_daily_vector(self, date):
        # 为指定日期创建1*15的向量
        daily_vector = self.data.loc[date]
        return np.array(daily_vector)

    def create_time_series_dataset(self):
        # 创建时间序列数据集，其中每个元素是一个1*15的向量
        news_time_series_data = []
        for date in self.data.index:
            daily_vector = self.create_daily_vector(date)
            news_time_series_data.append(daily_vector)
        return np.array(news_time_series_data)