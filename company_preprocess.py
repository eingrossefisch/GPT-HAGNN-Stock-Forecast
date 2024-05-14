import pandas as pd
import numpy as np
from collections import OrderedDict


class FinancialDataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.load_and_preprocess_data()
        self.company_names = self.extract_company_names()

    def load_and_preprocess_data(self):
        # 加载Excel文件，设置第一列为索引，并且假定第二行和第三行为多级列名
        data = pd.read_excel(self.filepath, header=[1, 2], index_col=0)
        # 因为已经将第一列设置为索引，剩下的才是公司数据列，合并两级列名为单一层
        data.columns = ['.'.join(col).strip() for col in data.columns.values]
        return data

    def extract_company_names(self):
        # 使用OrderedDict从左到右提取公司名，确保顺序和避免重复
        company_names = OrderedDict()
        for col in self.data.columns:
            company_name = col.split('.')[0]
            company_names[company_name] = None  # 使用None作为字典的值，只关心键的顺序
        return list(company_names.keys())

    def create_daily_tensor(self, date):
        # 为指定日期创建15*6的张量
        daily_data = []
        for company in self.company_names:
            company_columns = [f"{company}.{feature}" for feature in ['收盘价']]
         #                      ['开盘价', '最高价', '最低价', '收盘价', '均价', '涨跌']]
            company_data = self.data.loc[date, company_columns]
            daily_data.append(company_data.values)
        return np.array(daily_data)

    def create_time_series_dataset(self):
        # 创建时间序列数据集，其中每个元素是一个15*6的张量
        unique_dates = self.data.index.unique()
        company_time_series_data = []
        for date in unique_dates:
            daily_tensor = self.create_daily_tensor(date)
            company_time_series_data.append(daily_tensor)
        return np.array(company_time_series_data)
