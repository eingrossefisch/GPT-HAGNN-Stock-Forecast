import torch
import pandas as pd
from dataset import load_test_data
from graph import Graph


def save_results(results, output_file_path):
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_file_path, index=False)
    print("预测完成，结果已写入", output_file_path)


class ModelPredictor:
    def __init__(self, model_path, test_data, company_dates):
        self.model = Graph(company_features_dimension=1, news_features_dimension=1, num_heads=1, dropout=0.6)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.test_data = test_data
        self.company_dates = company_dates

    def predict(self):
        results = []
        forecast_window_width = 2  # 使用前4天的新闻进行预测
        start_date_index = 0  # 从提供的开始日期索引开始
        max_start_date_index = len(self.test_data[0]) - forecast_window_width

        for day_start in range(start_date_index, max_start_date_index):
            company_features, daily_news_features, target_company_features = load_test_data(
                self.test_data, self.company_dates, day_start, forecast_window_width)

            company_features = torch.tensor(company_features, dtype=torch.float)
            daily_news_features = [torch.tensor(day, dtype=torch.float) for day in daily_news_features]

            with torch.no_grad():
                predicted = self.model(company_features, daily_news_features)
                factors = [[1.87789400e+03],[4.92800000e+01],[3.27100000e+01],[1.05200000e+02],[4.82900000e+01],[2.46000000e+01],[2.08900000e+01],[9.72873400e+01],
                           [2.70330800e+01],[1.94000000e+01],[6.64800000e+01],[1.75692308e+01],[9.87100000e+01],[5.16270390e+02],[7.12020400e+01]]
                # factors = [
                #    [1.88889400e+03, 1.91589400e+03, 1.86385400e+03, 1.87789400e+03, 1.87942722e+03, 9.64100000e+01],
                #    [4.92200000e+01, 5.05000000e+01, 4.85800000e+01, 4.92800000e+01, 4.93536734e+01, 3.86000000e+00],
                #    [3.25400000e+01, 3.27600000e+01, 3.20800000e+01, 3.27100000e+01, 3.24806834e+01, 1.76000000e+00],
                #    [1.04410000e+02, 1.06400000e+02, 1.03690000e+02, 1.05200000e+02, 1.04614452e+02, 5.29000000e+00],
                #    [4.83000000e+01, 4.90800000e+01, 4.79200000e+01, 4.82900000e+01, 4.84922894e+01, 2.42000000e+00],
                #    [2.46000000e+01, 2.48500000e+01, 2.43200000e+01, 2.46000000e+01, 2.46371686e+01, 1.20833333e+00],
                #    [2.07500000e+01, 2.13600000e+01, 2.05000000e+01, 2.08900000e+01, 2.08989437e+01, 1.23000000e+00],
                #    [9.91073400e+01, 9.91073400e+01, 9.57073400e+01, 9.72873400e+01, 9.73323969e+01, 7.18000000e+00],
                #    [2.70330800e+01, 2.72830800e+01, 2.66630800e+01, 2.70330800e+01, 2.69071499e+01, 1.61000000e+00],
                #    [1.93800000e+01, 1.95700000e+01, 1.90400000e+01, 1.94000000e+01, 1.92092085e+01, 1.65000000e+00],
                #    [6.61300000e+01, 6.65000000e+01, 6.21100000e+01, 6.64800000e+01, 6.50817673e+01, 4.47000000e+00],
                #    [1.76153846e+01, 1.79461539e+01, 1.73000000e+01, 1.75692308e+01, 1.76011430e+01, 1.17692308e+00],
                #    [9.69600000e+01, 1.01960000e+02, 9.14400000e+01, 9.87100000e+01, 9.62117694e+01, 9.60000000e+00],
                #    [5.14170390e+02, 5.29770390e+02, 4.90380390e+02, 5.16270390e+02, 5.04798888e+02, 4.89900000e+01],
                #    [7.22020400e+01, 7.34020400e+01, 7.06020400e+01, 7.12020400e+01, 7.18126467e+01, 4.82000000e+00]]
            # 逆标准化
            predicted = predicted.numpy() * factors
            target = target_company_features * factors

            results.append({
                'Date': self.company_dates[day_start + 195 + forecast_window_width],
                'Predicted Features': predicted,
                'Actual Features': target
            })

        return results
