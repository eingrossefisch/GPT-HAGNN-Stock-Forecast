import torch
import pandas as pd
import openpyxl
from openpyxl.styles import Font
from sklearn.metrics import f1_score
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
        all_predicted_changes = []
        all_actual_changes = []
        forecast_window_width = 2
        start_date_index = 0
        max_start_date_index = len(self.test_data[0]) - forecast_window_width
        previous_predicted = None
        previous_actual = None

        for day_start in range(start_date_index, max_start_date_index):
            company_features, daily_news_features, target_company_features = load_test_data(
                self.test_data, self.company_dates, day_start, forecast_window_width)

            company_features = torch.tensor(company_features, dtype=torch.float)
            daily_news_features = [torch.tensor(day, dtype=torch.float) for day in daily_news_features]

            with torch.no_grad():
                predicted = self.model(company_features, daily_news_features)

            predicted = predicted.numpy().flatten()
            target = target_company_features.flatten()

            if previous_predicted is not None:
                predicted_changes = ['p' if pred > prev else 'f' for pred, prev in zip(predicted, previous_predicted)]
                actual_changes = ['p' if act > prev else 'f' for act, prev in zip(target, previous_actual)]
            else:
                predicted_changes = ['N/A'] * len(predicted)
                actual_changes = ['N/A'] * len(target)

            all_predicted_changes.extend(predicted_changes)
            all_actual_changes.extend(actual_changes)

            results.append({
                'Date': self.company_dates[day_start + 193 + forecast_window_width],
                'Predicted Changes': predicted_changes,
                'Actual Changes': actual_changes
            })

            previous_predicted = predicted
            previous_actual = target

        valid_predicted_changes = [p for p in all_predicted_changes if p != 'N/A']
        valid_actual_changes = [a for a in all_actual_changes if a != 'N/A']

        MicroF1 = f1_score(valid_actual_changes, valid_predicted_changes, average='micro')
        print("MicroF1: ", MicroF1)
        MacroF1 = f1_score(valid_actual_changes, valid_predicted_changes, average='macro')
        print("MacroF1: ", MacroF1)
        WeightedF1 = f1_score(valid_actual_changes, valid_predicted_changes, average='weighted')
        print("WeightedF1: ", WeightedF1)

        return results