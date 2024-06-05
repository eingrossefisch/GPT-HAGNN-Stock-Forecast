import torch
import torch.nn as nn
import os
import torch.optim as optim
from graph import Graph
from dataset import load_train_data


class ModelTrainer:
    def __init__(self, company_features_dim, news_features_dim, num_heads, dropout, learning_rate, epochs):
        self.company_features_dim = company_features_dim
        self.news_features_dim = news_features_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs

        # 实例化模型和优化器
        self.model = Graph(company_features_dimension=self.company_features_dim,
                           news_features_dimension=self.news_features_dim,
                           num_heads=self.num_heads, dropout=self.dropout)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, train_data, company_dates):
        num_days = len(train_data[0]) - 2  # 需要考虑三天的窗口

        # 训练模型
        for epoch in range(self.epochs):
            total_loss = 0
            for day_start in range(num_days):
                # 加载五天的数据，day_start 到 day_start+4
                company_features, daily_news_features, target_company_features = load_train_data(
                    train_data, company_dates, day_start, 2)
                #  print("Shape of company_features:", company_features.shape)
                #  print(f"company_features: ", company_features)
                #  for i, news_features in enumerate(daily_news_features):
                #      print(f"news_features[{i}]:", news_features)
                #  print(f"target_company_features: ", target_company_features)
                # 确保所有数据都是Tensor类型
                company_features = torch.tensor(company_features, dtype=torch.float32)
                daily_news_features = [torch.tensor(day, dtype=torch.float32) for day in daily_news_features]
                target_company_features = torch.tensor(target_company_features, dtype=torch.float32)

                self.optimizer.zero_grad()
                predicted_company_features = self.model(company_features, daily_news_features)
                loss = self.criterion(predicted_company_features, target_company_features)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f'Epoch {epoch + 1}, Loss: {total_loss:.4f}')

        # 保存模型
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, 'output', 'model.pth')
        torch.save(self.model.state_dict(), path)