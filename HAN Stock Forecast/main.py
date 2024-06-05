from dataset import initialize_dataset
from train import ModelTrainer
from forecast_pn import ModelPredictor
from forecast_pn import save_results
import os


def main():

    # 指定文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建相对路径
    COMPANY_FILEPATH = os.path.join(current_dir, 'dataset', 'fin_data_normalized.xlsx')
    NEWS_FILEPATH = os.path.join(current_dir, 'dataset', 'news_data_normalized.xlsx')
    MODEL_PATH = os.path.join(current_dir, 'model.pth')
    OUTPUT_FILE = "C:/Users/user/Desktop/predicted_findata.xlsx"

    # 初始化数据集，获取训练集、测试集、日期、测试开始日期，以及标准化参数
    train_data, test_data, company_dates = initialize_dataset(COMPANY_FILEPATH, NEWS_FILEPATH)

    # 实例化并训练模型
    trainer = ModelTrainer(
        company_features_dim=1,
        news_features_dim=1,
        num_heads=3,
        dropout=0.6,
        learning_rate=0.35,  # 0.15,0.35,0.4
        epochs=2  # 5,5,7
    )
    trainer.train(train_data, company_dates)

    # 实例化预测器并执行预测
    predictor = ModelPredictor(
        model_path=MODEL_PATH,
        test_data=test_data,
        company_dates=company_dates  # 确保使用正确的命名和参数
    )
    results = predictor.predict()

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print("旧文件已删除。")

    # 保存预测结果
    save_results(results, OUTPUT_FILE)

    print("流程完成！结果已经保存。")


if __name__ == "__main__":
    main()
