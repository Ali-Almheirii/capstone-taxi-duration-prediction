from config import Config
from data_loader import load_data
from feature_engineering import engineer_features
from model import train_and_evaluate
from eda.eda import run_eda

if __name__ == "__main__":
    df = load_data(Config.data_path)
    df = engineer_features(df)

    run_eda(df)  # EDA will always run (if no required comment it out)

    train_and_evaluate(df)