from config import Config
from data_loader import load_data
from feature_engineering.feature_engineering_v2 import engineer_features as engineer_features_v2
from feature_engineering.feature_engineering_v1 import engineer_features as engineer_features_v1
from model.model_v2 import train_and_evaluate as train_and_evaluate_v2
from model.model_v1 import train_and_evaluate as train_and_evaluate_v1
from eda.eda_v2 import run_eda as run_eda_v2
from eda.eda_v1 import run_eda as run_eda_v1

def main(version="v1"):

    if version == "v1":
        data_path = Config.data_path_v1
        engineer_features = engineer_features_v1
        run_eda = run_eda_v1
        train_and_evaluate = train_and_evaluate_v1

    elif version == "v2":
        data_path = Config.data_path_v2
        engineer_features = engineer_features_v2
        run_eda = run_eda_v2
        train_and_evaluate = train_and_evaluate_v2

    else:
        raise ValueError(f"Unsupported version: {version}")

    df = load_data(data_path)
    df = engineer_features(df)

    run_eda(df)  # Optional: comment out if EDA is not needed
    train_and_evaluate(df)


if __name__ == "__main__":
    main(Config.version)  # Set version in Config (e.g., "v1" or "v2")
