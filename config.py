class Config:
    data_path_v1 = "data/original/FlightDetails_Jan2025_May2025_v1.csv"
    data_path_v2 = "data/original/FlightDetails_Jan2025_May2025_v2.csv"

    model_save_path_v1 = "model_output/xgb_taxi_model_v1.pkl"
    model_save_path_v2 = "model_output/xgb_taxi_model_v2.pkl"

    plot_path_v1 = "eda/plots_v1"
    plot_path_v2 = "eda/plots_v2"

    preprocessed_data_path = "data/processed/"

    optuna_trials = 25
    test_size = 0.2
    random_seed = 42

    # either "v1" or "v2"
    version = "v2"