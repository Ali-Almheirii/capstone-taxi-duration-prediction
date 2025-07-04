class Config:
    data_path = "data/original/FlightDetails_Jan2025_May2025.csv"
    model_save_path = "model_output/xgb_taxi_model.pkl"
    optuna_trials = 25
    test_size = 0.2
    random_seed = 42
    plot_path = "eda/plots"
    preprocessed_data_path = "data/processed/"