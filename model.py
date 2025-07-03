import xgboost as xgb
import optuna
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.preprocessing import LabelEncoder
from config import Config
import matplotlib.pyplot as plt


def train_and_evaluate(df):
    features = [
        "AIRPORT", "RUNWAY", "STAND", "AIRCRAFT_TYPE_ICAO", "AIRCRAFT_TYPE_IATA",
        "SERVICE_TYPE", "HOUR", "DAY_OF_WEEK", "IS_WEEKEND", "FLIGHT_NATURE", "DIVERSION_STATUS", "TERMINAL",
        "FLIGHT_DIRECTION",
        "TRAFFIC_LAST_30_MIN", "TRAFFIC_TO_SAME_RUNWAY_LAST_30_MIN", "TRAFFIC_IN_TERMINAL_LAST_30_MIN",
        "AIRCRAFT_CATEGORY",
        "IS_PEAK_HOUR"
    ]
    df = df.dropna(subset=features)
    X = df[features].copy()
    y = df["TAXI_OUT_DURATION"]

    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=Config.test_size, random_state=Config.random_seed)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "objective": "reg:squarederror",
            "random_state": Config.random_seed,
            "verbosity": 0,
            "tree_method": "hist"
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=Config.optuna_trials)

    print("Best parameters:", study.best_params)

    final_model = xgb.XGBRegressor(**study.best_params)
    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_val)

    xgb.plot_importance(final_model)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

    print("\n\n📊 Model Performance Metrics:")
    print(f"Mean Absolute Error (MAE):\t\t{mean_absolute_error(y_val, preds):.2f} minutes")
    print(f"Root Mean Squared Error (RMSE):\t{mean_squared_error(y_val, preds, squared=False):.2f} minutes")
    print(f"Median Absolute Error:\t\t{median_absolute_error(y_val, preds):.2f} minutes")
    print(f"R\u00b2 Score:\t\t\t\t{r2_score(y_val, preds):.4f}")

    joblib.dump(final_model, Config.model_save_path)

