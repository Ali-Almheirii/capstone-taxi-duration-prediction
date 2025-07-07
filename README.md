
# ✈️ Capstone: Taxi-Out Duration Prediction

This project predicts aircraft taxi-out duration (specifically from pushback till runway).
Using flight operations data and machine learning.
Built as a capstone project, it leverages XGBoost (model), Optuna (for hyperparameter tuning), 
and extensive feature engineering.

---
## Next Steps:
- Try out other models such as: CatBoost, LightGBM, Stacking Model.
- Last resort try: MLP with embeddings OR Embedding + Deep Learning

- Note: Furhter enhancing feature engineering OR preprocessing pipeline is always a good idea.
---
## 📂 Project Structure

```
capstone-taxi-duration-prediction/
│
├── data/                        # Raw CSV data
├── model_output/               # Trained model (.pkl)
├── config.py                   # Global configs
├── data_loader.py              # Data loading logic
├── feature_engineering.py      # Feature creation
├── model.py                    # Training & evaluation
├── main.py                     # Pipeline runner
└── README.md                   # This file
```

---

## 🚀 How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/Ali-Almheiri/capstone-taxi-duration-prediction.git
cd capstone-taxi-duration-prediction
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Place Your Data

Put your flight dataset CSV inside the `data/` folder.  
Make sure the file name matches the one defined in `config.py`.

(update):
v1 is the first csv file shared in the WhatsApp group, v2 is the second csv file shared (shared on Friday 4-July)
### 4. Run the Full Pipeline

```bash
python main.py
```

This will:
- Load and clean the data
- Engineer features (e.g. time, aircraft type, traffic)
- Tune an XGBoost regressor using Optuna
- Print MAE, RMSE, R² metrics
- Save the model to `model_output/xgb_taxi_model.pkl`

---

## 📈 Features Used (not updated)

- Stand, runway, aircraft type
- Time-of-day, day-of-week
- Weekend & peak-hour flags
- Aircraft weight category
- Traffic in the last 30/60 minutes (by stand, terminal, runway)

---

## 🔧 Example Output

### v1:
```
📊 Model Performance Metrics: v1
Mean Absolute Error (MAE):		2.05 minutes
Root Mean Squared Error (RMSE):	2.59 minutes
Median Absolute Error:		1.72 minutes
R² Score:				0.4635
```

### v2:
```
📊 Model Performance Metrics: v2
Mean Absolute Error (MAE):		2.08 minutes
Root Mean Squared Error (RMSE):	2.62 minutes
Median Absolute Error:		1.76 minutes
R² Score:				0.4275
```

---
### Directories such as:
#### -> plots
##### -> model_output 
### will be populated as shown here
<img alt="img.png" height="600" src="img.png" width="250"/>
---
