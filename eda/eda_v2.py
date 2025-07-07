import os
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_data
from config import Config

# Setup
sns.set(style="whitegrid")

os.makedirs(Config.plot_path_v2, exist_ok=True)

def save_plot(fig, filename):
    path = os.path.join(Config.plot_path_v2, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"✅ Saved plot: {path}")

def run_eda(df):
    print("📊 Running EDA...")

    # 1. Distribution of Taxi-Out Duration
    fig, ax = plt.subplots()
    sns.histplot(df["TAXI_OUT_DURATION"], bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of Taxi-Out Duration")
    save_plot(fig, "taxi_out_distribution.png")

    # 2. Boxplot by Aircraft Category
    #fig, ax = plt.subplots()
    #sns.boxplot(x="AIRCRAFT_CATEGORY", y="TAXI_OUT_DURATION", data=df, ax=ax)
    #ax.set_title("Taxi Duration by Aircraft Category")
    #save_plot(fig, "boxplot_aircraft_category.png")

    # 3. Taxi Duration by Hour
    fig, ax = plt.subplots()
    sns.boxplot(x="HOUR", y="TAXI_OUT_DURATION", data=df, ax=ax)
    ax.set_title("Taxi Duration by Hour of Day")
    save_plot(fig, "boxplot_by_hour.png")

    # 4. Day of Week Effect
    fig, ax = plt.subplots()
    sns.boxplot(x="DAY_OF_WEEK", y="TAXI_OUT_DURATION", data=df, ax=ax)
    ax.set_title("Taxi Duration by Day of Week")
    save_plot(fig, "boxplot_by_day.png")

    # 5. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_corr = df.select_dtypes(include=["int", "float"]).corr()
    sns.heatmap(numeric_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    save_plot(fig, "correlation_heatmap.png")

    # 6. Traffic Feature vs Duration (if present)
    traffic_cols = [col for col in df.columns if "TRAFFIC" in col]
    for col in traffic_cols:
        fig, ax = plt.subplots()
        sns.scatterplot(x=col, y="TAXI_OUT_DURATION", data=df, alpha=0.3, ax=ax)
        ax.set_title(f"{col} vs Taxi-Out Duration")
        save_plot(fig, f"scatter_{col.lower()}.png")

    print("🎉 EDA complete. All plots_v1 saved to eda/plots_v2/\n")

    file_path = os.path.join(Config.preprocessed_data_path, "preprocessed_taxi_data_v2.csv")
    df.to_csv(file_path, index=False)
    print("✅ Preprocessed data saved to 'mode/processed/preprocessed_taxi_data_v2.csv'")

if __name__ == "__main__":
    from feature_engineering.feature_engineering_v2 import engineer_features as engineer_features_v2

    df = load_data(Config.data_path_v2)
    df = engineer_features_v2(df)
    run_eda(df)
