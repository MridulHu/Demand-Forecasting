import matplotlib.pyplot as plt
import seaborn as sns
import os

PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def plot_demand_trend(data):
    plt.figure(figsize=(14,6))
    plt.plot(data.index, data["Units Sold"])
    plt.title("Demand Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.savefig(f"{PLOT_DIR}/demand_trend.png")
    plt.close()


def plot_monthly_seasonality(data):
    monthly = data["Units Sold"].resample("M").mean()

    plt.figure(figsize=(12,6))
    plt.plot(monthly.index, monthly.values)
    plt.title("Monthly Average Demand")
    plt.xlabel("Month")
    plt.ylabel("Units Sold")
    plt.savefig(f"{PLOT_DIR}/monthly_demand.png")
    plt.close()


def plot_distribution(data):
    plt.figure(figsize=(8,6))
    sns.histplot(data["Units Sold"], bins=30, kde=True)
    plt.title("Distribution of Units Sold")
    plt.savefig(f"{PLOT_DIR}/units_sold_distribution.png")
    plt.close()


import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(data):

    numeric_data = data.select_dtypes(include=["number"])

    corr = numeric_data.corr()

    plt.figure(figsize=(14,10))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0
    )

    plt.title("Feature Correlation Heatmap")

    plt.savefig("outputs/plots/correlation_heatmap.png")
    plt.close()


def plot_rolling_average(data):

    rolling_mean = data["Units Sold"].rolling(30).mean()

    plt.figure(figsize=(14,6))

    plt.plot(data.index, data["Units Sold"], alpha=0.4)
    plt.plot(data.index, rolling_mean, linewidth=2)

    plt.title("Rolling Mean (30 days)")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")

    plt.savefig(f"{PLOT_DIR}/rolling_average.png")
    plt.close()


def plot_forecast(test_dates, actual, future_dates, preds):

    plt.figure(figsize=(14,6))

    plt.plot(test_dates, actual, label="Actual")
    plt.plot(future_dates, preds, "--", label="Forecast")

    plt.title("Demand Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")

    plt.legend()

    plt.savefig(f"{PLOT_DIR}/forecast.png")
    plt.close()


def plot_feature_importance(model):

    import xgboost as xgb

    plt.figure(figsize=(10,6))
    xgb.plot_importance(model, max_num_features=10)

    plt.title("Top Features Influencing Demand")

    plt.savefig(f"{PLOT_DIR}/feature_importance.png")
    plt.close()