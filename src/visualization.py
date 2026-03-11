import matplotlib.pyplot as plt
import seaborn as sns
import os

PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def plot_demand_trend(data):

    plt.figure(figsize=(14,6))
    plt.plot(data.index, data["Daily Demand"])
    plt.title("Daily Demand Trend")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")

    plt.savefig(f"{PLOT_DIR}/demand_trend.png")
    plt.close()


def plot_monthly_seasonality(data):

    monthly = data["Daily Demand"].resample("ME").mean()

    plt.figure(figsize=(12,6))
    plt.plot(monthly.index, monthly.values)

    plt.title("Monthly Average Demand")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")

    plt.savefig(f"{PLOT_DIR}/monthly_demand.png")
    plt.close()


def plot_distribution(data):

    plt.figure(figsize=(8,6))
    sns.histplot(data["Daily Demand"], bins=30, kde=True)

    plt.title("Demand Distribution")

    plt.savefig(f"{PLOT_DIR}/distribution.png")
    plt.close()


def plot_rolling_average(data):

    rolling = data["Daily Demand"].rolling(30).mean()

    plt.figure(figsize=(14,6))

    plt.plot(data.index, data["Daily Demand"], alpha=0.4)
    plt.plot(data.index, rolling)

    plt.title("30 Day Rolling Average")

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