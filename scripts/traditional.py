import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = "data/Historical_Product_Demand.csv"
OUTPUT_DIR = "outputs/traditional"

WINDOW = 7
ALPHA_1 = 0.2
ALPHA_2 = 0.6

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Load Dataset
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# Convert date
df["Date"] = pd.to_datetime(df["Date"])

# Clean demand column (dataset contains parentheses)
df["Order_Demand"] = df["Order_Demand"].astype(str)
df["Order_Demand"] = df["Order_Demand"].str.replace("(", "-", regex=False)
df["Order_Demand"] = df["Order_Demand"].str.replace(")", "", regex=False)

df["Order_Demand"] = pd.to_numeric(df["Order_Demand"], errors="coerce")

# Remove missing values
df = df.dropna(subset=["Order_Demand"])


# -----------------------------
# Choose Product with Most Data
# -----------------------------

product_counts = df.groupby("Product_Code").size().reset_index(name="count")

best_product = product_counts.sort_values("count", ascending=False).iloc[0]["Product_Code"]

data = df[df["Product_Code"] == best_product].copy()

data = data.sort_values("Date")

print("Using Product:", best_product)


# -----------------------------
# Aggregate Demand Per Day
# -----------------------------
data = data.groupby("Date")["Order_Demand"].sum().reset_index()

# -----------------------------
# Log transformation to reduce demand spikes
# -----------------------------
data["Order_Demand"] = np.log1p(data["Order_Demand"])

sales = data["Order_Demand"]

print("Total observations:", len(data))


# -----------------------------
# Simple Moving Average
# -----------------------------

data["SMA"] = sales.rolling(WINDOW).mean()


# -----------------------------
# Weighted Moving Average
# -----------------------------

weights = np.array([0.28, 0.24, 0.18, 0.12, 0.08, 0.06, 0.04])

data["WMA"] = sales.rolling(WINDOW).apply(
    lambda x: np.sum(weights * x),
    raw=True
)


# -----------------------------
# Exponential Smoothing
# -----------------------------

data["ES_02"] = sales.ewm(alpha=ALPHA_1, adjust=False).mean()
data["ES_06"] = sales.ewm(alpha=ALPHA_2, adjust=False).mean()


# -----------------------------
# Error Metrics
# -----------------------------

def calculate_errors(actual, forecast):

    actual = actual.dropna()
    forecast = forecast.dropna()

    actual, forecast = actual.align(forecast, join="inner")

    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mad = np.mean(np.abs(actual - forecast))

    # avoid division by zero for MAPE
    non_zero = actual != 0
    mape = np.mean(np.abs((actual[non_zero] - forecast[non_zero]) / actual[non_zero])) * 100

    return mse, rmse, mad, mape


methods = ["SMA", "WMA", "ES_02", "ES_06"]

results = []

print("\nForecast Error Metrics\n")

for m in methods:

    mse, rmse, mad, mape = calculate_errors(sales, data[m])

    results.append({
        "Method": m,
        "MSE": mse,
        "RMSE": rmse,
        "MAD": mad,
        "MAPE": mape
    })

    print(f"{m}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAD  : {mad:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print()


results_df = pd.DataFrame(results)

results_df.to_csv(f"{OUTPUT_DIR}/error_metrics.csv", index=False)

print("Saved error metrics to:", f"{OUTPUT_DIR}/error_metrics.csv")


# -----------------------------
# Next Forecast
# -----------------------------

forecast_sma = sales.tail(WINDOW).mean()
forecast_wma = np.sum(weights * sales.tail(WINDOW))
forecast_es02 = data["ES_02"].iloc[-1]
forecast_es06 = data["ES_06"].iloc[-1]


print("\nNext Demand Forecast")

print("SMA Forecast:", forecast_sma)
print("WMA Forecast:", forecast_wma)
print("ES (alpha=0.2):", forecast_es02)
print("ES (alpha=0.6):", forecast_es06)


# -----------------------------
# Plot Forecast Comparison
# -----------------------------

plots = {
    "SMA": "Simple Moving Average",
    "WMA": "Weighted Moving Average",
    "ES_02": "Exponential Smoothing (α=0.2)",
    "ES_06": "Exponential Smoothing (α=0.6)"
}

for key, name in plots.items():

    plt.figure(figsize=(14,6))

    plt.plot(
        data["Date"],
        sales,
        label="Actual Demand",
        color="black",
        alpha=0.4,
        linewidth=2
    )

    plt.plot(
        data["Date"],
        data[key],
        label=name,
        linewidth=3
    )

    plt.title(f"Actual vs {name}", fontsize=16)

    plt.xlabel("Date")
    plt.ylabel("Log Demand")

    plt.grid(True, linestyle="--", alpha=0.4)

    plt.legend()

    plt.tight_layout()

    plot_path = f"{OUTPUT_DIR}/{key}_forecast_plot.png"

    plt.savefig(plot_path, dpi=300)

    print("Saved plot:", plot_path)

    plt.close()