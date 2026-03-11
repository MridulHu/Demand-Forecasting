import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# import future forecasting utility
from src.forecasting_utils import future_forecast


# Output directory
OUTPUT_DIR = "outputs/plots/product_experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load dataset
data = pd.read_csv("data/retail_store_inventory.csv")
data["Date"] = pd.to_datetime(data["Date"])


# Take first 10 products
products = data["Product ID"].unique()[:10]


# -------------------------------
# Feature Engineering
# -------------------------------
def create_features(df):

    df = df.sort_values("Date")

    # Lag features
    df["Lag_1"] = df["Units Sold"].shift(1)
    df["Lag_7"] = df["Units Sold"].shift(7)
    df["Lag_14"] = df["Units Sold"].shift(14)
    df["Lag_30"] = df["Units Sold"].shift(30)

    # Rolling averages
    df["Rolling_Mean_7"] = df["Units Sold"].rolling(7).mean()
    df["Rolling_Mean_14"] = df["Units Sold"].rolling(14).mean()
    df["Rolling_Mean_30"] = df["Units Sold"].rolling(30).mean()

    # Seasonality features
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month

    df = df.dropna()

    return df


# Feature list
features = [
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Lag_30",
    "Rolling_Mean_7",
    "Rolling_Mean_14",
    "Rolling_Mean_30",
    "DayOfWeek",
    "Month"
]


# -------------------------------
# Run experiment across products
# -------------------------------
for i in range(len(products) - 1):

    product_A = products[i]
    product_B = products[i + 1]

    df_A = data[data["Product ID"] == product_A].copy()
    df_B = data[data["Product ID"] == product_B].copy()

    df_A = create_features(df_A)
    df_B = create_features(df_B)

    X_A = df_A[features]
    y_A = df_A["Units Sold"]

    X_B = df_B[features]
    y_B = df_B["Units Sold"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_A, y_A, test_size=0.2, shuffle=False
    )

    # Improved RandomForest
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    pred_A = model.predict(X_test)

    # Apply model A on product B
    pred_B = model.predict(X_B.iloc[-len(pred_A):])

    # -------------------------------
    # Smooth results for visualization
    # -------------------------------
    y_test_s = pd.Series(y_test.values).rolling(14, center=True).mean()
    pred_A_s = pd.Series(pred_A).rolling(14, center=True).mean()
    pred_B_s = pd.Series(pred_B).rolling(14, center=True).mean()

    # -------------------------------
    # Plot 1: Product A forecast
    # -------------------------------
    plt.figure(figsize=(10,4))

    plt.plot(y_test_s.values,
             label=f"Actual {product_A}",
             alpha=0.7)

    plt.plot(pred_A_s,
             label=f"Forecast {product_A}",
             linewidth=2)

    plt.title(f"Forecast for {product_A} (model trained on same product)")
    plt.xlabel("Time")
    plt.ylabel("Units Sold")

    plt.legend()

    filename_A = f"{OUTPUT_DIR}/{product_A}_self_forecast.png"
    plt.savefig(filename_A, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------
    # Plot 2: Product B wrong model
    # -------------------------------
    plt.figure(figsize=(10,4))

    plt.plot(y_B.iloc[-len(pred_B_s):].values,
             label=f"Actual {product_B}",
             alpha=0.7)

    plt.plot(pred_B_s,
             label=f"Forecast {product_B} using {product_A} model",
             linewidth=2)

    plt.title(f"Model trained on {product_A} applied to {product_B}")
    plt.xlabel("Time")
    plt.ylabel("Units Sold")

    plt.legend()

    filename_B = f"{OUTPUT_DIR}/{product_A}_to_{product_B}.png"
    plt.savefig(filename_B, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------
    # Plot 3: Future Forecast
    # -------------------------------
    future_preds = future_forecast(model, y_A.values, steps=120)

    future_s = pd.Series(future_preds).rolling(14, center=True).mean()

    plt.figure(figsize=(10,4))

    plt.plot(
        future_s.values,
        linewidth=2,
        label=f"Future Forecast {product_A}"
    )

    # Add uncertainty band
    std = np.std(future_preds)
    upper = future_s + std
    lower = future_s - std

    plt.fill_between(
        range(len(future_s)),
        lower,
        upper,
        alpha=0.2,
        label="Forecast uncertainty"
    )

    plt.title(f"Future Demand Forecast for {product_A}")
    plt.xlabel("Future Time Steps")
    plt.ylabel("Units Sold")

    plt.legend()

    filename_future = f"{OUTPUT_DIR}/{product_A}_future_forecast.png"
    plt.savefig(filename_future, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved → {filename_A}")
    print(f"Saved → {filename_B}")
    print(f"Saved → {filename_future}")