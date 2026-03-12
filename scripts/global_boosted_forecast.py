import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


OUTPUT_DIR = "outputs/plots/global_forecasting"
os.makedirs(OUTPUT_DIR, exist_ok=True)


data = pd.read_csv("data/retail_store_inventory.csv")
data["Date"] = pd.to_datetime(data["Date"])


# --------------------------------
# Feature Engineering
# --------------------------------

def create_features(df):

    df = df.sort_values(["Store ID","Product ID","Date"])

    for lag in [1,2,3,7,14,21,28]:
        df[f"Lag_{lag}"] = (
            df.groupby(["Store ID","Product ID"])["Units Sold"]
            .shift(lag)
        )

    df["Rolling_Mean_7"] = (
        df.groupby(["Store ID","Product ID"])["Units Sold"]
        .shift(1)
        .rolling(7)
        .mean()
        .reset_index(level=[0,1], drop=True)
    )

    df["Momentum_7"] = (
        df.groupby(["Store ID","Product ID"])["Units Sold"]
        .shift(1)
        - df.groupby(["Store ID","Product ID"])["Units Sold"].shift(7)
    )

    df["Price_Diff"] = df["Price"] - df["Competitor Pricing"]

    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["IsWeekend"] = df["DayOfWeek"].isin([5,6]).astype(int)

    return df.dropna()


data = create_features(data)


# --------------------------------
# Encode categorical variables
# --------------------------------

data = pd.get_dummies(
    data,
    columns=[
        "Weather Condition",
        "Seasonality",
        "Category",
        "Region",
        "Store ID",
        "Product ID"
    ],
    drop_first=True
)


# --------------------------------
# Features
# --------------------------------

feature_cols = [c for c in data.columns if c not in ["Units Sold","Date"]]

X = data[feature_cols]
y = data["Units Sold"]


# --------------------------------
# Time-based split
# --------------------------------

split = int(len(data)*0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


# --------------------------------
# Model
# --------------------------------

model = XGBRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.9,
    n_jobs=-1,
    random_state=42
)


model.fit(X_train, y_train)


preds = model.predict(X_test)


# --------------------------------
# Plot for presentation
# --------------------------------

plt.figure(figsize=(12,4))

plt.plot(y_test.values[:300], label="Actual", alpha=0.8)
plt.plot(preds[:300], label="Prediction", linewidth=2)

plt.title("Global Demand Forecast (All Stores / Products)")
plt.xlabel("Time")
plt.ylabel("Units Sold")

plt.legend()

file = f"{OUTPUT_DIR}/global_prediction.png"

plt.savefig(file, dpi=300, bbox_inches="tight")
plt.close()

print("Saved →", file)