import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


OUTPUT_DIR = "outputs/plots/boosted_forecasting"
os.makedirs(OUTPUT_DIR, exist_ok=True)


data = pd.read_csv("data/retail_store_inventory.csv")
data["Date"] = pd.to_datetime(data["Date"])


# ---------------------------
# Encode categorical variables
# ---------------------------

data = pd.get_dummies(
    data,
    columns=[
        "Weather Condition",
        "Seasonality",
        "Category",
        "Region"
    ],
    drop_first=True
)


# ---------------------------
# Feature Engineering
# ---------------------------

def create_features(df):

    df = df.sort_values("Date")

    for lag in [1,2,3,7,14,21,28]:
        df[f"Lag_{lag}"] = df["Units Sold"].shift(lag)

    df["Rolling_Mean_7"] = df["Units Sold"].shift(1).rolling(7).mean()
    df["Rolling_Mean_14"] = df["Units Sold"].shift(1).rolling(14).mean()
    df["Rolling_Std_7"] = df["Units Sold"].shift(1).rolling(7).std()

    df["Momentum_7"] = df["Units Sold"].shift(1) - df["Units Sold"].shift(7)
    df["Momentum_14"] = df["Units Sold"].shift(1) - df["Units Sold"].shift(14)

    df["Price_Diff"] = df["Price"] - df["Competitor Pricing"]

    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["IsWeekend"] = df["DayOfWeek"].isin([5,6]).astype(int)

    df = df.dropna()

    return df


# ---------------------------
# Recursive Forecast
# ---------------------------

def recursive_forecast(model, df, feature_cols, steps=120):

    history = list(df["Units Sold"].values)
    preds = []

    last_row = df.iloc[-1].copy()

    for _ in range(steps):

        for lag in [1,2,3,7,14,21,28]:
            last_row[f"Lag_{lag}"] = history[-lag]

        X_future = pd.DataFrame([last_row[feature_cols]], columns=feature_cols)

        pred = model.predict(X_future)[0]

        preds.append(pred)
        history.append(pred)

    return np.array(preds)


# ---------------------------
# Select Products
# ---------------------------

selected_products = data["Product ID"].unique()[:3]

pairs = data[
    data["Product ID"].isin(selected_products)
][["Store ID","Product ID"]].drop_duplicates()


# ---------------------------
# Train Model Per Store/Product
# ---------------------------

for _, row in pairs.iterrows():

    store = row["Store ID"]
    product = row["Product ID"]

    df = data[
        (data["Store ID"] == store) &
        (data["Product ID"] == product)
    ].copy()

    df = create_features(df)

    feature_cols = [
        "Lag_1","Lag_2","Lag_3",
        "Lag_7","Lag_14","Lag_21","Lag_28",
        "Rolling_Mean_7",
        "Rolling_Mean_14",
        "Rolling_Std_7",
        "Momentum_7",
        "Momentum_14",
        "Price",
        "Discount",
        "Competitor Pricing",
        "Price_Diff",
        "Holiday/Promotion",
        "DayOfWeek",
        "Month",
        "IsWeekend"
    ]

    encoded_cols = [
        c for c in df.columns
        if "Weather Condition_" in c
        or "Seasonality_" in c
    ]

    feature_cols.extend(encoded_cols)

    X = df[feature_cols]
    y = df["Units Sold"]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False
    )


    # ---------------------------
    # XGBoost Model
    # ---------------------------

    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.01,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42
    )


    model.fit(X_train, y_train)

    preds = model.predict(X_test)


    # ---------------------------
    # Actual vs Prediction Plot
    # ---------------------------

    plt.figure(figsize=(12,4))

    plt.plot(y_test.values, label="Actual", alpha=0.8)
    plt.plot(preds, label="Prediction", linewidth=2)

    plt.title(f"Boosted Demand Forecast – {store} / {product}")
    plt.xlabel("Time")
    plt.ylabel("Units Sold")

    plt.legend()

    file1 = f"{OUTPUT_DIR}/{product}_boosted_prediction.png"

    plt.savefig(file1, dpi=300, bbox_inches="tight")
    plt.close()


    # ---------------------------
    # Future Forecast
    # ---------------------------

    future_preds = recursive_forecast(
        model,
        df,
        feature_cols,
        steps=120
    )


    plt.figure(figsize=(12,4))

    plt.plot(future_preds, linewidth=2, label="Future Demand")

    std = np.std(future_preds)

    upper = future_preds + std
    lower = future_preds - std

    plt.fill_between(
        range(len(future_preds)),
        lower,
        upper,
        alpha=0.25,
        label="Forecast uncertainty"
    )

    plt.title(f"Future Demand Forecast – {product}")
    plt.xlabel("Future Time Steps")
    plt.ylabel("Units Sold")

    plt.legend()

    file2 = f"{OUTPUT_DIR}/{product}_boosted_future.png"

    plt.savefig(file2, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved →", file1)
    print("Saved →", file2)