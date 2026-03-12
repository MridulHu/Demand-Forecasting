import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


OUTPUT_DIR = "outputs/plots/advanced_forecasting"
os.makedirs(OUTPUT_DIR, exist_ok=True)


data = pd.read_csv("data/retail_store_inventory.csv")
data["Date"] = pd.to_datetime(data["Date"])


# Encode categorical variables
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


# choose only a few products
selected_products = data["Product ID"].unique()[:3]

product_store_pairs = data[
    data["Product ID"].isin(selected_products)
][["Store ID", "Product ID"]].drop_duplicates()


def create_features(df):

    df = df.sort_values("Date")

    # Lag features (more memory)
    for lag in [1,2,3,7,14,21,28]:
        df[f"Lag_{lag}"] = df["Units Sold"].shift(lag)

    # Rolling demand statistics
    df["Rolling_Mean_7"] = df["Units Sold"].shift(1).rolling(7).mean()
    df["Rolling_Mean_14"] = df["Units Sold"].shift(1).rolling(14).mean()
    df["Rolling_Std_7"] = df["Units Sold"].shift(1).rolling(7).std()

    # Demand momentum
    df["Momentum_7"] = df["Units Sold"].shift(1) - df["Units Sold"].shift(7)
    df["Momentum_14"] = df["Units Sold"].shift(1) - df["Units Sold"].shift(14)

    # Price sensitivity
    df["Price_Diff"] = df["Price"] - df["Competitor Pricing"]

    # Time features
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month

    df = df.dropna()

    return df


def recursive_forecast(model, df, feature_cols, steps=120):

    history = list(df["Units Sold"].values)
    future_preds = []

    last_row = df.iloc[-1].copy()

    for i in range(steps):

        for lag in [1,2,3,7,14,21,28]:
            last_row[f"Lag_{lag}"] = history[-lag]

        X_future = pd.DataFrame([last_row[feature_cols]], columns=feature_cols)

        pred = model.predict(X_future)[0]

        future_preds.append(pred)
        history.append(pred)

    return np.array(future_preds)


for _, row in product_store_pairs.iterrows():

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
    "Month"
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
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
    n_estimators=800,
    max_depth=25,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # -------- PLOT ACTUAL VS PREDICTION --------

    plt.figure(figsize=(11,4))

    plt.plot(y_test.values, label="Actual", alpha=0.7)
    plt.plot(preds, label="Model Prediction", linewidth=2)

    plt.title(f"Advanced Forecasting – {store} / {product}")
    plt.xlabel("Time")
    plt.ylabel("Units Sold")

    plt.legend()

    filename = f"{OUTPUT_DIR}/{product}_advanced_model.png"

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


    # -------- FUTURE FORECAST --------

    future_preds = recursive_forecast(
        model,
        df,
        feature_cols,
        steps=120
    )

    plt.figure(figsize=(11,4))

    plt.plot(
        future_preds,
        linewidth=2,
        label="Future Forecast"
    )

    std = np.std(future_preds)

    upper = future_preds + std
    lower = future_preds - std

    plt.fill_between(
        range(len(future_preds)),
        lower,
        upper,
        alpha=0.2,
        label="Forecast uncertainty"
    )

    plt.title(f"Future Demand Forecast – {product}")
    plt.xlabel("Future Time Steps")
    plt.ylabel("Units Sold")

    plt.legend()

    filename_future = f"{OUTPUT_DIR}/{product}_future_advanced.png"

    plt.savefig(filename_future, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved →", filename)
    print("Saved →", filename_future)