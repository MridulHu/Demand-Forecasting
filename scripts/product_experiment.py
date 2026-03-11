import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Output directory
OUTPUT_DIR = "outputs/plots/product_experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load dataset
data = pd.read_csv("data/retail_store_inventory.csv")

data["Date"] = pd.to_datetime(data["Date"])


# Take first 10 products
products = data["Product ID"].unique()[:10]


def create_features(df):

    df = df.sort_values("Date")

    df["Lag_1"] = df["Units Sold"].shift(1)
    df["Lag_7"] = df["Units Sold"].shift(7)

    df["Rolling_Mean_7"] = df["Units Sold"].rolling(7).mean()

    df = df.dropna()

    return df


for i in range(len(products) - 1):

    product_A = products[i]
    product_B = products[i + 1]

    df_A = data[data["Product ID"] == product_A]
    df_B = data[data["Product ID"] == product_B]

    df_A = create_features(df_A)
    df_B = create_features(df_B)

    X_A = df_A[["Lag_1", "Lag_7", "Rolling_Mean_7"]]
    y_A = df_A["Units Sold"]

    X_B = df_B[["Lag_1", "Lag_7", "Rolling_Mean_7"]]
    y_B = df_B["Units Sold"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_A, y_A, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    pred_A = model.predict(X_test)

    # Use model from product A on product B
    pred_B = model.predict(X_B.iloc[-len(pred_A):])


    # ---- Smooth results for visualization ----
    y_test_s = pd.Series(y_test.values).rolling(7).mean()
    pred_A_s = pd.Series(pred_A).rolling(7).mean()
    pred_B_s = pd.Series(pred_B).rolling(7).mean()


    # -------- Plot 1: Product A forecast --------
    plt.figure(figsize=(10,4))

    plt.plot(y_test_s.values,
             label=f"Actual {product_A}",
             alpha=0.6)

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



    # -------- Plot 2: Product B with wrong model --------
    plt.figure(figsize=(10,4))

    plt.plot(y_B.iloc[-len(pred_B_s):].values,
             label=f"Actual {product_B}",
             alpha=0.6)

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


    print(f"Saved → {filename_A}")
    print(f"Saved → {filename_B}")