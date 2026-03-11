import pandas as pd
import os

PRED_DIR = "outputs/predictions"
os.makedirs(PRED_DIR, exist_ok=True)


def save_predictions(dates, preds):

    df = pd.DataFrame({
        "Date": dates,
        "Predicted_Demand": preds
    })

    df.to_csv(f"{PRED_DIR}/future_predictions.csv", index=False)