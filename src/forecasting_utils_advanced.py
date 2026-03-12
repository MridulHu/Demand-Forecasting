import pandas as pd


import pandas as pd

def future_forecast_advanced(model, last_row, steps=120):

    preds = []
    current = last_row.copy()

    for _ in range(steps):

        pred = model.predict(current)[0]
        preds.append(pred)

        if "Lag_14" in current.columns:
            current["Lag_14"] = current["Lag_7"]

        if "Lag_7" in current.columns:
            current["Lag_7"] = current["Lag_1"]

        if "Lag_1" in current.columns:
            current["Lag_1"] = pred

        if "Rolling_Mean_7" in current.columns:
            current["Rolling_Mean_7"] = (
                current["Rolling_Mean_7"] * 6 + pred
            ) / 7

    return preds