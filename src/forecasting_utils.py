import pandas as pd
import numpy as np


def future_forecast(model, history, steps=120):

    preds = []
    hist = list(history)

    for step in range(steps):

        lag1 = hist[-1]
        lag7 = hist[-7] if len(hist) >= 7 else hist[-1]
        lag14 = hist[-14] if len(hist) >= 14 else hist[-7]
        lag30 = hist[-30] if len(hist) >= 30 else np.mean(hist[-7:])

        rolling7 = np.mean(hist[-7:])
        rolling14 = np.mean(hist[-14:])
        rolling30 = np.mean(hist[-30:])

        # match the SAME features used in training
        day_of_week = step % 7
        month = (step % 12) + 1

        X = pd.DataFrame([[
            lag1,
            lag7,
            lag14,
            lag30,
            rolling7,
            rolling14,
            rolling30,
            day_of_week,
            month
        ]], columns=[
            "Lag_1",
            "Lag_7",
            "Lag_14",
            "Lag_30",
            "Rolling_Mean_7",
            "Rolling_Mean_14",
            "Rolling_Mean_30",
            "DayOfWeek",
            "Month"
        ])

        pred = model.predict(X)[0]

        preds.append(pred)
        hist.append(pred)

    return preds