import numpy as np
import pandas as pd


def forecast_future(model, last_row, steps=60):

    future_preds = []

    future_X = last_row.copy().astype(float)

    demand_history = [
        future_X["Lag_30"].values[0],
        future_X["Lag_7"].values[0],
        future_X["Lag_1"].values[0],
    ]

    for _ in range(steps):

        pred = model.predict(future_X)[0]
        future_preds.append(pred)

        demand_history.append(pred)

        # update lag features
        future_X["Lag_1"] = demand_history[-1]
        future_X["Lag_7"] = demand_history[-7] if len(demand_history) >= 7 else demand_history[-1]
        future_X["Lag_30"] = demand_history[-30] if len(demand_history) >= 30 else demand_history[-1]

        # update rolling stats
        future_X["Rolling_Mean_7"] = np.mean(demand_history[-7:])
        future_X["Rolling_Std_7"] = np.std(demand_history[-7:])

        future_X["Rolling_Mean_30"] = np.mean(demand_history[-30:])
        future_X["Rolling_Std_30"] = np.std(demand_history[-30:])

    return future_preds