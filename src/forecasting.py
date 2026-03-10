import pandas as pd

def forecast_future(model, last_row, steps=60):

    future_preds = []
    future_X = last_row.copy()

    for _ in range(steps):

        pred = model.predict(future_X)[0]
        future_preds.append(pred)

        future_X.iloc[0, :-1] = future_X.iloc[0, 1:].values
        future_X.iloc[0, -1] = pred

    return future_preds