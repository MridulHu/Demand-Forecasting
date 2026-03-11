import pandas as pd
import numpy as np


def forecast_future(model, last_row, steps=60):

    future_preds = []

    future_X = last_row.copy()

    # ensure numeric types
    future_X = future_X.apply(pd.to_numeric, errors="coerce").fillna(0)

    for _ in range(steps):

        # convert again inside loop (important)
        future_X = future_X.astype(float)

        pred = model.predict(future_X)[0]
        future_preds.append(pred)

        # shift lag values
        future_X.iloc[0, :-1] = future_X.iloc[0, 1:].values
        future_X.iloc[0, -1] = pred

    return future_preds