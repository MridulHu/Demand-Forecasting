import pandas as pd

def encode_categorical(data):
    categorical_cols = data.select_dtypes(include=["object"]).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    return data