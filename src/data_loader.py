import pandas as pd
from .config import DATA_PATH

def load_data():
    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)
    return data