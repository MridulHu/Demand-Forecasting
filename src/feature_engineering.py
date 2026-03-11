def create_features(data):

    df = data.copy()

    df["Lag_1"] = df["Daily Demand"].shift(1)
    df["Lag_7"] = df["Daily Demand"].shift(7)
    df["Lag_30"] = df["Daily Demand"].shift(30)

    df["Rolling_Mean_7"] = df["Daily Demand"].rolling(7).mean()
    df["Rolling_Std_7"] = df["Daily Demand"].rolling(7).std()

    df["Rolling_Mean_30"] = df["Daily Demand"].rolling(30).mean()
    df["Rolling_Std_30"] = df["Daily Demand"].rolling(30).std()

    df["Month"] = df.index.month
    df["Day"] = df.index.day
    df["Week"] = df.index.isocalendar().week.astype(int)
    df["Year"] = df.index.year

    df = df.dropna()

    return df