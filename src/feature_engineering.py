def create_features(data):

    data["Month"] = data.index.month
    data["Day"] = data.index.day
    data["Week"] = data.index.isocalendar().week
    data["Year"] = data.index.year

    data["Lag_1"] = data["Units Sold"].shift(1)
    data["Lag_7"] = data["Units Sold"].shift(7)
    data["Lag_30"] = data["Units Sold"].shift(30)

    data["Rolling_Mean_7"] = data["Units Sold"].rolling(7).mean()
    data["Rolling_Std_7"] = data["Units Sold"].rolling(7).std()

    return data.dropna()