import matplotlib.pyplot as plt

def plot_forecast(dates, actual, future_dates, preds):

    plt.figure(figsize=(14,6))

    plt.plot(dates, actual, label="Actual")
    plt.plot(future_dates, preds, "--", label="Forecast")

    plt.legend()
    plt.title("Demand Forecast")

    plt.show()