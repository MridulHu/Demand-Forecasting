import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_data
from src.feature_engineering import create_features
from src.models import train_xgboost
from src.forecasting import forecast_future
from src.inventory import calculate_inventory_metrics
from src.visualization import (
    plot_demand_trend,
    plot_monthly_seasonality,
    plot_distribution,
    plot_rolling_average,
    plot_forecast
)

from src.save_results import save_predictions

from sklearn.model_selection import train_test_split
import pandas as pd


print("Loading dataset...")
data = load_data()

print("Generating EDA plots...")
plot_demand_trend(data)
plot_monthly_seasonality(data)
plot_distribution(data)
plot_rolling_average(data)

print("Creating features...")
data = create_features(data)

X = data.drop(columns=["Daily Demand"])
y = data["Daily Demand"]

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

print("Training XGBoost model...")
model = train_xgboost(X_train, y_train)

print("Generating forecast...")
future_preds = forecast_future(model, X_test.iloc[-1:].copy())

future_dates = pd.date_range(
    start=X_test.index[-1],
    periods=60,
    freq="D"
)

plot_forecast(X_test.index, y_test, future_dates, future_preds)

print("Saving predictions...")
save_predictions(future_dates, future_preds)

print("Calculating inventory metrics...")
safety_stock, reorder_point = calculate_inventory_metrics(future_preds)

print("\nResults")
print("Safety Stock:", safety_stock)
print("Reorder Point:", reorder_point)