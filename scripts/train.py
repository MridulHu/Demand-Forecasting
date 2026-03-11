import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_data
from src.preprocessing import encode_categorical
from src.feature_engineering import create_features
from src.models import train_xgboost
from src.forecasting import forecast_future
from src.inventory import calculate_inventory_metrics

from src.visualization import (
    plot_demand_trend,
    plot_monthly_seasonality,
    plot_distribution,
    plot_correlation_heatmap,
    plot_rolling_average,
    plot_forecast,
    plot_feature_importance
)

from src.save_results import save_predictions

from sklearn.model_selection import train_test_split
import pandas as pd


print("Loading dataset...")
data = load_data()

# -----------------------
# EDA Visualizations
# -----------------------
print("Generating EDA plots...")

plot_demand_trend(data)
plot_monthly_seasonality(data)
plot_distribution(data)
plot_correlation_heatmap(data)
plot_rolling_average(data)

# -----------------------
# Feature Engineering
# -----------------------
print("Preprocessing data...")

data = encode_categorical(data)
data = create_features(data)

X = data.drop(columns=["Units Sold"])
y = data["Units Sold"]

# -----------------------
# Train/Test Split
# -----------------------
print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

X_train = X_train.astype(float)
X_test = X_test.astype(float)

# -----------------------
# Train Model
# -----------------------
print("Training XGBoost model...")

model = train_xgboost(X_train, y_train)

# -----------------------
# Forecast Future Demand
# -----------------------
print("Generating future demand forecast...")

future_preds = forecast_future(model, X_test.iloc[-1:].copy())

future_dates = pd.date_range(
    start=X_test.index[-1],
    periods=60,
    freq="D"
)

# -----------------------
# Visualization
# -----------------------
print("Saving visualization plots...")

plot_forecast(X_test.index, y_test, future_dates, future_preds)
plot_feature_importance(model)

# -----------------------
# Save Predictions
# -----------------------
print("Saving predictions...")

save_predictions(future_dates, future_preds)

# -----------------------
# Inventory Metrics
# -----------------------
print("Calculating inventory metrics...")

safety_stock, reorder_point = calculate_inventory_metrics(future_preds)

print("\nResults")
print("Safety Stock:", safety_stock)
print("Reorder Point:", reorder_point)