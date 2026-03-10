from src.data_loader import load_data
from src.preprocessing import encode_categorical
from src.feature_engineering import create_features
from src.models import train_xgboost
from src.forecasting import forecast_future
from src.inventory import calculate_inventory_metrics

from sklearn.model_selection import train_test_split

data = load_data()

data = encode_categorical(data)
data = create_features(data)

X = data.drop(columns=["Units Sold"])
y = data["Units Sold"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = train_xgboost(X_train, y_train)

future_preds = forecast_future(model, X_test.iloc[-1:].copy())

safety_stock, reorder_point = calculate_inventory_metrics(future_preds)

print("Safety Stock:", safety_stock)
print("Reorder Point:", reorder_point)