from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_random_forest(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):

    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        random_state=42
    )

    model.fit(X_train, y_train)
    return model