import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_xgboost(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DMatrix (optional but preferred)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define model parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'eta': 0.1,
        'seed': 42
    }

    # Train the model
    model = xgb.train(params, dtrain, num_boost_round=1000, verbose_eval=1)

    # Predict
    preds = model.predict(dtest)

    # Evaluate
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    print("XGBoost Results:")
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")

    return model

if __name__ == "__main__":
    # from data_extraction import load_data
    # from preprocessing import preprocess_data
    import pandas as pd
    
    X = pd.read_csv('clean_output.csv')
    y = pd.read_csv('target.csv')
    train_xgboost(X, y)