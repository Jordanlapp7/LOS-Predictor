import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import numpy as np

def train_xgboost(X, y, classify=False):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Classification-specific setup
    if classify:
        classes, y_train = np.unique(y_train, return_inverse=True)
        y_test = np.searchsorted(classes, y_test)  # Match test labels to encoded values
        objective = 'multi:softmax'
        eval_metric = 'merror'
        num_class = len(classes)
    else:
        objective = 'reg:squarederror'
        eval_metric = 'rmse'
        num_class = None

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define model parameters
    params = {
        'objective': objective,
        'eval_metric': eval_metric,
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    if classify:
        params['num_class'] = num_class

    # Train
    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=1)

    # Predict
    preds = model.predict(dtest)

    # Evaluate
    print("\nXGBoost Results:")
    if classify:
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.3f}")
    else:
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        print(f"MAE: {mae:.3f}")
        print(f"MSE: {mse:.3f}")
        print(f"RMSE: {rmse:.3f}")

    return model

if __name__ == "__main__":
    from data.load_pipeline import get_clean_data
    
    from_cache = True
    classify = True

    X, y = get_clean_data(from_cache, classify)
    train_xgboost(X, y, classify)