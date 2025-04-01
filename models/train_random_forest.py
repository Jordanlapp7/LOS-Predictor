import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models.random_forest import RandomForest

def train_random_forest(X, y):
    """Trains Random Forest model."""
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

    # Initialize your custom random forest
    model = RandomForest(n_trees=10, max_depth=10, min_samples_split=5)

    # Train the model
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5

    print("Random Forest Results:")
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")

if __name__ == "__main__":
    from data.load_pipeline import get_clean_data
    
    X, y = get_clean_data(True)
    train_random_forest(X, y)
