import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score
)
from models.random_forest import RandomForest

def train_random_forest(X, y, classify=True):
    """Trains Random Forest model (classification or regression)."""
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

    # Initialize your custom random forest
    model = RandomForest(
        n_trees=10,
        max_depth=10,
        min_samples_split=5,
        classify=classify
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Evaluate
    print("\nRandom Forest Results:")
    if classify:
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.3f}")
    else:
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        print(f"MAE: {mae:.3f}")
        print(f"MSE: {mse:.3f}")
        print(f"RMSE: {rmse:.3f}")

if __name__ == "__main__":
    from data.load_pipeline import get_clean_data
    
    from_cache = True
    classify = True
    X, y = get_clean_data(from_cache, classify)
    train_random_forest(X, y, classify=classify)
