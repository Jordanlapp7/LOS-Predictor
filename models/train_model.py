import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from random_forest import RandomForest

def train_model(X, y):
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

    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)

if __name__ == "__main__":
    # from data_extraction import load_data
    # from preprocessing import preprocess_data
    
    X = pd.read_csv('clean_output.csv')
    y = pd.read_csv('target.csv')
    train_model(X, y)
