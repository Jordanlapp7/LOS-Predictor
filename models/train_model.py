import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_model(X, y):
    """Trains Random Forest model with cross-validation."""
    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5-Fold Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    print("Cross-Validation MAE Scores:", -cv_scores)
    print("Mean Cross-Validation MAE:", -cv_scores.mean())

    # Train the final model

    # Make predictions

    # Evaluate model performance

    # Save model

if __name__ == "__main__":
    from preprocessing import preprocess_data
    from data_extraction import load_data
    
    df = load_data()
    df_cleaned = preprocess_data(df)
    train_model(df_cleaned)
