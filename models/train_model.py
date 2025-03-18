import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_model(df):
    """Trains Random Forest model with cross-validation."""
    
    # Define features and target
    X = df.drop(columns=["length_of_stay", "subject_id", "hadm_id"])
    y = df["length_of_stay"]

    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # 5-Fold Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    print("Cross-Validation MAE Scores:", -cv_scores)
    print("Mean Cross-Validation MAE:", -cv_scores.mean())

    # Train the final model
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate model performance
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print("Training MAE:", train_mae)
    print("Testing MAE:", test_mae)

    # Save model
    joblib.dump(model, "models/los_predictor.pkl")

if __name__ == "__main__":
    from preprocessing import preprocess_data
    from data_extraction import load_data
    
    df = load_data()
    df_cleaned = preprocess_data(df)
    train_model(df_cleaned)
