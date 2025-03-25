import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def target_encode_oof_with_fallback(X, y, column, min_count=10, n_splits=5):
    X = X.copy()
    global_mean = y.mean()
    encoded_column = np.zeros(len(X))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        # Get counts and means from training fold
        train_group = X_train[column]
        value_counts = train_group.value_counts()
        means = y_train.groupby(train_group).mean()

        # Only keep means for categories that meet count threshold
        safe_means = means[value_counts >= min_count]

        # Map values in validation fold
        val_mapping = X_val[column].map(safe_means).fillna(global_mean)

        # Assign to output
        encoded_column[val_idx] = val_mapping

    return encoded_column

def preprocess_data(df):
    """Extracts target, handles missing values, encodes categorical variables, and scales numerical features."""


    # Extract target
    y = df['length_of_stay']
    X = df.drop(columns=['length_of_stay', 'admittime', 'dischtime', 'subject_id', 'hadm_id'])

    # Extract features
    X['primary_diagnosis'] = X['primary_diagnosis'].str[:3]


    X['encoded_diagnosis'] = target_encode_oof_with_fallback(X, y, 'primary_diagnosis')
    # Fill missing values
    

    # Encode categorical variables


    return X, y

if __name__ == "__main__":
    from data_extraction import load_data
    df = load_data()
    df_cleaned, y = preprocess_data(df)
    df_cleaned.to_csv('clean_output.csv', index=False)
    y.to_csv('target.csv', index=False)