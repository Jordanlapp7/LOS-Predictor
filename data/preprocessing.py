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

def simplify_race(value):
    val = value.upper()
    
    if "WHITE" in val or "PORTUGUESE" in val:
        return "white"
    elif "BLACK" in val:
        return "black"
    elif "HISPANIC" in val or "SOUTH AMERICAN" in val or "CENTRAL AMERICAN" in val:
        return "hispanic"
    elif "ASIAN" in val:
        return "asian"
    elif "OTHER" in val or "MULTIPLE" in val:
        return "other"
    elif "UNKNOWN" in val or "UNABLE" in val or "DECLINED" in val:
        return "unknown"
    else:
        return "other"

def one_hot_encode_categoricals(X, categorical_cols):
    X = X.copy()
    
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    return X_encoded

def preprocess_data(df):
    """Extracts target, handles missing values, encodes categorical variables, and scales numerical features."""


    # Extract target
    y = df['length_of_stay']
    X = df.drop(columns=['length_of_stay', 'admittime', 'dischtime', 'subject_id', 'hadm_id'])

    # Extract features
    X['primary_diagnosis'] = X['primary_diagnosis'].str[:3]


    encoded = target_encode_oof_with_fallback(X, y, 'primary_diagnosis', min_count=10)
    X['encoded_diagnosis'] = pd.Series(encoded, index=X.index)
    X = X.drop(columns=['primary_diagnosis'])

    # Re-categorize race column into broader buckets
    X['race'] = X['race'].apply(simplify_race)


    # One-hot encode the remaining categorical features
    categorical_cols = ['gender', 'race', 'admission_type', 'insurance']
    X = one_hot_encode_categoricals(X, categorical_cols)

    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)

    return X, y

if __name__ == "__main__":
    from data_extraction import load_data
    df = load_data()
    X, y = preprocess_data(df)
    X.to_csv('clean_output.csv', index=False)
    y.to_csv('target.csv', index=False)