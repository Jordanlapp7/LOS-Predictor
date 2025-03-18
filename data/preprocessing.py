import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    """Handles missing values, encodes categorical variables, and scales numerical features."""

    # Fill missing values
    df["ethnicity"].fillna("Unknown", inplace=True)
    df["primary_diagnosis"].fillna("No Diagnosis", inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])
    df["admission_type"] = le.fit_transform(df["admission_type"])
    df["insurance"] = le.fit_transform(df["insurance"])

    # Scale numerical features
    scaler = StandardScaler()
    df[["age", "length_of_stay"]] = scaler.fit_transform(df[["age", "length_of_stay"]])

    return df

if __name__ == "__main__":
    from data_extraction import load_data
    df = load_data()
    df_cleaned = preprocess_data(df)
    print(df_cleaned.head())