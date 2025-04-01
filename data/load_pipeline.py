from data.data_extraction import load_data
from data.preprocessing import preprocess_data

def get_clean_data():
    df = load_data()
    X, y = preprocess_data(df)
    return X, y