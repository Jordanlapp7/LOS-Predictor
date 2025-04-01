from data.data_extraction import load_data
from data.preprocessing import preprocess_data
import pandas as pd

def get_clean_data(from_cache=False):
    if from_cache:
        X = pd.read_csv("clean_output.csv")
        y = pd.read_csv("target.csv").squeeze()
        return X, y
    else:
        df = load_data()
        X, y = preprocess_data(df)
        return X, y
    
if __name__ == "__main__":
    X, y = get_clean_data(True)
    print(X.head(), y.head())