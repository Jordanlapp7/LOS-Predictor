from data.data_extraction import load_data
from data.preprocessing import preprocess_data
import pandas as pd

def get_clean_data(from_cache=False, classify=False):
    if from_cache:
        X = pd.read_csv("clean_output.csv")
        y = pd.read_csv("target.csv").squeeze()
    else:
        df = load_data()
        X, y = preprocess_data(df, classify=classify)

    return X, y
    
if __name__ == "__main__":
    X, y = get_clean_data(True)
    print(X.head(), y.head())