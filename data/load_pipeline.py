from data.data_extraction import load_data
from data.preprocessing import preprocess_data
import pandas as pd
import numpy as np

def get_clean_data(from_cache=False, classify=False):
    if from_cache:
        try:
            X = pd.read_csv("clean_output.csv")
            y = pd.read_csv("target.csv").squeeze()
        except FileNotFoundError:
            raise FileNotFoundError(
                "Cached files not found. Make sure 'clean_output.csv' and 'target.csv' exist "
                "or set `from_cache=False` to generate them from scratch."
            )

        if classify:
            y = pd.cut(y, bins=[-1, 3, 7, float("inf")], labels=["short", "medium", "long"])
            _, y = np.unique(y, return_inverse=True)

    else:
        df = load_data()
        X, y = preprocess_data(df, classify=classify)
        if classify:
            _, y = np.unique(y, return_inverse=True)

    return X, y
    
if __name__ == "__main__":
    X, y = get_clean_data(True)
    print(X.head(), y.head())