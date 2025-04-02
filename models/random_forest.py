from models.decision_tree import DecisionTree
import numpy as np
from collections import Counter
from tqdm import tqdm

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, classify=False, n_feature=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.classify=classify
        self.n_features=n_feature
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in tqdm(range(self.n_trees), desc="Training Random Forest", unit="tree"):
            tree = DecisionTree(max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            classify=self.classify,
                            n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        if self.classify:
            tree_preds = np.swapaxes(predictions, 0, 1)
            predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
            return predictions
        else:
            mean_preds = np.mean(predictions, axis=0)
            return mean_preds

if __name__ == "__main__":
    # Test random forest using small dataset
    import pandas as pd
    from sklearn.metrics import mean_squared_error, accuracy_score

    classify = False

    # Features
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [5, 4, 3, 2, 1, 0]
    })

    if classify:
        y = pd.Series(["short", "short", "medium", "medium", "long", "long"])
    else:
        y = pd.Series([1.5, 1.7, 3.0, 3.2, 5.0, 5.2])

    forest = RandomForest(n_trees=10, max_depth=2, classify=classify)
    forest.fit(X.values, y.values)
    pred = forest.predict(X.values)

    print("Forest prediction:", pred)
    print("True values:", y.values)

    if classify:
        acc = accuracy_score(y.values, pred)
        print("Accuracy:", acc)
    else:
        mse = mean_squared_error(y.values, pred)
        print("Mean squared error:", mse)
        print("Root mean squared error:", np.sqrt(mse))