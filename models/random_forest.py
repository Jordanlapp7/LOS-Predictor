from decision_tree import DecisionTree
import numpy as np

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_feature
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        mean_preds = np.mean(predictions, axis=0)
        return mean_preds

if __name__ == "__main__":
    # Test random forest using small dataset
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    # Features (X)
    X = pd.DataFrame({
      'feature1': [1, 2, 3, 4, 5, 6],
      'feature2': [5, 4, 3, 2, 1, 0]
    })

    # Target (y) - Continuous
    y = pd.Series([1.5, 1.7, 3.0, 3.2, 5.0, 5.2])

    # Train on sample training data
    forest = RandomForest(n_trees=100, max_depth=2)
    forest.fit(X.values, y.values)

    # Predict on sample training data
    pred = forest.predict(X.values)
    mse = mean_squared_error(y.values, pred)

    print("Forest prediction:", pred)
    print("True values:", y.values)
    print("Forest Mean squared error:", mse)
    print("Forest Root mean squared error:", np.sqrt(mse))