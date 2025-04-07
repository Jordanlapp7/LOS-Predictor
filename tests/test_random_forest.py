import unittest
import numpy as np
import pandas as pd
from models.random_forest import RandomForest

class TestRandomForest(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [6, 5, 4, 3, 2, 1]
        }).values

        self.y_reg = np.array([1.5, 1.7, 3.0, 3.2, 5.0, 5.2])
        self.y_clf = np.array(["short", "short", "medium", "medium", "long", "long"])

    def test_regression_prediction_shape(self):
        rf = RandomForest(n_trees=5, max_depth=2, classify=False)
        rf.fit(self.X, self.y_reg)
        preds = rf.predict(self.X)
        self.assertEqual(preds.shape, self.y_reg.shape)
        self.assertTrue(np.issubdtype(preds.dtype, np.number))

    def test_classification_prediction_shape(self):
        rf = RandomForest(n_trees=5, max_depth=2, classify=True)
        rf.fit(self.X, self.y_clf)
        preds = rf.predict(self.X)
        self.assertEqual(preds.shape, self.y_clf.shape)
        self.assertTrue(preds.dtype.kind in {'U', 'O'})  # string-like

    def test_bootstrap_sample_size(self):
        rf = RandomForest()
        X_sample, y_sample = rf._bootstrap_samples(self.X, self.y_reg)
        self.assertEqual(X_sample.shape, self.X.shape)
        self.assertEqual(len(y_sample), len(self.y_reg))

    def test_most_common_label(self):
        rf = RandomForest(classify=True)
        labels = np.array(['a', 'b', 'a', 'c', 'a'])
        self.assertEqual(rf._most_common_label(labels), 'a')

    def test_output_type_regression(self):
        rf = RandomForest(n_trees=1, classify=False)
        rf.fit(self.X, self.y_reg)
        preds = rf.predict(self.X)
        self.assertTrue(np.issubdtype(preds.dtype, np.floating))

    def test_output_type_classification(self):
        rf = RandomForest(n_trees=1, classify=True)
        rf.fit(self.X, self.y_clf)
        preds = rf.predict(self.X)
        self.assertTrue(all(p in {"short", "medium", "long"} for p in preds))

if __name__ == "__main__":
    unittest.main()
