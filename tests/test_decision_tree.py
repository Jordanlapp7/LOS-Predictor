import unittest
import numpy as np
from models.decision_tree import DecisionTree

class TestDecisionTree(unittest.TestCase):

    def setUp(self):
        # Simple dataset (same for classification & regression)
        self.X = np.array([
            [1, 5],
            [2, 4],
            [3, 3],
            [4, 2],
            [5, 1],
            [6, 0]
        ])
        self.y_regression = np.array([1.5, 1.7, 3.0, 3.2, 5.0, 5.2])
        self.y_classification = np.array(["short", "short", "medium", "medium", "long", "long"])

    def test_regression_prediction(self):
        tree = DecisionTree(max_depth=2, classify=False)
        tree.fit(self.X, self.y_regression)
        preds = tree.predict(self.X)
        self.assertEqual(len(preds), len(self.y_regression))
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(np.all((preds >= min(self.y_regression)) & (preds <= max(self.y_regression))))

    def test_classification_prediction(self):
        tree = DecisionTree(max_depth=2, classify=True)
        tree.fit(self.X, self.y_classification)
        preds = tree.predict(self.X)
        self.assertEqual(len(preds), len(self.y_classification))
        self.assertTrue(set(preds).issubset(set(self.y_classification)))

    def test_entropy_output(self):
        tree = DecisionTree(classify=True)
        y = np.array(["short", "short", "medium", "medium", "long", "long"])
        entropy = tree._entropy(y)
        self.assertGreater(entropy, 0)
        self.assertLessEqual(entropy, np.log(3))

    def test_information_gain_split(self):
        tree = DecisionTree(classify=True)
        X_column = np.array([1, 2, 3, 4, 5, 6])
        threshold = 3
        y = np.array(["a", "a", "a", "b", "b", "b"])
        gain = tree._information_gain(y, X_column, threshold)
        self.assertGreaterEqual(gain, 0)

    def test_leaf_node_behavior(self):
        tree = DecisionTree(max_depth=0, classify=True)
        tree.fit(self.X, self.y_classification)
        preds = tree.predict(self.X)
        self.assertTrue(np.all(preds == preds[0]))  # All predictions same at depth 0

    def test_handling_small_datasets(self):
        X_small = np.array([[1, 2]])
        y_small = np.array(["short"])
        tree = DecisionTree(classify=True)
        tree.fit(X_small, y_small)
        preds = tree.predict(X_small)
        self.assertEqual(preds[0], "short")

if __name__ == '__main__':
    unittest.main()