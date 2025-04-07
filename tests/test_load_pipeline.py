import unittest
import numpy as np
import pandas as pd
from data.load_pipeline import get_clean_data

class TestLoadPipeline(unittest.TestCase):

    def test_from_cache_regression(self):
        X, y = get_clean_data(from_cache=True, classify=False)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertTrue(isinstance(y, pd.Series) or isinstance(y, np.ndarray))
        self.assertEqual(len(X), len(y))
        self.assertTrue(np.issubdtype(y.dtype, np.number))

    def test_from_cache_classification(self):
        X, y = get_clean_data(from_cache=True, classify=True)
        self.assertEqual(len(X), len(y))
        self.assertTrue(np.issubdtype(y.dtype, np.integer))
        self.assertTrue(set(np.unique(y)).issubset({0, 1, 2}))

if __name__ == '__main__':
    unittest.main()