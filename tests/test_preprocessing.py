import unittest
import pandas as pd
import os
import tempfile
from data.preprocessing import preprocess_data

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "gender": ["M", "F", "M", "F", "M"],
            "race": ["WHITE", "BLACK", "ASIAN", "HISPANIC", "UNKNOWN"],
            "admission_type": ["ELECTIVE", "URGENT", "ELECTIVE", "URGENT", "ELECTIVE"],
            "insurance": ["Medicare", "Private", "Medicaid", "Medicare", "Other"],
            "admittime": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05"]),
            "dischtime": pd.to_datetime(["2022-01-03", "2022-01-05", "2022-01-10", "2022-01-06", "2022-01-07"]),
            "length_of_stay": [2, 3, 7, 2, 2],
            "primary_diagnosis": ["A41", "I10", "J18", "K35", "E11"],
            "subject_id": [1, 2, 3, 4, 5],
            "hadm_id": [11, 12, 13, 14, 15]
        })

    def test_preprocess_data_output_shapes(self):
        X, y = preprocess_data(self.df, classify=False)
        self.assertEqual(len(X), len(y))
        self.assertFalse(X.isnull().any().any(), "X contains null values after preprocessing.")

    def test_preprocess_data_with_classification(self):
        X, y = preprocess_data(self.df, classify=True)
        self.assertEqual(len(X), len(y))
        self.assertTrue(y.dtype.name == "category" or y.dtype == object,"Classification labels should be categorical or string type.")

    def test_preprocess_data_with_temp_save(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_x, \
             tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_y:
            tmp_x.close()
            tmp_y.close()

            try:
                X, y = preprocess_data(self.df, classify=True, save=True,
                                       output_X_path=tmp_x.name,
                                       output_y_path=tmp_y.name)

                saved_X = pd.read_csv(tmp_x.name)
                saved_y = pd.read_csv(tmp_y.name)

                self.assertFalse(saved_X.empty)
                self.assertFalse(saved_y.empty)
            finally:
                os.remove(tmp_x.name)
                os.remove(tmp_y.name)

if __name__ == "__main__":
    unittest.main()