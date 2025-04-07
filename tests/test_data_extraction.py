import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from data import data_extraction

class TestDataExtraction(unittest.TestCase):
    @patch("data.data_extraction.client")  # Patches bigquery.Client() instance
    def test_load_data_returns_dataframe(self, mock_client):
        # Create dummy data
        dummy_df = pd.DataFrame({
            "subject_id": [1],
            "hadm_id": [101],
            "gender": ["M"],
            "age": [70],
            "race": ["WHITE"],
            "admission_type": ["EMERGENCY"],
            "insurance": ["Medicare"],
            "admittime": ["2022-01-01 00:00:00"],
            "dischtime": ["2022-01-05 00:00:00"],
            "length_of_stay": [4],
            "primary_diagnosis": ["I10"]
        })

        # Set up the mock behavior
        mock_query_job = MagicMock()
        mock_query_job.to_dataframe.return_value = dummy_df
        mock_client.query.return_value = mock_query_job

        # Call function
        result = data_extraction.load_data()

        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, dummy_df.shape)
        self.assertTrue("subject_id" in result.columns)

if __name__ == "__main__":
    unittest.main()