
import unittest
import pandas as pd
import numpy as np
from src.data_processing import create_features, process_data

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        """Set up for the tests."""
        data = {
            'customer_id': [1, 1, 2, 2, 1],
            'transaction_amount': [100, 200, 50, 75, 150],
            'transaction_date': pd.to_datetime(['2023-01-15 10:30:00', '2023-01-16 12:00:00', '2023-01-17 08:00:00', '2023-01-18 14:00:00', '2023-01-19 18:45:00']),
            'age': [35, 35, 45, 45, 35],
            'income': [50000, 50000, 75000, 75000, 50000],
            'gender': ['Male', 'Male', 'Female', 'Female', 'Male'],
            'education': ['Bachelor', 'Bachelor', 'Master', 'Master', 'Bachelor'],
            'marital_status': ['Married', 'Married', 'Single', 'Single', 'Married']
        }
        self.df = pd.DataFrame(data)

    def test_create_features(self):
        """Test the feature creation function."""
        df_featured = create_features(self.df)
        self.assertIn('transaction_hour', df_featured.columns)
        self.assertIn('total_transaction_amount', df_featured.columns)
        self.assertEqual(df_featured[df_featured['customer_id'] == 1]['total_transaction_amount'].iloc[0], 450)

    def test_process_data(self):
        """Test the data processing pipeline."""
        processed_data, _ = process_data(self.df)
        self.assertIsNotNone(processed_data)
        self.assertEqual(processed_data.shape[0], 5)

if __name__ == '__main__':
    unittest.main()
