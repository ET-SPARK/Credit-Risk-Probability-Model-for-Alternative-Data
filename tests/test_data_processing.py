
import unittest
import pandas as pd
import numpy as np
from src.data_processing import create_features, process_data, calculate_rfm

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        """Set up for the tests."""
        data = {
            'customer_id': [1, 1, 2, 2, 3, 3, 4],
            'transaction_id': [101, 102, 103, 104, 105, 106, 107],
            'transaction_amount': [100, 200, 50, 75, 150, 250, 300],
            'transaction_date': pd.to_datetime([
                '2023-01-15 10:30:00', '2023-01-16 12:00:00',
                '2023-01-17 08:00:00', '2023-01-18 14:00:00',
                '2023-01-19 18:45:00', '2023-01-20 09:00:00',
                '2023-01-21 11:00:00'
            ]),
            'age': [35, 35, 45, 45, 28, 28, 50],
            'income': [50000, 50000, 75000, 75000, 60000, 60000, 80000],
            'gender': ['Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female'],
            'education': ['Bachelor', 'Bachelor', 'Master', 'Master', 'PhD', 'PhD', 'Bachelor'],
            'marital_status': ['Married', 'Married', 'Single', 'Single', 'Married', 'Married', 'Single'],
            'target': [0, 0, 1, 1, 0, 0, 1]
        }
        self.df = pd.DataFrame(data)

    def test_create_features(self):
        """Test the feature creation function."""
        df_featured = create_features(self.df)
        self.assertIn('transaction_hour', df_featured.columns)
        self.assertIn('total_transaction_amount', df_featured.columns)
        self.assertEqual(df_featured[df_featured['customer_id'] == 1]['total_transaction_amount'].iloc[0], 300) # 100 + 200 + 150 = 450

    def test_calculate_rfm(self):
        """Test the RFM calculation function."""
        # Sample data for RFM calculation
        rfm_data = {
            'customer_id': [1, 1, 2, 3, 3, 3],
            'transaction_id': [1, 2, 3, 4, 5, 6],
            'transaction_date': pd.to_datetime([
                '2023-01-01', '2023-01-05', 
                '2023-01-02', 
                '2023-01-10', '2023-01-12', '2023-01-15'
            ]),
            'transaction_amount': [100, 200, 50, 300, 150, 50]
        }
        rfm_df_test = pd.DataFrame(rfm_data)

        # Calculate RFM
        rfm_result = calculate_rfm(rfm_df_test)

        # Expected results (snapshot_date = max_transaction_date + 1 day = 2023-01-16)
        # Customer 1: Recency = (2023-01-16 - 2023-01-05).days = 11, Frequency = 2, Monetary = 300
        # Customer 2: Recency = (2023-01-16 - 2023-01-02).days = 14, Frequency = 1, Monetary = 50
        # Customer 3: Recency = (2023-01-16 - 2023-01-15).days = 1, Frequency = 3, Monetary = 500

        # Assertions for Customer 1
        customer_1_rfm = rfm_result[rfm_result['customer_id'] == 1].iloc[0]
        self.assertEqual(customer_1_rfm['Recency'], 11)
        self.assertEqual(customer_1_rfm['Frequency'], 2)
        self.assertEqual(customer_1_rfm['Monetary'], 300)

        # Assertions for Customer 2
        customer_2_rfm = rfm_result[rfm_result['customer_id'] == 2].iloc[0]
        self.assertEqual(customer_2_rfm['Recency'], 14)
        self.assertEqual(customer_2_rfm['Frequency'], 1)
        self.assertEqual(customer_2_rfm['Monetary'], 50)

        # Assertions for Customer 3
        customer_3_rfm = rfm_result[rfm_result['customer_id'] == 3].iloc[0]
        self.assertEqual(customer_3_rfm['Recency'], 1)
        self.assertEqual(customer_3_rfm['Frequency'], 3)
        self.assertEqual(customer_3_rfm['Monetary'], 500)

    def test_process_data(self):
        """Test the data processing pipeline."""
        # process_data now returns X_processed, y, and preprocessor
        X_processed, y, preprocessor = process_data(self.df)
        self.assertIsNotNone(X_processed)
        self.assertIsNotNone(y)
        self.assertIsNotNone(preprocessor)
        self.assertEqual(X_processed.shape[0], 5)
        self.assertEqual(y.shape[0], 5)

if __name__ == '__main__':
    unittest.main()
