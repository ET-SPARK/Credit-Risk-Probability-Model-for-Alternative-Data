
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import os
from joblib import dump
from datetime import datetime

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def create_features(df):
    """Create new features from the original data."""
    df = df.copy()
    # Extract date features
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['transaction_hour'] = df['transaction_date'].dt.hour
    df['transaction_day'] = df['transaction_date'].dt.day
    df['transaction_month'] = df['transaction_date'].dt.month
    df['transaction_year'] = df['transaction_date'].dt.year

    # Create aggregate features
    df['total_transaction_amount'] = df.groupby('customer_id')['transaction_amount'].transform('sum')
    df['average_transaction_amount'] = df.groupby('customer_id')['transaction_amount'].transform('mean')
    df['transaction_count'] = df.groupby('customer_id')['transaction_amount'].transform('count')
    df['std_dev_transaction_amounts'] = df.groupby('customer_id')['transaction_amount'].transform('std')
    df['std_dev_transaction_amounts'] = df['std_dev_transaction_amounts'].fillna(0)
    
    return df

def calculate_rfm(df):
    """Calculates Recency, Frequency, and Monetary (RFM) values for each customer."""
    # Ensure transaction_date is datetime
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    # Define a snapshot date for RFM calculation (one day after the latest transaction)
    snapshot_date = df['transaction_date'].max() + pd.Timedelta(days=1)

    # Calculate RFM metrics
    rfm = df.groupby('customer_id').agg(
        Recency=('transaction_date', lambda date: (snapshot_date - date.max()).days),
        Frequency=('transaction_id', 'count'),
        Monetary=('transaction_amount', 'sum')
    ).reset_index()

    return rfm

def cluster_customers(rfm_df, random_state=42):
    """Clusters customers into high-risk and low-risk groups based on RFM values."""
    # Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled'], index=rfm_df.index)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled_df)

    # Analyze clusters to identify the high-risk group
    # High-risk group is typically characterized by high Recency (less recent), low Frequency, and low Monetary
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled'])
    
    # Find the cluster with the highest Recency and lowest Frequency/Monetary
    # This is a heuristic; a more robust approach might involve domain expertise or more sophisticated analysis
    high_risk_cluster_id = cluster_centers['Recency_scaled'].idxmax() # Highest Recency
    
    # As a secondary check, we can also look for lowest Frequency and Monetary among the clusters
    # For simplicity, we'll assume the highest Recency cluster is the high-risk one for now.
    # If there's a tie or ambiguity, further analysis would be needed.

    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster_id).astype(int)
    
    return rfm_df[['customer_id', 'is_high_risk']]

def process_data(df):
    """Applies the feature engineering pipeline to the data."""
    
    df_featured = create_features(df)

    # Calculate RFM metrics and cluster customers
    rfm_df = calculate_rfm(df)
    customer_risk = cluster_customers(rfm_df)
    
    # Merge the is_high_risk column back into the main DataFrame
    df_featured = pd.merge(df_featured, customer_risk, on='customer_id', how='left')

    # Define numerical and categorical features
    numerical_features = [
        'transaction_amount', 'age', 'income',
        'transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year',
        'total_transaction_amount', 'average_transaction_amount', 'transaction_count', 'std_dev_transaction_amounts',
        'is_high_risk'
    ]
    categorical_features = ['gender', 'education', 'marital_status']

    # Create pipelines for numerical and categorical features
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop' # drop other columns
    )

    return df_featured, preprocessor

if __name__ == '__main__':
    # Create a dummy raw data file if it doesn't exist
    if not os.path.exists('data/raw/credit_risk_data.csv'):
        print("Creating dummy data file...")
        os.makedirs('data/raw', exist_ok=True)
        dummy_data = {
            'customer_id': [1, 1, 2, 2, 1, 3],
            'transaction_id': [101, 102, 103, 104, 105, 106],
            'transaction_amount': [100, 200, 50, 75, 150, 300],
            'transaction_date': ['2023-01-15 10:30:00', '2023-01-16 12:00:00', '2023-01-17 08:00:00', '2023-01-18 14:00:00', '2023-01-19 18:45:00', '2023-01-20 10:00:00'],
            'age': [35, 35, 45, 45, 35, 28],
            'income': [50000, 50000, 75000, 75000, 50000, 60000],
            'gender': ['Male', 'Male', 'Female', 'Female', 'Male', 'Female'],
            'education': ['Bachelor', 'Bachelor', 'Master', 'Master', 'Bachelor', 'PhD'],
            'marital_status': ['Married', 'Married', 'Single', 'Single', 'Married', 'Single'],
            'target': [0, 0, 1, 1, 0, 1]
        }
        pd.DataFrame(dummy_data).to_csv('data/raw/credit_risk_data.csv', index=False)


    # Load the data
    df = load_data('data/raw/credit_risk_data.csv')

    # Process the data
    processed_df, pipeline = process_data(df)
    
    # Save the processed data
    os.makedirs('data/processed', exist_ok=True)
    processed_df.to_csv('data/processed/processed_credit_risk_data.csv', index=False)
    
    # Save the pipeline
    dump(pipeline, 'src/pipeline.joblib')

    print("Data processing complete. Processed data saved to data/processed/processed_credit_risk_data.csv")
    print("Pipeline saved to src/pipeline.joblib")
