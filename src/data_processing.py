
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
from joblib import dump

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

def process_data(df, target_column=None):
    """Applies the feature engineering pipeline to the data."""
    
    df_featured = create_features(df)

    if target_column:
        X = df_featured.drop(columns=[target_column])
        y = df_featured[target_column]
    else:
        X = df_featured
        y = None

    # Define numerical and categorical features
    numerical_features = [
        'transaction_amount', 'age', 'income',
        'transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year',
        'total_transaction_amount', 'average_transaction_amount', 'transaction_count', 'std_dev_transaction_amounts'
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

    # Apply the feature engineering pipeline
    X_processed = preprocessor.fit_transform(X)

    if y is not None:
        return X_processed, y, preprocessor
    else:
        return X_processed, preprocessor

if __name__ == '__main__':
    # Create a dummy raw data file if it doesn't exist
    if not os.path.exists('data/raw/credit_risk_data.csv'):
        print("Creating dummy data file...")
        os.makedirs('data/raw', exist_ok=True)
        dummy_data = {
            'customer_id': [1, 1, 2, 2, 1, 3],
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
    X_processed, y, pipeline = process_data(df, target_column='target')
    
    # Save the processed data
    os.makedirs('data/processed', exist_ok=True)
    processed_df = pd.concat([pd.DataFrame(X_processed), y], axis=1)
    processed_df.to_csv('data/processed/processed_credit_risk_data.csv', index=False)
    
    # Save the pipeline
    dump(pipeline, 'src/pipeline.joblib')

    print("Data processing complete. Processed data saved to data/processed/processed_credit_risk_data.csv")
    print("Pipeline saved to src/pipeline.joblib")
