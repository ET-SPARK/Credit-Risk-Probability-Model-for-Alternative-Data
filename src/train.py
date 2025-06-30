import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump
import os

def train_model(processed_data_path, model_save_path):
    """Trains a model on the processed data."""
    
    # Load the processed data
    df = pd.read_csv(processed_data_path)
    
    # The last column is the target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Train a simple model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Save the model
    dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    processed_data_path = 'data/processed/processed_credit_risk_data.csv'
    model_save_path = 'src/model.joblib'
    
    # Check if processed data exists
    if not os.path.exists(processed_data_path):
        print("Processed data not found. Please run src/data_processing.py first.")
    else:
        train_model(processed_data_path, model_save_path)