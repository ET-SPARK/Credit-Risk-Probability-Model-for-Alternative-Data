
import pandas as pd
from joblib import load
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processing import create_features

def predict(new_data_path, model_path, pipeline_path):
    """Makes predictions on new data."""
    
    # Load the new data
    df = pd.read_csv(new_data_path)
    
    # Create features
    df_featured = create_features(df)
    
    # Load the model and pipeline
    model = load(model_path)
    pipeline = load(pipeline_path)
    
    # Process the new data using the loaded pipeline
    processed_data = pipeline.transform(df_featured)
    
    # Make predictions
    predictions = model.predict(processed_data)
    
    return predictions

if __name__ == '__main__':
    # Adjust paths to be relative to the project root, assuming the script is run from there
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    new_data_path = os.path.join(project_root, 'data/raw/new_credit_risk_data.csv')
    model_path = os.path.join(project_root, 'src/model.joblib')
    pipeline_path = os.path.join(project_root, 'src/pipeline.joblib')
    
    # Create a dummy new data file if it doesn't exist
    if not os.path.exists(new_data_path):
        print("Creating dummy new data file...")
        os.makedirs(os.path.dirname(new_data_path), exist_ok=True)
        dummy_data = {
            'customer_id': [4, 5],
            'transaction_amount': [250, 350],
            'transaction_date': ['2023-01-21 11:00:00', '2023-01-22 15:30:00'],
            'age': [30, 40],
            'income': [65000, 80000],
            'gender': ['Male', 'Female'],
            'education': ['Master', 'PhD'],
            'marital_status': ['Single', 'Married']
        }
        pd.DataFrame(dummy_data).to_csv(new_data_path, index=False)

    # Check if model and pipeline exist
    if not os.path.exists(model_path) or not os.path.exists(pipeline_path):
        print("Model or pipeline not found. Please run src/train.py first.")
    else:
        predictions = predict(new_data_path, model_path, pipeline_path)
        print("Predictions:", predictions)
