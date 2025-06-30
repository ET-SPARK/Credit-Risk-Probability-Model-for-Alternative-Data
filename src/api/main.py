from fastapi import FastAPI
from mlflow import pyfunc
from src.api.pydantic_models import PredictRequest, PredictResponse

app = FastAPI()

# Load the best model from MLflow registry
# Replace 'CreditRiskModel' with the actual name of your best model in MLflow
# and '1' with the appropriate version if you have multiple versions.
model = pyfunc.load_model("models:/CreditRiskModel/Production")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    # Convert the list of features to a format suitable for your model
    # This might require reshaping or converting to a pandas DataFrame
    # depending on how your model was trained.
    # For a scikit-learn model, it's usually a 2D array or DataFrame.
    import pandas as pd
    features_df = pd.DataFrame([request.features])

    prediction = model.predict(features_df)
    # Assuming the model outputs a single probability for binary classification
    probability = float(prediction[0])

    return PredictResponse(probability=probability)