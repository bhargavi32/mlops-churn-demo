# app.py — FastAPI Model Serving

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import mlflow.sklearn
import numpy as np

# Initialize FastAPI
app = FastAPI(title="Churn Prediction API")

# Load model and scaler
model = mlflow.sklearn.load_model("models/random_forest_model")
scaler = joblib.load("models/scaler.pkl")
feature_order = joblib.load("models/feature_order.pkl")  # ensures correct column order

# Request schema
class ChurnInput(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_churn(data: ChurnInput):

    # Convert input to dict
    input_dict = data.dict()

    # Arrange in model's training order
    ordered_values = [input_dict[feat] for feat in feature_order]

    # Convert to array
    features = np.array(ordered_values).reshape(1, -1)

    # Scale features
    scaled_features = scaler.transform(features)

    # Predict
    pred = model.predict(scaled_features)[0]

    return {
        "churn_prediction": int(pred),
        "message": "Customer is likely to churn" if pred == 1 else "Customer is not likely to churn"
    }
