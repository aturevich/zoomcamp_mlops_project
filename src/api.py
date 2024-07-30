# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the model
model = joblib.load("models/random_forest_model.joblib")


class PredictionInput(BaseModel):
    features: list[float]


class PredictionOutput(BaseModel):
    prediction: float


@app.post("/predict", response_model=PredictionOutput)
async def predict(input: PredictionInput):
    try:
        features = np.array(input.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return PredictionOutput(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
