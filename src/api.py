from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date
import joblib
import pandas as pd

app = FastAPI()

# Load the models (you'll need to train these models)
magnitude_model = joblib.load("models/magnitude_prediction_model.joblib")
location_model = joblib.load("models/location_prediction_model.joblib")


class MagnitudePredictionInput(BaseModel):
    latitude: float
    longitude: float
    date: date


class MagnitudePredictionOutput(BaseModel):
    predicted_magnitude: float


class LocationPredictionInput(BaseModel):
    date: date


class LocationPredictionOutput(BaseModel):
    predicted_locations: list[dict]


@app.post("/predict_magnitude", response_model=MagnitudePredictionOutput)
async def predict_magnitude(input: MagnitudePredictionInput):
    try:
        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input.dict()])

        # Convert date to numerical features (e.g., day of year, month, year)
        input_df["day_of_year"] = input_df["date"].dt.dayofyear
        input_df["month"] = input_df["date"].dt.month
        input_df["year"] = input_df["date"].dt.year

        # Make prediction
        prediction = magnitude_model.predict(input_df)[0]

        return MagnitudePredictionOutput(predicted_magnitude=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_locations", response_model=LocationPredictionOutput)
async def predict_locations(input: LocationPredictionInput):
    try:
        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input.dict()])

        # Convert date to numerical features
        input_df["day_of_year"] = input_df["date"].dt.dayofyear
        input_df["month"] = input_df["date"].dt.month
        input_df["year"] = input_df["date"].dt.year

        # Make prediction (assume the model returns a list of potential locations)
        predictions = location_model.predict(input_df)

        # Convert predictions to list of dictionaries
        predicted_locations = [
            {"latitude": lat, "longitude": lon, "probability": prob}
            for lat, lon, prob in predictions
        ]

        return LocationPredictionOutput(predicted_locations=predicted_locations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
