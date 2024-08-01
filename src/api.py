from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import date
import joblib
import pandas as pd
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Earthquake Prediction API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the models
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

class PredictedLocation(BaseModel):
    latitude: float
    longitude: float
    probability: float

class LocationPredictionOutput(BaseModel):
    predicted_locations: List[PredictedLocation]

def preprocess_input(input_data: dict):
    df = pd.DataFrame([input_data])
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    features = ['latitude', 'longitude', 'day_of_year', 'month', 'year', 'week_of_year']
    return df[features]

@app.post("/predict_magnitude", response_model=MagnitudePredictionOutput)
async def predict_magnitude(input: MagnitudePredictionInput):
    try:
        input_dict = input.dict()
        input_dict['date'] = input_dict['date'].isoformat()  # Convert date to string
        input_df = preprocess_input(input_dict)
        prediction = magnitude_model.predict(input_df)[0]
        return MagnitudePredictionOutput(predicted_magnitude=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_locations", response_model=LocationPredictionOutput)
async def predict_locations(input: LocationPredictionInput):
    try:
        input_dict = input.dict()
        input_dict['date'] = input_dict['date'].isoformat()  # Convert date to string
        input_df = preprocess_input(input_dict)
        predictions = location_model.predict(input_df)
        predicted_locations = [
            PredictedLocation(latitude=lat, longitude=lon, probability=prob)
            for lat, lon, prob in predictions
        ]
        return LocationPredictionOutput(predicted_locations=predicted_locations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Earthquake Prediction API</title>
        </head>
        <body>
            <h1>Welcome to the Earthquake Prediction API</h1>
            <p>Available endpoints:</p>
            <ul>
                <li><a href="/docs">/docs</a> - Interactive API documentation</li>
                <li><a href="/predict_magnitude">/predict_magnitude</a> - Predict earthquake magnitude</li>
                <li><a href="/predict_locations">/predict_locations</a> - Predict potential earthquake locations</li>
                <li><a href="/health">/health</a> - API health check</li>
            </ul>
        </body>
    </html>
    """
