from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
import logging
import os
from .monitoring import log_prediction, check_data_drift, generate_data_drift_report, init_db
from src.ref_data import load_reference_data

# Initialize the database
init_db()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the models
MODELS_DIR = "models"
magnitude_model = CatBoostRegressor()
magnitude_model.load_model(os.path.join(MODELS_DIR, "magnitude_model.cbm"))
depth_model = CatBoostRegressor()
depth_model.load_model(os.path.join(MODELS_DIR, "depth_model.cbm"))
significant_model = CatBoostClassifier()
significant_model.load_model(os.path.join(MODELS_DIR, "significant_model.cbm"))

# Load reference data
reference_data = load_reference_data()

class EarthquakePredictionInput(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    date: str

class EarthquakePredictionOutput(BaseModel):
    predicted_magnitude: float
    predicted_depth: float
    significant_probability: float

@app.post("/predict_earthquake", response_model=EarthquakePredictionOutput)
async def predict_earthquake(input: EarthquakePredictionInput):
    try:
        # Prepare input data
        input_df = pd.DataFrame([input.dict()])
        input_df['date'] = pd.to_datetime(input_df['date'])
        
        # Add time-based features
        input_df['day_of_year'] = input_df['date'].dt.dayofyear
        input_df['month'] = input_df['date'].dt.month
        input_df['year'] = input_df['date'].dt.year
        input_df['week_of_year'] = input_df['date'].dt.isocalendar().week
        
        # Select only the features used by the model
        features = ["latitude", "longitude", "day_of_year", "month", "year", "week_of_year"]
        input_df = input_df[features]
        
        # Make predictions
        magnitude_prediction = magnitude_model.predict(input_df)[0]
        depth_prediction = depth_model.predict(input_df)[0]
        significant_probability = significant_model.predict_proba(input_df)[0][1]  # Probability of class 1 (significant event)
        
        # Log prediction
        log_prediction(input.latitude, input.longitude, depth_prediction, magnitude_prediction, significant_probability)
        
        # Check for data drift
        drift_score, drift_detected = check_data_drift(reference_data, input_df)
        if drift_score is not None and drift_detected is not None:
            logger.info(f"Data drift check results - Score: {drift_score}, Detected: {drift_detected}")
            if drift_detected:
                logger.warning(f"Data drift detected! Score: {drift_score}")
                generate_data_drift_report(reference_data, input_df, "reports/data_drift_report.html")
        else:
            logger.warning("Data drift check failed or returned no results. Check the logs for more information.")
        
        return EarthquakePredictionOutput(
            predicted_magnitude=float(magnitude_prediction),
            predicted_depth=float(depth_prediction),
            significant_probability=float(significant_probability)
        )
    except Exception as e:
        logger.exception("An error occurred during earthquake prediction")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Initialize the database when the app starts
@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("Database initialized")
