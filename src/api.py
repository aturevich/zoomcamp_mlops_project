import os
import logging
import pandas as pd
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from catboost import CatBoostRegressor, CatBoostClassifier
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from .monitoring import (
    log_prediction, 
    check_data_drift, 
    generate_data_drift_report, 
    init_db, 
    get_recent_predictions, 
    get_data_drift_results
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database
init_db()
logger.info("Database initialized")

# Initialize FastAPI app
app = FastAPI()

# Set up static files and templates
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory="templates")

# Load models
MODELS_DIR = "models"
magnitude_model = CatBoostRegressor().load_model(os.path.join(MODELS_DIR, "magnitude_model.cbm"))
depth_model = CatBoostRegressor().load_model(os.path.join(MODELS_DIR, "depth_model.cbm"))
significant_model = CatBoostClassifier().load_model(os.path.join(MODELS_DIR, "significant_model.cbm"))

# Pydantic models
class EarthquakePredictionInput(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    date: str

class EarthquakePredictionOutput(BaseModel):
    predicted_magnitude: float
    predicted_depth: float
    significant_probability: float

# Global variable for Evidently dashboard
evidently_dashboard = None

# Startup event
@app.on_event("startup")
async def startup_event():
    global evidently_dashboard
    logger.info("Application starting up")

    recent_predictions = get_recent_predictions(n=2000)
    if len(recent_predictions) > 200:
        split_index = len(recent_predictions) // 2
        reference_data = recent_predictions.iloc[split_index:]
        current_data = recent_predictions.iloc[:split_index]

        common_columns = list(set(current_data.columns) & set(reference_data.columns))
        current_data = current_data[common_columns]
        reference_data = reference_data[common_columns]

        column_mapping = ColumnMapping()
        column_mapping.numerical_features = current_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

        evidently_dashboard = Dashboard(tabs=[DataDriftTab()])
        evidently_dashboard.calculate(reference_data, current_data, column_mapping=column_mapping)
        logger.info("Evidently dashboard created")
    else:
        logger.warning(f"Not enough data to generate Evidently AI dashboard. We need at least 200 predictions, but we only have {len(recent_predictions)}.")

# Routes
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/evidently-dashboard")
async def get_evidently_dashboard():
    recent_predictions = get_recent_predictions(n=2000)
    if len(recent_predictions) > 200:
        split_index = len(recent_predictions) // 2
        reference_data = recent_predictions.iloc[split_index:]
        current_data = recent_predictions.iloc[:split_index]

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        
        # Convert the report to JSON
        report_json = json.loads(report.json())
        
        return JSONResponse(content=report_json)
    else:
        return JSONResponse(content={"message": "Not enough data to generate Evidently AI dashboard."})

@app.get("/monitoring/recent_predictions")
async def recent_predictions():
    return get_recent_predictions(n=100)

@app.get("/monitoring/drift_results")
async def drift_results():
    return get_data_drift_results(n=100)

@app.post("/predict_earthquake", response_model=EarthquakePredictionOutput)
async def predict_earthquake(input: EarthquakePredictionInput):
    try:
        input_df = pd.DataFrame([input.dict()])
        input_df['date'] = pd.to_datetime(input_df['date'])
        
        input_df['day_of_year'] = input_df['date'].dt.dayofyear
        input_df['month'] = input_df['date'].dt.month
        input_df['year'] = input_df['date'].dt.year
        input_df['week_of_year'] = input_df['date'].dt.isocalendar().week
        
        features = ["latitude", "longitude", "day_of_year", "month", "year", "week_of_year"]
        input_df = input_df[features]
        
        magnitude_prediction = magnitude_model.predict(input_df)[0]
        depth_prediction = depth_model.predict(input_df)[0]
        significant_probability = significant_model.predict_proba(input_df)[0][1]
        
        log_prediction(input.latitude, input.longitude, depth_prediction, magnitude_prediction, significant_probability)
        
        recent_predictions = get_recent_predictions(n=1000)
        drift_score, drift_detected = check_data_drift(recent_predictions.iloc[500:], recent_predictions.iloc[:500])
        if drift_score is not None and drift_detected is not None:
            logger.info(f"Data drift check results - Score: {drift_score}, Detected: {drift_detected}")
            if drift_detected:
                logger.warning(f"Data drift detected! Score: {drift_score}")
                generate_data_drift_report(recent_predictions.iloc[500:], recent_predictions.iloc[:500], "reports/data_drift_report.html")
        else:
            logger.warning("Data drift check failed or returned no results.")
        
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
