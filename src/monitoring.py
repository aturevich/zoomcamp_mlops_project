import sqlite3
import pandas as pd
from datetime import datetime
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database file path
DB_FILE = 'data/monitoring.db'

# Ensure the data directory exists
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (timestamp TEXT, latitude REAL, longitude REAL, depth REAL, 
                  magnitude REAL, significant_probability REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS data_drift_results
                 (timestamp TEXT, drift_score REAL, drift_detected BOOLEAN)''')
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

# Log prediction
def log_prediction(latitude, longitude, depth, magnitude, significant_probability):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
              (timestamp, latitude, longitude, depth, magnitude, significant_probability))
    conn.commit()
    conn.close()

# Check for data drift
def check_data_drift(reference_data, current_data):
    try:
        # Ensure column names match
        reference_columns = set(reference_data.columns)
        current_columns = set(current_data.columns)
        
        common_columns = list(reference_columns.intersection(current_columns))
        
        profile = Profile(sections=[DataDriftProfileSection()])
        profile.calculate(reference_data[common_columns], current_data[common_columns])
        
        # Parse the JSON string into a dictionary
        profile_json = json.loads(profile.json())
        
        # Log the structure of profile_json for debugging
        logger.info(f"Profile JSON structure: {json.dumps(profile_json, indent=2)}")
        
        # Try to extract drift information, with fallbacks
        drift_score = profile_json.get("data_drift", {}).get("data_drift_score", 
                      profile_json.get("data_drift", {}).get("share_of_drifted_columns", None))
        drift_detected = profile_json.get("data_drift", {}).get("data_drift_detected", None)
        
        if drift_score is None or drift_detected is None:
            logger.warning("Could not extract drift score or detection status from the profile.")
            return None, None
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        timestamp = datetime.now().isoformat()
        c.execute("INSERT INTO data_drift_results VALUES (?, ?, ?)",
                  (timestamp, drift_score, drift_detected))
        conn.commit()
        conn.close()
        
        return drift_score, drift_detected
    except Exception as e:
        logger.exception(f"An error occurred during data drift check: {str(e)}")
        return None, None

# Generate data drift report
def generate_data_drift_report(reference_data, current_data, output_path):
    try:
        # Ensure column names match
        reference_columns = set(reference_data.columns)
        current_columns = set(current_data.columns)
        
        common_columns = list(reference_columns.intersection(current_columns))
        
        dashboard = Dashboard(tabs=[DataDriftTab()])
        dashboard.calculate(reference_data[common_columns], current_data[common_columns])
        dashboard.save(output_path)
        logger.info(f"Data drift report generated and saved to {output_path}")
    except Exception as e:
        logger.exception(f"An error occurred while generating the data drift report: {str(e)}")

# Get recent predictions
def get_recent_predictions(n=100):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", conn, params=(n,))
    conn.close()
    return df

# Get data drift results
def get_data_drift_results(n=100):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM data_drift_results ORDER BY timestamp DESC LIMIT ?", conn, params=(n,))
    conn.close()
    return df
