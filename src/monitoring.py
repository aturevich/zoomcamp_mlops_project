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
DB_FILE = "data/monitoring.db"

# Ensure the data directory exists
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)


# Initialize database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS predictions
                 (timestamp TEXT, latitude REAL, longitude REAL, depth REAL,
                  magnitude REAL, significant_probability REAL)"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS data_drift_results
                 (timestamp TEXT, drift_score REAL, drift_detected BOOLEAN)"""
    )
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")


# Log prediction
def log_prediction(latitude, longitude, depth, magnitude, significant_probability):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute(
        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
        (timestamp, latitude, longitude, depth, magnitude, significant_probability),
    )
    conn.commit()
    conn.close()


# Check for data drift
def check_data_drift(reference_data, current_data):
    try:
        # Ensure column names match
        reference_columns = set(reference_data.columns)
        current_columns = set(current_data.columns)

        common_columns = list(reference_columns.intersection(current_columns))

        logger.info(f"Common columns: {common_columns}")
        logger.info(f"Reference data shape: {reference_data.shape}")
        logger.info(f"Current data shape: {current_data.shape}")

        profile = Profile(sections=[DataDriftProfileSection()])
        profile.calculate(reference_data[common_columns], current_data[common_columns])

        # Parse the JSON string into a dictionary
        profile_json = json.loads(profile.json())

        # Log the entire profile JSON for debugging
        logger.info(f"Full Profile JSON: {json.dumps(profile_json, indent=2)}")

        # Try to extract drift information, with fallbacks
        drift_score = profile_json.get("data_drift", {}).get(
            "data_drift_score",
            profile_json.get("data_drift", {}).get("share_of_drifted_columns", None),
        )
        drift_detected = profile_json.get("data_drift", {}).get(
            "data_drift_detected", None
        )

        logger.info(f"Extracted drift_score: {drift_score}")
        logger.info(f"Extracted drift_detected: {drift_detected}")

        if drift_score is None or drift_detected is None:
            logger.warning(
                "Could not extract drift score or detection status from the profile."
            )
            # Instead of returning None, let's set default values
            drift_score = 0.0
            drift_detected = False

        # Always insert into the database, even if we couldn't extract the values
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        timestamp = datetime.now().isoformat()
        c.execute(
            "INSERT INTO data_drift_results VALUES (?, ?, ?)",
            (timestamp, drift_score, int(drift_detected)),
        )
        conn.commit()
        conn.close()

        return drift_score, drift_detected
    except Exception as e:
        logger.exception(f"An error occurred during data drift check: {str(e)}")
        return 0.0, False


# Generate data drift report
def generate_data_drift_report(reference_data, current_data, output_path):
    try:
        # Ensure column names match
        reference_columns = set(reference_data.columns)
        current_columns = set(current_data.columns)

        common_columns = list(reference_columns.intersection(current_columns))

        dashboard = Dashboard(tabs=[DataDriftTab()])
        dashboard.calculate(
            reference_data[common_columns], current_data[common_columns]
        )
        dashboard.save(output_path)
        logger.info(f"Data drift report generated and saved to {output_path}")
    except Exception as e:
        logger.exception(
            f"An error occurred while generating the data drift report: {str(e)}"
        )


# Get recent predictions
def get_recent_predictions(n=100):
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?"
    df = pd.read_sql_query(query, conn, params=(n,))
    conn.close()
    logger.info(f"Retrieved {len(df)} predictions from database")
    logger.debug(f"Columns in DataFrame: {df.columns}")
    logger.debug(f"First few rows: {df.head().to_dict(orient='records')}")
    return df


# Get data drift results
def get_data_drift_results(n=100):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        "SELECT * FROM data_drift_results ORDER BY timestamp DESC LIMIT ?",
        conn,
        params=(n,),
    )
    conn.close()
    return df
