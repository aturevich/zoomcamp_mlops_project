import pandas as pd
from src.monitoring import generate_data_drift_report, check_data_drift, DB_FILE
import os
import sqlite3


def test_generate_data_drift_report(tmp_path):
    data = pd.read_csv("tests/sample_data/sample_earthquake.csv")
    reference_data = data.iloc[:50].copy()
    current_data = data.iloc[50:].copy()

    output_path = tmp_path / "data_drift_report.html"
    generate_data_drift_report(reference_data, current_data, str(output_path))
    assert output_path.exists()


def test_check_data_drift():
    # Ensure the database file exists
    if not os.path.exists(DB_FILE):
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS data_drift_results
                     (timestamp TEXT, drift_score REAL, drift_detected BOOLEAN)"""
        )
        conn.commit()
        conn.close()

    data = pd.read_csv("tests/sample_data/sample_earthquake.csv")
    reference_data = data.iloc[:50].copy()
    current_data = data.iloc[50:].copy()

    print(f"Reference data columns: {reference_data.columns}")
    print(f"Current data shape: {reference_data.shape}")
    print(f"Current data columns: {current_data.columns}")
    print(f"Current data shape: {current_data.shape}")

    drift_score, drift_detected = check_data_drift(reference_data, current_data)

    print(f"Returned drift_score: {drift_score}")
    print(f"Returned drift_detected: {drift_detected}")

    # Check that the function returns a tuple
    assert isinstance(
        (drift_score, drift_detected), tuple
    ), "check_data_drift should return a tuple"

    # Check individual return values
    assert isinstance(drift_score, float), f"Expected float, got {type(drift_score)}"
    assert (
        0 <= drift_score <= 1
    ), f"Drift score should be between 0 and 1, got {drift_score}"
    assert isinstance(
        drift_detected, bool
    ), f"Expected bool, got {type(drift_detected)}"

    # Check if data was inserted into the database
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM data_drift_results ORDER BY timestamp DESC LIMIT 1")
    result = c.fetchone()
    conn.close()

    assert result is not None, "No data was inserted into the database"
    assert len(result) == 3, "Expected 3 columns in the database result"
    assert isinstance(result[1], float), "Drift score in database should be a float"
    assert isinstance(
        result[2], int
    ), "Drift detected in database should be an integer (0 or 1)"
