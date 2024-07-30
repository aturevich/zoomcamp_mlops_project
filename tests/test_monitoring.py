import pandas as pd
import pytest
from src.monitoring import generate_evidently_report


def test_generate_evidently_report(tmp_path):
    data = pd.read_csv("tests/sample_data/sample_earthquake.csv")
    reference_data = data.iloc[:50].copy()
    current_data = data.iloc[50:].copy()

    # Add a 'prediction' column for testing purposes
    reference_data["prediction"] = reference_data["mag"]
    current_data["prediction"] = current_data["mag"]

    output_path = tmp_path / "test_report.html"
    generate_evidently_report(reference_data, current_data, str(output_path))
    assert output_path.exists()
