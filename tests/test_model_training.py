import pandas as pd
import numpy as np
import mlflow
from src.train_magnitude_model import (
    train_earthquake_model,
    load_data,
    preprocess_data,
    prepare_features,
    train_model,
    evaluate_model,
)
import os
import tempfile


def test_load_data():
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
        tmp.write("date,latitude,longitude,depth,mag\n")
        tmp.write("2021-01-01,35.0,-118.0,10.0,3.5\n")
        tmp_path = tmp.name

    data = load_data(tmp_path)
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    os.unlink(tmp_path)


def test_preprocess_data():
    data = pd.DataFrame(
        {
            "date": ["2021-01-01"],
            "latitude": [35.0],
            "longitude": [-118.0],
            "depth": [10.0],
            "mag": [3.5],
        }
    )
    processed_data = preprocess_data(data)
    assert "day_of_year" in processed_data.columns
    assert "month" in processed_data.columns
    assert "year" in processed_data.columns


def test_prepare_features():
    data = pd.DataFrame(
        {
            "date": pd.date_range(start="2021-01-01", periods=5),
            "latitude": [35.0] * 5,
            "longitude": [-118.0] * 5,
            "depth": [10.0] * 5,
            "mag": [3.5] * 5,
        }
    )
    data = preprocess_data(data)
    X, y_magnitude, y_depth, y_significant = prepare_features(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y_magnitude, pd.Series)
    assert isinstance(y_depth, pd.Series)
    assert isinstance(y_significant, pd.Series)


def test_train_model():
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.Series([1.0, 2.0, 3.0])
    model = train_model(X, y, X, y, target="magnitude")
    assert hasattr(model, "predict")


class DummyModel:
    def predict(self, X):
        return X["feature"]  # Just return the feature as the prediction


def test_evaluate_model():
    # Create a dummy model
    dummy_model = DummyModel()

    # Create dummy test data
    X_test = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y_true = pd.Series([1.0, 2.0, 3.0])

    # Call evaluate_model with the correct parameters
    metrics = evaluate_model(dummy_model, X_test, y_true, task="regression")

    # Add assertions to check the returned metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics

    # You can add more specific checks if you want, e.g.:
    assert metrics["mse"] >= 0
    assert 0 <= metrics["r2"] <= 1


def test_train_earthquake_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_data_path = os.path.join(tmpdir, "input.csv")
        pd.DataFrame(
            {
                "date": pd.date_range(start="2021-01-01", periods=100),
                "latitude": np.random.uniform(30, 40, 100),
                "longitude": np.random.uniform(-120, -110, 100),
                "depth": np.random.uniform(0, 100, 100),
                "mag": np.random.uniform(2, 6, 100),
            }
        ).to_csv(input_data_path, index=False)

        model_output_path = os.path.join(tmpdir, "models")

        # Ensure any existing runs are ended before starting the test
        mlflow.end_run()

        try:
            result = train_earthquake_model(input_data_path, model_output_path)
            assert os.path.exists(result)
        finally:
            # Ensure runs are ended after the test
            mlflow.end_run()
