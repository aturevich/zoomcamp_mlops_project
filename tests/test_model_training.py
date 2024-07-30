import pandas as pd
import numpy as np
from src.model_training import train_model, evaluate_model


def test_train_model():
    data = pd.read_csv("tests/sample_data/sample_earthquake.csv")
    X = data[["latitude", "longitude", "depth"]]
    y = data["mag"]
    model = train_model(X, y, {"n_estimators": 10})
    assert model is not None
    assert hasattr(model, "predict")


def test_evaluate_model():
    data = pd.read_csv("tests/sample_data/sample_earthquake.csv")
    X = data[["latitude", "longitude", "depth"]]
    y = data["mag"]
    model = train_model(X, y, {"n_estimators": 10})
    rmse, r2 = evaluate_model(model, X, y)
    assert isinstance(rmse, float)
    assert isinstance(r2, float)
