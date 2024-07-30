import pandas as pd
from src.data_prep import filter_data, create_datetime_features, create_spatial_features


def test_filter_data():
    data = pd.read_csv("tests/sample_data/sample_earthquake.csv")
    filtered = filter_data(data, "type", "earthquake")
    assert len(filtered) > 0
    assert all(filtered["type"] == "earthquake")


def test_create_datetime_features():
    data = pd.read_csv("tests/sample_data/sample_earthquake.csv")
    result = create_datetime_features(data)
    assert "year" in result.columns
    assert "month" in result.columns
    assert "day" in result.columns
    assert "hour" in result.columns


def test_create_spatial_features():
    data = pd.read_csv("tests/sample_data/sample_earthquake.csv")
    result = create_spatial_features(data)
    assert "location_cluster" in result.columns
