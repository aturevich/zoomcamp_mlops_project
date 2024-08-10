import pandas as pd
from src.train_magnitude_model import preprocess_data


def test_preprocess_data():
    # Create a sample DataFrame
    data = pd.DataFrame(
        {
            "date": pd.date_range(start="2021-01-01", periods=5),
            "latitude": [1.0, 2.0, 3.0, 4.0, 5.0],
            "longitude": [1.0, 2.0, 3.0, 4.0, 5.0],
            "depth": [10.0, 20.0, 30.0, 40.0, 50.0],
            "mag": [3.0, 4.0, 5.0, 4.0, 3.0],
        }
    )

    processed_data = preprocess_data(data)

    # Check if the expected columns are present
    expected_columns = [
        "date",
        "latitude",
        "longitude",
        "depth",
        "mag",
        "day_of_year",
        "month",
        "year",
        "week_of_year",
        "significant_event",
    ]
    for column in expected_columns:
        assert (
            column in processed_data.columns
        ), f"Expected column {column} not found in processed data"

    # Check if 'significant_event' is correctly calculated
    assert all(
        processed_data["significant_event"]
        == (processed_data["mag"] >= 5.0).astype(int)
    )

    # Check if date-related columns are correctly calculated
    assert all(processed_data["day_of_year"] == processed_data["date"].dt.dayofyear)
    assert all(processed_data["month"] == processed_data["date"].dt.month)
    assert all(processed_data["year"] == processed_data["date"].dt.year)
    assert all(
        processed_data["week_of_year"] == processed_data["date"].dt.isocalendar().week
    )

    # Check if the number of rows hasn't changed
    assert len(processed_data) == len(data)
