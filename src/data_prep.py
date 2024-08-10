import os
import pandas as pd
import numpy as np
from prefect import task, flow


@task
def load_data(filepath):
    """Load the data from a CSV file."""
    print(f"Attempting to load data from: {filepath}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('data/raw')}")
    data = pd.read_csv(filepath)
    return data


@task
def filter_data(data, column, value):
    """Filter the dataset to include only rows where the column matches the given value."""
    filtered_data = data[data[column].str.lower() == value.lower()].copy()
    return filtered_data


@task
def drop_columns(data, columns_to_drop):
    """Drop unnecessary columns from the dataset."""
    data = data.drop(columns=columns_to_drop)
    return data


@task
def remove_duplicates(data):
    data = data.drop_duplicates()
    return data


@task
def create_datetime_features(data):
    data["datetime"] = pd.to_datetime(data["date"], format="ISO8601")
    data["year"] = data["datetime"].dt.year
    data["month"] = data["datetime"].dt.month
    data["day_of_year"] = data["datetime"].dt.dayofyear
    data["week_of_year"] = data["datetime"].dt.isocalendar().week
    return data


@task
def save_processed_data(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


@flow(name="data_preparation_flow")
def data_preparation_pipeline(input_filepath, output_filepath):
    data = load_data(input_filepath)

    # Print column names to see what we're working with
    print("Columns in the dataset:", data.columns.tolist())

    data = filter_data(data, "data_type", "earthquake")
    data = filter_data(data, "status", "reviewed")
    data = drop_columns(
        data, ["tsunami", "significance", "status", "data_type", "time"]
    )
    data = create_datetime_features(data)
    data = remove_duplicates(data)

    # Adjust column names based on what's actually in your dataset
    magnitude_column = (
        "magnitudo"
        if "magnitudo" in data.columns
        else "mag" if "mag" in data.columns else None
    )
    if magnitude_column is None:
        raise ValueError("No magnitude column found in the dataset.")

    # Keep only the necessary columns
    columns_to_keep = [
        "datetime",
        "latitude",
        "longitude",
        "depth",
        magnitude_column,
        "year",
        "month",
        "day_of_year",
        "week_of_year",
    ]
    data = data[columns_to_keep]

    # Rename the magnitude column to 'mag' for consistency
    data = data.rename(columns={magnitude_column: "mag"})

    save_processed_data(data, output_filepath)


if __name__ == "__main__":
    input_filepath = "data/raw/Eartquakes-1990-2023.csv"
    output_filepath = "data/processed/earthquake_data.csv"
    data_preparation_pipeline(input_filepath, output_filepath)
