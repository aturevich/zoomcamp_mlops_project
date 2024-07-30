import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from prefect import task, flow


@task
def load_data(filepath):
    """Load the data from a CSV file."""
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
def encode_column(data, column):
    """Encode a specified column to integers."""
    label_encoder = LabelEncoder()
    data[column] = label_encoder.fit_transform(data[column])
    return data


@task
def remove_duplicates(data):
    data = data.drop_duplicates()
    return data


@task
def save_data(data, output_filepath):
    """Save the filtered dataset to a new CSV file."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    data.to_csv(output_filepath, index=False)


@task
def create_datetime_features(data):
    data["datetime"] = pd.to_datetime(data["date"], format="ISO8601")
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["year"] = data["datetime"].dt.year
    data["month"] = data["datetime"].dt.month
    data["day"] = data["datetime"].dt.day
    data["hour"] = data["datetime"].dt.hour
    data["day_of_week"] = data["datetime"].dt.dayofweek
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    return data


@task
def create_spatial_features(data):
    kmeans = KMeans(n_clusters=10, random_state=42)
    data["location_cluster"] = kmeans.fit_predict(data[["latitude", "longitude"]])
    return data


@task
def create_rolling_features(data):
    data = data.sort_values("datetime")
    for cluster in data["location_cluster"].unique():
        cluster_mask = data["location_cluster"] == cluster
        cluster_data = data[cluster_mask]
        window_7d = 7 * 24
        window_30d = 30 * 24
        data.loc[cluster_mask, "mag_rolling_mean_7d"] = (
            cluster_data["magnitudo"]
            .rolling(window=window_7d, min_periods=1)
            .mean()
            .values
        )
        data.loc[cluster_mask, "mag_rolling_mean_30d"] = (
            cluster_data["magnitudo"]
            .rolling(window=window_30d, min_periods=1)
            .mean()
            .values
        )
        data.loc[cluster_mask, "depth_rolling_mean_7d"] = (
            cluster_data["depth"].rolling(window=window_7d, min_periods=1).mean().values
        )
        data.loc[cluster_mask, "depth_rolling_mean_30d"] = (
            cluster_data["depth"]
            .rolling(window=window_30d, min_periods=1)
            .mean()
            .values
        )
        time_diff = cluster_data["datetime"].diff().dt.total_seconds() / 3600
        data.loc[cluster_mask, "time_since_last_eq"] = time_diff.values
    return data


@task
def create_interaction_features(data):
    data["lat_long_interaction"] = data["latitude"] * data["longitude"]
    data["mag_depth_interaction"] = data["magnitudo"] * data["depth"]
    return data


@flow(name="data_preparation_flow")
def data_preparation_pipeline():
    filepath = "data/raw/Earthquakes-1990-2023.csv"
    output_filepath = "data/processed/filtered_processed_features.csv"

    data = load_data(filepath)
    data = filter_data(data, "data_type", "earthquake")
    data = filter_data(data, "status", "reviewed")
    data = drop_columns(data, ["tsunami", "significance", "status", "data_type"])
    data = encode_column(data, "state")
    data = create_datetime_features(data)
    data = create_spatial_features(data)
    data = remove_duplicates(data)
    data = create_rolling_features(data)
    data = create_interaction_features(data)
    data = drop_columns(data, ["date", "time"])
    save_data(data, output_filepath)


if __name__ == "__main__":
    data_preparation_pipeline()
