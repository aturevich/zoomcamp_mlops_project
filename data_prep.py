# data_prep.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

def load_data(filepath):
    """Load the data from a CSV file."""
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)
    print(f"Data loaded. Shape: {data.shape}")
    return data

def filter_data(data, column, value):
    """Filter the dataset to include only rows where the column matches the given value."""
    print(f"Filtering data to include only rows where {column} is '{value}'...")
    filtered_data = data[data[column].str.lower() == value.lower()].copy()
    print(f"Filtered data. Shape: {filtered_data.shape}")
    return filtered_data

def drop_columns(data, columns_to_drop):
    """Drop unnecessary columns from the dataset."""
    print(f"Dropping unnecessary columns: {', '.join(columns_to_drop)}...")
    data = data.drop(columns=columns_to_drop)
    print(f"Columns dropped. Shape: {data.shape}")
    return data

def encode_column(data, column):
    """Encode a specified column to integers."""
    print(f"Encoding '{column}' column to integers...")
    label_encoder = LabelEncoder()
    data[column] = label_encoder.fit_transform(data[column])
    print(f"{column} column encoded.")
    return data

def remove_duplicates(data):
    print("Removing duplicate rows...")
    initial_shape = data.shape
    data = data.drop_duplicates()
    print(f"Removed {initial_shape[0] - data.shape[0]} duplicate rows.")
    return data

def save_data(data, output_filepath):
    """Save the filtered dataset to a new CSV file."""
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    print(f"Saving filtered data to {output_filepath}...")
    data.to_csv(output_filepath, index=False)
    print(f"Filtered data saved to {output_filepath}")

def create_datetime_features(data):
    print("Creating datetime features...")
    
    # Convert 'date' to datetime
    data['datetime'] = pd.to_datetime(data['date'], format='ISO8601')
    
    # Ensure 'datetime' is recognized as datetime type
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # Extracting features from datetime
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['day'] = data['datetime'].dt.day
    data['hour'] = data['datetime'].dt.hour
    data['day_of_week'] = data['datetime'].dt.dayofweek
    
    # Cyclical encoding of time features
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    
    return data

def create_spatial_features(data):
    print("Creating spatial features...")
    # Create clusters based on latitude and longitude
    kmeans = KMeans(n_clusters=10, random_state=42)
    data['location_cluster'] = kmeans.fit_predict(data[['latitude', 'longitude']])
    
    return data

def create_rolling_features(data):
    print("Creating rolling features...")
    # Sort data by datetime
    data = data.sort_values('datetime')
    
    # Create rolling averages of magnitude and depth
    for cluster in data['location_cluster'].unique():
        cluster_mask = data['location_cluster'] == cluster
        cluster_data = data[cluster_mask]
        
        # Use a numeric window instead of time-based window
        window_7d = 7 * 24  # Assuming average of 1 earthquake per hour
        window_30d = 30 * 24
        
        data.loc[cluster_mask, 'mag_rolling_mean_7d'] = cluster_data['magnitudo'].rolling(window=window_7d, min_periods=1).mean().values
        data.loc[cluster_mask, 'mag_rolling_mean_30d'] = cluster_data['magnitudo'].rolling(window=window_30d, min_periods=1).mean().values
        data.loc[cluster_mask, 'depth_rolling_mean_7d'] = cluster_data['depth'].rolling(window=window_7d, min_periods=1).mean().values
        data.loc[cluster_mask, 'depth_rolling_mean_30d'] = cluster_data['depth'].rolling(window=window_30d, min_periods=1).mean().values
        
        # Calculate time since last earthquake in the same cluster
        time_diff = cluster_data['datetime'].diff().dt.total_seconds() / 3600  # in hours
        data.loc[cluster_mask, 'time_since_last_eq'] = time_diff.values
    
    return data

def create_interaction_features(data):
    print("Creating interaction features...")
    data['lat_long_interaction'] = data['latitude'] * data['longitude']
    data['mag_depth_interaction'] = data['magnitudo'] * data['depth']
    
    return data

def main():
    # Filepath to the original data
    filepath = 'data/raw/Eartquakes-1990-2023.csv'
    
    # Load the data
    data = load_data(filepath)
    
    # Filter the data to include only earthquakes and reviewed status
    data = filter_data(data, 'data_type', 'earthquake')
    data = filter_data(data, 'status', 'reviewed')
    
    # Drop unnecessary columns
    columns_to_drop = ['tsunami', 'significance', 'status', 'data_type']
    data = drop_columns(data, columns_to_drop)
    
    # Encode the 'state' column to integers
    data = encode_column(data, 'state')
    
    # Create new features
    data = create_datetime_features(data)
    data = create_spatial_features(data)
    data = remove_duplicates(data)
    data = create_rolling_features(data)
    data = create_interaction_features(data)
    
    # Drop original 'date' and 'time' columns
    data = drop_columns(data, ['date', 'time'])
    
    # Save the processed data to a new CSV file
    output_filepath = 'data/processed/filtered_processed_features.csv'
    save_data(data, output_filepath)

if __name__ == "__main__":
    main()

