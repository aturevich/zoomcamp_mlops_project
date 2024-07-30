# create_sample_data.py

import pandas as pd
import numpy as np
import os


def create_sample_earthquake_data(num_samples=100):
    data = {
        "date": pd.date_range(start="2020-01-01", periods=num_samples),
        "latitude": np.random.uniform(30, 40, num_samples),
        "longitude": np.random.uniform(-125, -115, num_samples),
        "depth": np.random.uniform(0, 100, num_samples),
        "mag": np.random.uniform(2, 7, num_samples),
        "magType": np.random.choice(["ml", "md", "mw"], num_samples),
        "nst": np.random.randint(1, 100, num_samples),
        "gap": np.random.uniform(0, 360, num_samples),
        "dmin": np.random.uniform(0, 5, num_samples),
        "rms": np.random.uniform(0, 2, num_samples),
        "net": np.random.choice(["us", "ci", "nc"], num_samples),
        "id": [f"eq{i:04d}" for i in range(num_samples)],
        "updated": pd.date_range(start="2020-01-02", periods=num_samples),
        "place": [f"Sample Place {i}" for i in range(num_samples)],
        "type": np.random.choice(
            ["earthquake", "quarry blast"], num_samples, p=[0.95, 0.05]
        ),
        "horizontalError": np.random.uniform(0, 10, num_samples),
        "depthError": np.random.uniform(0, 5, num_samples),
        "magError": np.random.uniform(0, 0.5, num_samples),
        "magNst": np.random.randint(1, 50, num_samples),
        "status": np.random.choice(["reviewed", "automatic"], num_samples),
        "locationSource": np.random.choice(["us", "ci", "nc"], num_samples),
        "magSource": np.random.choice(["us", "ci", "nc"], num_samples),
    }

    return pd.DataFrame(data)


def save_sample_data(
    data, directory="tests/sample_data", filename="sample_earthquake.csv"
):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    data.to_csv(filepath, index=False)
    print(f"Sample data saved to {filepath}")


if __name__ == "__main__":
    sample_data = create_sample_earthquake_data()
    save_sample_data(sample_data)
