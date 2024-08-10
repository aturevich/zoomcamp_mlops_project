import pandas as pd
import numpy as np
from src.train_magnitude_model import train_earthquake_model  # Update this import
import os
import tempfile


def test_train_earthquake_model():
    # Create a temporary directory for test data and models
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample data
        data = pd.DataFrame(
            {
                "date": pd.date_range(start="2021-01-01", periods=100),
                "latitude": np.random.uniform(30, 40, 100),
                "longitude": np.random.uniform(-120, -110, 100),
                "depth": np.random.uniform(0, 100, 100),
                "mag": np.random.uniform(2, 6, 100),
            }
        )

        # Save sample data
        input_data_path = os.path.join(tmpdir, "sample_data.csv")
        data.to_csv(input_data_path, index=False)

        # Define output path
        model_output_path = os.path.join(tmpdir, "models")

        # Run the model training
        result = train_earthquake_model(input_data_path, model_output_path)

        # Check if models were saved
        assert os.path.exists(result)

        # You might want to add more specific checks here, depending on what
        # train_earthquake_model returns and what files it creates
