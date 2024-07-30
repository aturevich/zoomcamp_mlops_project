import pandas as pd
import numpy as np
from src.model_training import full_model_training_pipeline
import os
import tempfile

def test_full_model_training_pipeline():
    # Create a temporary directory for test data and models
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample data
        data = pd.DataFrame({
            'date': pd.date_range(start='2021-01-01', periods=100),
            'latitude': np.random.uniform(30, 40, 100),
            'longitude': np.random.uniform(-120, -110, 100),
            'depth': np.random.uniform(0, 100, 100),
            'mag': np.random.uniform(2, 6, 100),
        })
        
        # Save sample data
        input_data_path = os.path.join(tmpdir, 'sample_data.csv')
        data.to_csv(input_data_path, index=False)
        
        # Define output paths
        magnitude_model_output_path = os.path.join(tmpdir, 'magnitude_model.joblib')
        location_model_output_path = os.path.join(tmpdir, 'location_model.joblib')
        
        # Run the pipeline
        magnitude_path, location_path = full_model_training_pipeline(
            input_data_path, 
            magnitude_model_output_path, 
            location_model_output_path
        )
        
        # Check if models were saved
        assert os.path.exists(magnitude_path)
        assert os.path.exists(location_path)
