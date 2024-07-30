import pandas as pd
import numpy as np
from src.data_prep import preprocess_data

def test_preprocess_data():
    # Create a sample DataFrame
    data = pd.DataFrame({
        'date': pd.date_range(start='2021-01-01', periods=5),
        'latitude': [1.0, 2.0, 3.0, 4.0, 5.0],
        'longitude': [1.0, 2.0, 3.0, 4.0, 5.0],
        'depth': [10.0, 20.0, 30.0, 40.0, 50.0],
        'mag': [3.0, 4.0, 5.0, 4.0, 3.0],
    })
    
    processed_data = preprocess_data(data)
    
    assert 'day_of_year' in processed_data.columns
    assert 'month' in processed_data.columns
    assert 'year' in processed_data.columns
    assert 'week_of_year' in processed_data.columns
    assert 'mag_lag_1' in processed_data.columns
    assert 'mag_lag_7' in processed_data.columns
    assert 'mag_lag_30' in processed_data.columns
    assert 'mag_rolling_mean_7d' in processed_data.columns
    assert 'mag_rolling_mean_30d' in processed_data.columns
