import pandas as pd
import numpy as np
from src.model_training import (
    preprocess_data,
    prepare_magnitude_features,
    prepare_location_features,
    train_model,
    train_location_model,
    evaluate_model,
    evaluate_location_model,
)

def test_prepare_magnitude_features():
    # Create a sample DataFrame with all required columns
    data = pd.DataFrame({
        'latitude': [1.0, 2.0, 3.0],
        'longitude': [4.0, 5.0, 6.0],
        'depth': [10.0, 20.0, 30.0],
        'mag': [3.0, 4.0, 5.0],
        'datetime': pd.date_range(start='2021-01-01', periods=3),
        'day_of_year': [1, 2, 3],
        'month': [1, 1, 1],
        'year': [2021, 2021, 2021],
        'week_of_year': [1, 1, 1],
        'mag_lag_1': [np.nan, 3.0, 4.0],
        'mag_lag_7': [np.nan, np.nan, np.nan],
        'mag_lag_30': [np.nan, np.nan, np.nan],
        'depth_lag_1': [np.nan, 10.0, 20.0],
        'depth_lag_7': [np.nan, np.nan, np.nan],
        'depth_lag_30': [np.nan, np.nan, np.nan],
        'mag_rolling_mean_7d': [3.0, 3.5, 4.0],
        'mag_rolling_mean_30d': [3.0, 3.5, 4.0],
        'depth_rolling_mean_7d': [10.0, 15.0, 20.0],
        'depth_rolling_mean_30d': [10.0, 15.0, 20.0],
    })
    
    X, y = prepare_magnitude_features(data)
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert 'latitude' in X.columns
    assert 'longitude' in X.columns
    assert 'depth' in X.columns
    assert 'day_of_year' in X.columns
    assert 'month' in X.columns
    assert 'year' in X.columns
    assert 'week_of_year' in X.columns
    assert 'mag_lag_1' in X.columns
    assert 'mag_lag_7' in X.columns
    assert 'mag_lag_30' in X.columns
    assert 'depth_lag_1' in X.columns
    assert 'depth_lag_7' in X.columns
    assert 'depth_lag_30' in X.columns
    assert 'mag_rolling_mean_7d' in X.columns
    assert 'mag_rolling_mean_30d' in X.columns
    assert 'depth_rolling_mean_7d' in X.columns
    assert 'depth_rolling_mean_30d' in X.columns
    assert y.name == 'mag'

def test_prepare_location_features():
    # Create a sample DataFrame
    data = pd.DataFrame({
        'latitude': [1.0, 2.0, 3.0],
        'longitude': [4.0, 5.0, 6.0],
        'datetime': pd.date_range(start='2021-01-01', periods=3),
    })

    data = preprocess_data(data)
    
    X, y = prepare_location_features(data)
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)
    assert 'day_of_year' in X.columns
    assert 'month' in X.columns
    assert 'year' in X.columns
    assert 'week_of_year' in X.columns
    assert list(y.columns) == ['latitude', 'longitude']
    assert X.shape[0] == 3  # Number of rows should match input
    assert y.shape[0] == 3  # Number of rows should match input
    assert X.shape[1] == 4  # Should have 4 feature columns
    assert y.shape[1] == 2  # Should have latitude and longitude

def test_train_magnitude_model():
    X = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
    })
    y = pd.Series([1.0, 2.0, 3.0])
    
    model = train_model(X, y, model_type='rf')
    assert hasattr(model, 'predict')
    
    model = train_model(X, y, model_type='xgb')
    assert hasattr(model, 'predict')

def test_train_location_model():
    X = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
    })
    y = pd.DataFrame({
        'latitude': [1.0, 2.0, 3.0],
        'longitude': [4.0, 5.0, 6.0],
    })
    
    model = train_location_model(X, y)
    assert hasattr(model, 'predict')

def test_evaluate_magnitude_model():
    X = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
    })
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.1, 2.1, 3.1])
    
    rmse, r2 = evaluate_model(y_true, y_pred)
    assert isinstance(rmse, float)
    assert isinstance(r2, float)

def test_evaluate_location_model():
    y_true = pd.DataFrame({
        'latitude': [1.0, 2.0, 3.0],
        'longitude': [4.0, 5.0, 6.0],
    })
    y_pred = pd.DataFrame({
        'latitude': [1.1, 2.1, 3.1],
        'longitude': [4.1, 5.1, 6.1],
    })
    
    rmse, r2 = evaluate_location_model(y_true, y_pred)
    assert isinstance(rmse, float)
    assert isinstance(r2, float)
