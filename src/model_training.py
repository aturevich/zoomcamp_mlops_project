import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib
import mlflow
import mlflow.sklearn
from prefect import task, flow


@task
def load_data(filepath):
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)
    print(f"Data loaded. Shape: {data.shape}")
    print(f"Columns in the dataset: {data.columns.tolist()}")
    print(data.head())
    return data


@task
def preprocess_data(data):
    # Check if 'date' column exists, if not, use 'datetime' column if it exists
    date_column = (
        "date"
        if "date" in data.columns
        else "datetime" if "datetime" in data.columns else None
    )

    if date_column is None:
        raise ValueError("No date or datetime column found in the dataset.")

    # Convert to datetime if it's not already
    if data[date_column].dtype != "datetime64[ns]":
        data[date_column] = pd.to_datetime(data[date_column], format="ISO8601")

    # Sort data by date
    data = data.sort_values(date_column)

    # Extract features from date
    data["day_of_year"] = data[date_column].dt.dayofyear
    data["month"] = data[date_column].dt.month
    data["year"] = data[date_column].dt.year
    data["week_of_year"] = data[date_column].dt.isocalendar().week

    # Ensure 'mag' column exists
    mag_column = (
        "mag"
        if "mag" in data.columns
        else "magnitudo" if "magnitudo" in data.columns else None
    )
    if mag_column is None:
        raise ValueError("No magnitude column found in the dataset.")

    # Create lag features
    for lag in [1, 7, 30]:  # 1 day, 1 week, 1 month
        data[f"mag_lag_{lag}"] = data.groupby(["latitude", "longitude"])[
            mag_column
        ].shift(lag)
        data[f"depth_lag_{lag}"] = data.groupby(["latitude", "longitude"])[
            "depth"
        ].shift(lag)

    # Create rolling window features
    for window in [7, 30]:  # 1 week, 1 month
        rolling_mag = (
            data.groupby(["latitude", "longitude"])[mag_column]
            .rolling(window=window)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        rolling_depth = (
            data.groupby(["latitude", "longitude"])["depth"]
            .rolling(window=window)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        data[f"mag_rolling_mean_{window}d"] = rolling_mag
        data[f"depth_rolling_mean_{window}d"] = rolling_depth

    # Drop rows with NaN values created by lag and rolling features
    data = data.dropna()

    return data


@task
def prepare_magnitude_features(data):
    features = [
        "latitude",
        "longitude",
        "depth",
        "day_of_year",
        "month",
        "year",
        "week_of_year",
        "mag_lag_1",
        "mag_lag_7",
        "mag_lag_30",
        "depth_lag_1",
        "depth_lag_7",
        "depth_lag_30",
        "mag_rolling_mean_7d",
        "mag_rolling_mean_30d",
        "depth_rolling_mean_7d",
        "depth_rolling_mean_30d",
    ]

    X = data[features]
    y_magnitude = data["mag"] if "mag" in data.columns else data["magnitudo"]

    return X, y_magnitude


@task
def prepare_location_features(data):
    features = ["day_of_year", "month", "year", "week_of_year"]
    X = data[features]
    y = data[["latitude", "longitude"]]
    return X, y


@task
def train_location_model(X_train, y_train):
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42)
    )
    model.fit(X_train, y_train)
    return model


@task
def evaluate_location_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred, multioutput="uniform_average")
    return np.sqrt(mse), r2


@flow
def train_location_model_flow(input_data_path, model_output_path):
    data = load_data(input_data_path)
    data = preprocess_data(data)
    X, y_location = prepare_location_features(data)
    
    tscv = TimeSeriesSplit(n_splits=5)

    rmse_scores, r2_scores = cross_validate_model(X, y_location, tscv, 'location')

    avg_rmse = np.mean(rmse_scores)
    avg_r2 = np.mean(r2_scores)

    print(f"Location model - Average RMSE: {avg_rmse}, Average R2: {avg_r2}")

    mlflow.log_metric("avg_rmse", avg_rmse)
    mlflow.log_metric("avg_r2", avg_r2)

    # Train final model on all data
    final_model = train_location_model(X, y_location)

    save_model(final_model, model_output_path)
    mlflow.sklearn.log_model(final_model, "best_location_model")

    return model_output_path


@flow
def train_magnitude_model(input_data_path, model_output_path):
    mlflow.set_experiment("Earthquake Magnitude Prediction")

    data = load_data(input_data_path)
    data = preprocess_data(data)
    X, y_magnitude = prepare_magnitude_features(data)
    
    tscv = TimeSeriesSplit(n_splits=5)

    best_model = None
    best_rmse = float('inf')
    best_model_type = None

    for model_type in ['rf', 'xgb']:
        with mlflow.start_run(run_name=f"{model_type}_model", nested=True):
            rmse_scores, r2_scores = cross_validate_model(X, y_magnitude, tscv, model_type)

            avg_rmse = np.mean(rmse_scores)
            avg_r2 = np.mean(r2_scores)

            print(f"{model_type} model - Average RMSE: {avg_rmse}, Average R2: {avg_r2}")

            mlflow.log_metric("avg_rmse", avg_rmse)
            mlflow.log_metric("avg_r2", avg_r2)

            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_model_type = model_type

    print(f"Best model: {best_model_type} with RMSE: {best_rmse}")

    # Train final model on all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    final_model = train_model(X_scaled, y_magnitude, best_model_type)

    save_model(final_model, model_output_path)
    mlflow.sklearn.log_model(final_model, "best_magnitude_model")

    return model_output_path

@task
def train_model(X_train, y_train, model_type="rf"):
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "xgb":
        model = XGBRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)
    return model


@task
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2


@task
def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


@task
def cross_validate_model(X, y, tscv, model_type):
    rmse_scores = []
    r2_scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_type == "location":
            model = train_location_model(X_train_scaled, y_train)
        else:
            model = train_model(X_train_scaled, y_train, model_type)

        if model_type == "location":
            rmse, r2 = evaluate_location_model(model, X_test_scaled, y_test)
        else:
            rmse, r2 = evaluate_model(model, X_test_scaled, y_test)

        rmse_scores.append(rmse)
        r2_scores.append(r2)

    return rmse_scores, r2_scores


# Rename this function
@task
def prepare_magnitude_features(data):
    features = [
        "latitude",
        "longitude",
        "depth",
        "day_of_year",
        "month",
        "year",
        "week_of_year",
        "mag_lag_1",
        "mag_lag_7",
        "mag_lag_30",
        "depth_lag_1",
        "depth_lag_7",
        "depth_lag_30",
        "mag_rolling_mean_7d",
        "mag_rolling_mean_30d",
        "depth_rolling_mean_7d",
        "depth_rolling_mean_30d",
    ]

    X = data[features]
    y_magnitude = data["mag"] if "mag" in data.columns else data["magnitudo"]

    return X, y_magnitude

@flow
def full_model_training_pipeline(input_data_path, magnitude_model_output_path, location_model_output_path):
    with mlflow.start_run(run_name="full_pipeline") as run:
        magnitude_model_path = train_magnitude_model(input_data_path, magnitude_model_output_path)
        
        # Ensure the previous run is ended
        mlflow.end_run()
        
        # Start a new run for location model
        with mlflow.start_run(run_name="location_model", nested=True) as nested_run:
            location_model_path = train_location_model_flow(input_data_path, location_model_output_path)
    
    return magnitude_model_path, location_model_path


if __name__ == "__main__":
    input_data_path = "data/processed/earthquake_data.csv"
    magnitude_model_output_path = "models/magnitude_prediction_model.joblib"
    location_model_output_path = "models/location_prediction_model.joblib"
    full_model_training_pipeline(
        input_data_path, magnitude_model_output_path, location_model_output_path
    )
