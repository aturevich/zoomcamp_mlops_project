import os
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
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
from contextlib import contextmanager

@contextmanager
def mlflow_run(run_name=None, nested=False):
    try:
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            yield run
    finally:
        mlflow.end_run()

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")

# End any active runs from previous executions
mlflow.end_run()

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
    print("Preprocessing data...")
    date_column = "date" if "date" in data.columns else "datetime" if "datetime" in data.columns else None
    if date_column is None:
        raise ValueError("No date or datetime column found in the dataset.")

    if data[date_column].dtype != "datetime64[ns]":
        data[date_column] = pd.to_datetime(data[date_column], format="ISO8601")

    data = data.sort_values(date_column)

    data["day_of_year"] = data[date_column].dt.dayofyear
    data["month"] = data[date_column].dt.month
    data["year"] = data[date_column].dt.year
    data["week_of_year"] = data[date_column].dt.isocalendar().week

    return data

@task
def prepare_magnitude_features(data):
    print("Preparing magnitude features...")
    features = [
        "latitude",
        "longitude",
        "depth",
        "day_of_year",
        "month",
        "year",
        "week_of_year",
    ]

    X = data[features]
    y_magnitude = data["mag"]

    return X, y_magnitude

@task
def prepare_location_features(data):
    print("Preparing location features...")
    features = ["day_of_year", "month", "year", "week_of_year"]
    X = data[features]
    y = data[["latitude", "longitude"]]
    return X, y

@task
def train_location_model(X_train, y_train):
    print("Training location model...")
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42)
    )
    model.fit(X_train, y_train)
    return model

@task
def evaluate_location_model(model, X_test, y_test):
    print("Evaluating location model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred, multioutput="uniform_average")
    return np.sqrt(mse), r2

@flow
def train_location_model_flow(input_data_path, model_output_path):
    print("Starting location model training flow...")
    mlflow.set_experiment("Earthquake Location Prediction")

    try:
        with mlflow.start_run(run_name="location_model") as run:
            print(f"Location model MLflow run ID: {run.info.run_id}")
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

            # Log the model
            mlflow.sklearn.log_model(final_model, "location_model")

            # Save the model locally
            save_model(final_model, model_output_path)

    except Exception as e:
        print(f"An error occurred in location model training: {e}")
        raise
    finally:
        print("Ended location model MLflow run")
    return model_output_path

@flow
def train_magnitude_model(input_data_path, model_output_path):
    print("Starting magnitude model training flow...")
    mlflow.set_experiment("Earthquake Magnitude Prediction")
    try:
        with mlflow.start_run(run_name="magnitude_model") as run:
            print(f"Magnitude model MLflow run ID: {run.info.run_id}")
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

                    mlflow.log_metric(f"{model_type}_avg_rmse", avg_rmse)
                    mlflow.log_metric(f"{model_type}_avg_r2", avg_r2)

                    if avg_rmse < best_rmse:
                        best_rmse = avg_rmse
                        best_model_type = model_type

            print(f"Best model: {best_model_type} with RMSE: {best_rmse}")

            # Train final model on all data
            final_model = train_model(X, y_magnitude, best_model_type)

            # Log the best model
            mlflow.sklearn.log_model(final_model, "best_magnitude_model")
            mlflow.log_metric("best_model_rmse", best_rmse)
            mlflow.log_param("best_model_type", best_model_type)

            # Save the model locally
            save_model(final_model, model_output_path)

    except Exception as e:
        print(f"An error occurred in magnitude model training: {e}")
        raise
    finally:
        print("Ended magnitude model MLflow run")

    return model_output_path

@task
def train_model(X_train, y_train, model_type="rf"):
    print(f"Training {model_type} model...")
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
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

@task
def save_model(model, filepath):
    print(f"Saving model to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

@task
def cross_validate_model(X, y, tscv, model_type):
    print(f"Cross-validating {model_type} model...")
    rmse_scores = []
    r2_scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if model_type == "location":
            model = train_location_model(X_train, y_train)
        else:
            model = train_model(X_train, y_train, model_type)

        if model_type == "location":
            rmse, r2 = evaluate_location_model(model, X_test, y_test)
        else:
            rmse, r2 = evaluate_model(model, X_test, y_test)

        rmse_scores.append(rmse)
        r2_scores.append(r2)

    return rmse_scores, r2_scores

@flow
def train_prophet_flow(input_data_path, model_output_path):
    print("Starting Prophet model training flow...")
    mlflow.set_experiment("Earthquake Prophet Prediction")

    try:
        with mlflow.start_run(run_name="prophet_model") as run:
            print(f"Prophet model MLflow run ID: {run.info.run_id}")
            data = load_data(input_data_path)
            data = preprocess_data(data)
            
            # Prepare data for Prophet (it expects columns 'ds' for date and 'y' for target)
            prophet_data = data[['datetime', 'mag']].rename(columns={'datetime': 'ds', 'mag': 'y'})
            
            # Log data info
            mlflow.log_param("data_shape", str(prophet_data.shape))
            
            # Define and fit the model
            model = Prophet()
            model.fit(prophet_data)
            
            # Log model parameters
            for param, value in model.params.items():
                mlflow.log_param(str(param), str(value))
            
            # Perform cross-validation
            df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
            df_p = performance_metrics(df_cv)
            
            rmse = df_p['rmse'].mean()
            mae = df_p['mae'].mean()
            
            print(f"Prophet model - RMSE: {rmse}, MAE: {mae}")
            
            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            
            # Save the model locally
            joblib.dump(model, model_output_path)
            
            # Log the model file
            mlflow.log_artifact(model_output_path, "prophet_model")
            
            # Log the model using MLflow's built-in support for sklearn models
            # (Prophet models can be treated as sklearn models for MLflow logging purposes)
            mlflow.sklearn.log_model(model, "prophet_model")
    
    except Exception as e:
        print(f"An error occurred in Prophet model training: {e}")
        raise
    finally:
        print("Ended Prophet model MLflow run")
        
    return model_output_path

@flow
def full_model_training_pipeline(input_data_path, magnitude_model_output_path, location_model_output_path, prophet_model_output_path):
    print("Starting full model training pipeline...")
    try:
        with mlflow.start_run(run_name="full_pipeline") as run:
            print(f"Full pipeline MLflow run ID: {run.info.run_id}")
            magnitude_model_path = train_magnitude_model(input_data_path, magnitude_model_output_path)
            location_model_path = train_location_model_flow(input_data_path, location_model_output_path)
            prophet_model_path = train_prophet_flow(input_data_path, prophet_model_output_path)
        
        return magnitude_model_path, location_model_path, prophet_model_path
    except Exception as e:
        print(f"An error occurred in the full pipeline: {e}")
        raise
    finally:
        print("Ended full pipeline MLflow run")

if __name__ == "__main__":
    mlflow.end_run()  # Ensure any lingering runs are ended
    
    input_data_path = "data/processed/earthquake_data.csv"
    magnitude_model_output_path = "models/magnitude_prediction_model.joblib"
    location_model_output_path = "models/location_prediction_model.joblib"
    prophet_model_output_path = "models/prophet_model.joblib"
    
    try:
        full_model_training_pipeline(
            input_data_path, magnitude_model_output_path, location_model_output_path, prophet_model_output_path
        )
    finally:
        mlflow.end_run()  # Ensure the run is ended even if an exception occurs
    
    print("Script execution completed")
