import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor, Pool
import mlflow
import mlflow.catboost
from mlflow.models import infer_signature
from geopy.distance import geodesic
from prefect import task, flow

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("file:./mlruns")

class MLflowCallback:
    def __init__(self, metric_period=1):
        self.metric_period = metric_period

    def after_iteration(self, info):
        if info.iteration % self.metric_period == 0:
            for metric_name, metric_value in info.metrics["learn"].items():
                if isinstance(metric_value, list):
                    metric_value = metric_value[-1]  # Take the last value if it's a list
                mlflow.log_metric(f"train_{metric_name}", metric_value, step=info.iteration)
            
            if "validation" in info.metrics:
                for metric_name, metric_value in info.metrics["validation"].items():
                    if isinstance(metric_value, list):
                        metric_value = metric_value[-1]  # Take the last value if it's a list
                    mlflow.log_metric(f"val_{metric_name}", metric_value, step=info.iteration)
        return True

@task
def load_data(filepath, sample_size=None):
    logger.info(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)
    if sample_size:
        data = data.sample(n=sample_size, random_state=42)
    logger.info(f"Data loaded. Shape: {data.shape}")
    return data

@task
def preprocess_data(data):
    logger.info("Preprocessing data...")
    date_column = "date" if "date" in data.columns else "datetime" if "datetime" in data.columns else None
    if date_column is None:
        raise ValueError("No date or datetime column found in the dataset.")

    data[date_column] = pd.to_datetime(data[date_column], format='mixed', utc=True)
    data = data.sort_values(date_column)

    data["day_of_year"] = data[date_column].dt.dayofyear
    data["month"] = data[date_column].dt.month
    data["year"] = data[date_column].dt.year
    data["week_of_year"] = data[date_column].dt.isocalendar().week

    logger.info("Preprocessing completed successfully.")
    return data

@task
def prepare_location_features(data):
    logger.info("Preparing location features...")
    features = [
        "mag", "depth", "day_of_year", "month", "year", "week_of_year",
    ]
    X = data[features]
    y = data[["latitude", "longitude"]]
    return X, y

@task
def train_model(X_train, y_train, X_val=None, y_val=None):
    logger.info("Training CatBoost model...")

    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_val, y_val) if X_val is not None and y_val is not None else None

    logger.info(f"Train data shape: {X_train.shape}, {y_train.shape}")
    if X_val is not None and y_val is not None:
        logger.info(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    else:
        logger.info("No validation data provided.")

    if not mlflow.active_run():
        mlflow.start_run(run_name="location_model")

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiRMSE',
        verbose=100
    )

    mlflow_callback = MLflowCallback(metric_period=1)

    if eval_pool is not None:
        model.fit(train_pool, eval_set=eval_pool, use_best_model=True, callbacks=[mlflow_callback])
    else:
        model.fit(train_pool, callbacks=[mlflow_callback])

    if eval_pool is not None:
        val_metrics = model.eval_metrics(eval_pool, metrics=['MultiRMSE'], ntree_end=model.best_iteration_)
        mlflow.log_metric("final_val_MultiRMSE", val_metrics['MultiRMSE'][-1])
        logger.info(f"Validation MultiRMSE: {val_metrics['MultiRMSE'][-1]}")

    return model

@flow
def train_location_model(input_data_path, model_output_path, sample_size=None):
    logger.info("Starting location model training flow...")
    mlflow.set_experiment("Earthquake Location Prediction")
    
    with mlflow.start_run(run_name="location_model_parent") as parent_run:  # Start the parent run
        logger.info(f"MLflow Parent Run ID: {parent_run.info.run_id}")
        
        try:
            data = load_data(input_data_path, sample_size)
            data = preprocess_data(data)
            X, y_location = prepare_location_features(data)
            
            mlflow.log_param("input_rows", X.shape[0])
            mlflow.log_param("input_columns", X.shape[1])
            mlflow.log_param("features", ", ".join(X.columns))
            
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y_location, test_size=0.2, random_state=42)
            
            tscv = TimeSeriesSplit(n_splits=5)

            logger.info("Training and evaluating location model...")
            cv_metrics = cross_validate_model(X, y_location, tscv)

            # Log average metrics
            for metric, values in cv_metrics.items():
                avg_value = np.mean(values)
                std_value = np.std(values)
                mlflow.log_metric(f"{metric}_avg", avg_value)
                mlflow.log_metric(f"{metric}_std", std_value)
                logger.info(f"{metric}: {avg_value:.4f} (+/- {std_value:.4f})")

            # Train final model on all data
            final_model = train_model(X_train, y_train, X_val, y_val)

            # Evaluate final model
            final_metrics = evaluate_model(final_model, X_val, y_val)
            for metric, value in final_metrics.items():
                mlflow.log_metric(f"final_{metric}", value)
                logger.info(f"Final model {metric}: {value:.4f}")

            # Log feature importances
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_json("feature_importances.json")
            mlflow.log_artifact("feature_importances.json")
            logger.info(f"Top 5 important features: {', '.join(feature_importance['feature'].head().tolist())}")

            # Log the model
            mlflow.catboost.log_model(final_model, "location_model")

            # Save the model locally
            save_model(final_model, model_output_path)

            model_uri = mlflow.get_artifact_uri("location_model")
            logger.info(f"Logged model URI: {model_uri}")
            mlflow.log_param("model_output_path", model_output_path)  # Log model path as a parameter

        except Exception as e:
            logger.error(f"An error occurred during location model training: {e}")
            mlflow.log_param("error", str(e))
            raise

    logger.info("Location model training completed.")
    return model_output_path

@task
def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)

    # Assuming y_test is a DataFrame with columns for latitude and longitude
    lat_true, lon_true = y_test.iloc[:, 0], y_test.iloc[:, 1]
    lat_pred, lon_pred = y_pred[:, 0], y_pred[:, 1]

    # Calculate mean squared error (MSE) and mean absolute error (MAE) for each coordinate
    mse_lat = mean_squared_error(lat_true, lat_pred)
    mse_lon = mean_squared_error(lon_true, lon_pred)
    rmse_lat = np.sqrt(mse_lat)
    rmse_lon = np.sqrt(mse_lon)
    mae_lat = mean_absolute_error(lat_true, lat_pred)
    mae_lon = mean_absolute_error(lon_true, lon_pred)

    # Calculate mean geodesic distance error (in kilometers)
    distances = [geodesic((lat_true.iloc[i], lon_true.iloc[i]), (lat_pred[i], lon_pred[i])).kilometers for i in range(len(lat_true))]
    mean_distance_error = np.mean(distances)
    max_distance_error = np.max(distances)

    return {
        "mse_lat": mse_lat,
        "mse_lon": mse_lon,
        "rmse_lat": rmse_lat,
        "rmse_lon": rmse_lon,
        "mae_lat": mae_lat,
        "mae_lon": mae_lon,
        "mean_distance_error": mean_distance_error,
        "max_distance_error": max_distance_error
    }

@task
def save_model(model, filepath):
    logger.info(f"Saving model to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save_model(filepath)
    logger.info(f"Model saved to {filepath}")

@task
def cross_validate_model(X, y, tscv):
    logger.info("Cross-validating CatBoost model...")
    metrics = {
        "mse_lat": [], "mse_lon": [], "rmse_lat": [], "rmse_lon": [],
        "mae_lat": [], "mae_lon": [], "mean_distance_error": [], "max_distance_error": []
    }
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Start a nested run
        with mlflow.start_run(nested=True, run_name=f"cv_fold_{fold}"):
            model = train_model(X_train, y_train, X_test, y_test)
            fold_metrics = evaluate_model(model, X_test, y_test)
        
            for metric, value in fold_metrics.items():
                if metric in metrics:
                    metrics[metric].append(value)
                    mlflow.log_metric(f"fold_{metric}", value)
                else:
                    logger.warning(f"Unexpected metric '{metric}' encountered. Skipping...")

    return metrics

@flow
def train_location_model(input_data_path, model_output_path, sample_size=None):
    logger.info("Starting location model training flow...")
    mlflow.set_experiment("Earthquake Location Prediction")
    
    with mlflow.start_run(run_name="location_model") as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        try:
            data = load_data(input_data_path, sample_size)
            data = preprocess_data(data)
            X, y_location = prepare_location_features(data)
            
            mlflow.log_param("input_rows", X.shape[0])
            mlflow.log_param("input_columns", X.shape[1])
            mlflow.log_param("features", ", ".join(X.columns))
            
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y_location, test_size=0.2, random_state=42)
            
            tscv = TimeSeriesSplit(n_splits=5)

            logger.info("Training and evaluating location model...")
            cv_metrics = cross_validate_model(X, y_location, tscv)

            # Log average metrics
            for metric, values in cv_metrics.items():
                avg_value = np.mean(values)
                std_value = np.std(values)
                mlflow.log_metric(f"{metric}_avg", avg_value)
                mlflow.log_metric(f"{metric}_std", std_value)
                logger.info(f"{metric}: {avg_value:.4f} (+/- {std_value:.4f})")

            # Train final model on all data
            final_model = train_model(X_train, y_train, X_val, y_val)

            # Evaluate final model
            final_metrics = evaluate_model(final_model, X_val, y_val)
            for metric, value in final_metrics.items():
                mlflow.log_metric(f"final_{metric}", value)
                logger.info(f"Final model {metric}: {value:.4f}")

            # Log feature importances
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_json("feature_importances.json")
            mlflow.log_artifact("feature_importances.json")
            logger.info(f"Top 5 important features: {', '.join(feature_importance['feature'].head().tolist())}")

            # Log the model
            mlflow.catboost.log_model(final_model, "location_model")

            # Save the model locally
            save_model(final_model, model_output_path)

            model_uri = mlflow.get_artifact_uri("location_model")
            logger.info(f"Logged model URI: {model_uri}")
            mlflow.log_param("model_output_path", model_output_path)  # Log model path as a parameter

        except Exception as e:
            logger.error(f"An error occurred during location model training: {e}")
            mlflow.log_param("error", str(e))
            raise

    logger.info("Location model training completed.")
    return model_output_path

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    input_data_path = "data/processed/earthquake_data.csv"
    model_output_path = "models/location_prediction_model.cbm"

    try:
        train_location_model(input_data_path, model_output_path, sample_size=100000)  # Adjust sample size as needed
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}")
    finally:
        if mlflow.active_run():
            mlflow.end_run()

    logger.info("Script execution completed")
