import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor, Pool
import mlflow
import mlflow.catboost
from mlflow.models import infer_signature
import joblib
from prefect import task, flow

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
def prepare_magnitude_features(data):
    logger.info("Preparing magnitude features...")
    features = [
        "latitude", "longitude", "depth", "day_of_year", "month", "year", "week_of_year",
    ]
    X = data[features]
    y_magnitude = data["mag"]
    return X, y_magnitude

@task
def train_model(X_train, y_train, X_val=None, y_val=None):
    logger.info("Training CatBoost model...")

    # Convert training and validation data to Pool format
    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_val, y_val) if X_val is not None and y_val is not None else None

    # Log the type and shape of the data
    logger.info(f"Train data shape: {X_train.shape}, {y_train.shape}")
    if X_val is not None and y_val is not None:
        logger.info(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    else:
        logger.info("No validation data provided.")

    # Ensure any previous MLflow run is ended before starting a new one
    if mlflow.active_run():
        mlflow.end_run()

    # Start a new MLflow run
    with mlflow.start_run():
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function='MultiRMSE',
            verbose=0
        )

        # Train model with or without eval_set based on availability
        model.fit(train_pool, eval_set=eval_pool, use_best_model=True if eval_pool else False)

        # Log final validation metrics if eval_pool is provided
        if eval_pool:
            logger.info("Evaluating metrics on the validation set...")
            val_metrics = model.eval_metrics(eval_pool, metrics=['MultiRMSE'], ntree_end=model.best_iteration_)
            mlflow.log_metric("final_val_MultiRMSE", val_metrics['MultiRMSE'][-1])

    return model

@task
def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    max_error = np.max(np.abs(y_test - y_pred))
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "max_error": max_error
    }

@task
def save_model(model, filepath):
    logger.info(f"Saving model to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save_model(filepath)
    logger.info(f"Model saved to {filepath}")

@task
def cross_validate_model(X, y, tscv):
    logger.info(f"Cross-validating CatBoost model...")
    metrics = {
        "mse": [], "rmse": [], "mae": [], "r2": [], "mape": [], "max_error": []
    }
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        with mlflow.start_run(nested=True, run_name=f"fold_{fold}"):
            model = train_model(X_train, y_train, X_test, y_test)
            fold_metrics = evaluate_model(model, X_test, y_test)
        
            for metric, value in fold_metrics.items():
                metrics[metric].append(value)
                mlflow.log_metric(f"fold_{metric}", value)

    return metrics

@flow
def train_magnitude_model(input_data_path, model_output_path, sample_size=None):
    logger.info("Starting magnitude model training flow...")
    mlflow.set_experiment("Earthquake Magnitude Prediction")
    
    tracking_uri = mlflow.get_tracking_uri()
    logger.info(f"MLflow Tracking URI: {tracking_uri}")
    
    experiment = mlflow.get_experiment_by_name("Earthquake Magnitude Prediction")
    logger.info(f"MLflow Experiment ID: {experiment.experiment_id}")
    
    with mlflow.start_run(run_name="magnitude_model_catboost") as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        try:
            data = load_data(input_data_path, sample_size)
            data = preprocess_data(data)
            X, y_magnitude = prepare_magnitude_features(data)
            
            mlflow.log_param("input_rows", X.shape[0])
            mlflow.log_param("input_columns", X.shape[1])
            mlflow.log_param("features", ", ".join(X.columns))
            
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y_magnitude, test_size=0.2, random_state=42)
            
            tscv = TimeSeriesSplit(n_splits=5)

            logger.info("Training and evaluating CatBoost model...")
            cv_metrics = cross_validate_model(X, y_magnitude, tscv)

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
            
            mlflow.log_table(feature_importance, "feature_importances.json")
            logger.info(f"Top 5 important features: {', '.join(feature_importance['feature'].head().tolist())}")

            # Log the model
            signature = infer_signature(X, y_magnitude)
            mlflow.catboost.log_model(
                cb_model=final_model,
                artifact_path="magnitude_model",
                signature=signature,
                input_example=X.iloc[:5]
            )

            model_uri = mlflow.get_artifact_uri("magnitude_model")
            logger.info(f"Logged model URI: {model_uri}")

            # Save the model locally
            save_model(final_model, model_output_path)

        except Exception as e:
            logger.error(f"An error occurred during magnitude model training: {e}")
            mlflow.log_param("error", str(e))
            raise

    logger.info("Magnitude model training completed.")
    return model_output_path

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    input_data_path = "data/processed/earthquake_data.csv"
    model_output_path = "models/magnitude_prediction_model.cbm"
    
    try:
        train_magnitude_model(input_data_path, model_output_path, sample_size=100000)  # Adjust sample size as needed
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}")
    finally:
        mlflow.end_run()
    
    logger.info("Script execution completed")
