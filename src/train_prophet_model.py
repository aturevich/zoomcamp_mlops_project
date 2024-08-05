import os
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import mlflow
import joblib
from prefect import task, flow

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("file:./mlruns")

@task
def load_data(filepath):
    logger.info(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)
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

    # Remove timezone information
    data[date_column] = data[date_column].dt.tz_localize(None)

    logger.info("Preprocessing completed successfully.")
    return data

@task
def prepare_prophet_data(data):
    logger.info("Preparing data for Prophet...")
    prophet_data = data[['datetime', 'mag']].rename(columns={'datetime': 'ds', 'mag': 'y'})
    return prophet_data

@task
def train_prophet_model(data):
    logger.info("Training Prophet model...")
    model = Prophet()
    model.fit(data)

    # Log model parameters
    for param_name, param_value in model.params.items():
        mlflow.log_param(param_name, param_value)

    return model

@task
def evaluate_prophet_model(model, data):
    logger.info("Evaluating Prophet model...")
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
    df_p = performance_metrics(df_cv)
    
    # Log evaluation metrics
    for metric_name in df_p.columns:
        mlflow.log_metric(metric_name, df_p[metric_name].mean())
    
    return df_p

@task
def save_model(model, filepath):
    logger.info(f"Saving model to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")

@flow
def train_prophet_model_flow(input_data_path, model_output_path):
    logger.info("Starting Prophet model training flow...")
    mlflow.set_experiment("Earthquake Prophet Prediction")
    
    with mlflow.start_run(run_name="prophet_model"):
        try:
            data = load_data(input_data_path)
            data = preprocess_data(data)
            prophet_data = prepare_prophet_data(data)
            
            # Log data info
            mlflow.log_param("data_shape", str(prophet_data.shape))
            
            model = train_prophet_model(prophet_data)
            
            # Evaluate model and log metrics
            df_p = evaluate_prophet_model(model, prophet_data)
            
            rmse = df_p['rmse'].mean()
            mae = df_p['mae'].mean()
            logger.info(f"Prophet model - RMSE: {rmse}, MAE: {mae}")
            
            # Save the model locally
            save_model(model, model_output_path)
            
            # Log the model file
            mlflow.log_artifact(model_output_path, "prophet_model")
            mlflow.pyfunc.log_model("prophet_model", python_model=model)
            
        except Exception as e:
            logger.error(f"An error occurred during Prophet model training: {e}")
            raise

    logger.info("Prophet model training completed.")
    return model_output_path

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    input_data_path = "data/processed/earthquake_data.csv"
    model_output_path = "models/prophet_model.joblib"
    
    try:
        train_prophet_model_flow(input_data_path, model_output_path)
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}")
    finally:
        mlflow.end_run()
    
    logger.info("Script execution completed")

