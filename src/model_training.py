import os
from prefect import flow
import mlflow
from src.train_location_model import train_location_model
from src.train_magnitude_model import train_magnitude_model
from src.train_prophet_model import train_prophet_model_flow

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")

@flow
def full_model_training_pipeline(input_data_path, output_dir):
    logger.info("Starting full model training pipeline...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths for each model
    magnitude_model_output_path = os.path.join(output_dir, "magnitude_prediction_model.joblib")
    location_model_output_path = os.path.join(output_dir, "location_prediction_model.joblib")
    prophet_model_output_path = os.path.join(output_dir, "prophet_model.joblib")
    
    try:
        # Train magnitude model
        logger.info("Training magnitude model...")
        magnitude_model_path = train_magnitude_model(input_data_path, magnitude_model_output_path)
        logger.info(f"Magnitude model saved to: {magnitude_model_path}")
        
        # Train location model
        logger.info("Training location model...")
        location_model_path = train_location_model(input_data_path, location_model_output_path)
        logger.info(f"Location model saved to: {location_model_path}")
        
        # Train Prophet model
        logger.info("Training Prophet model...")
        prophet_model_path = train_prophet_model_flow(input_data_path, prophet_model_output_path)
        logger.info(f"Prophet model saved to: {prophet_model_path}")
        
        logger.info("All models trained successfully.")
        return magnitude_model_path, location_model_path, prophet_model_path
    
    except Exception as e:
        logger.error(f"An error occurred during the full pipeline execution: {e}")
        raise

if __name__ == "__main__":
    input_data_path = "data/processed/earthquake_data.csv"
    output_dir = "models"
    
    try:
        full_model_training_pipeline(input_data_path, output_dir)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
    finally:
        mlflow.end_run()  # Ensure any active MLflow runs are properly closed
    
    logger.info("Script execution completed")
