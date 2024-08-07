import os
from prefect import flow
import mlflow
from src.train_magnitude_model import train_magnitude_model

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("file:./mlruns")

@flow
def model_training_pipeline(input_data_path, output_dir):
    logger.info("Starting model training pipeline...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    magnitude_model_output_path = os.path.join(output_dir, "magnitude_prediction_model.cbm")
    
    try:
        logger.info("Training magnitude model...")
        magnitude_model_path = train_magnitude_model(input_data_path, magnitude_model_output_path)
        logger.info(f"Magnitude model saved to: {magnitude_model_path}")
        
        return magnitude_model_path
    
    except Exception as e:
        logger.error(f"An error occurred during the pipeline execution: {e}")
        raise

if __name__ == "__main__":
    input_data_path = "data/processed/earthquake_data.csv"
    output_dir = "models"
    
    try:
        model_training_pipeline(input_data_path, output_dir)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
    finally:
        if mlflow.active_run():
            mlflow.end_run()
    
    logger.info("Script execution completed")
