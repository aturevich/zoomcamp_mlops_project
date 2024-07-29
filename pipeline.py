import os
import yaml
from prefect import flow
from prefect.deployments import Deployment
from prefect.filesystems import LocalFileSystem
from prefect.client.schemas.schedules import IntervalSchedule
from datetime import timedelta
from src.ds_profiling import data_profiling_pipeline
from src.data_prep import data_preparation_pipeline
from src.model_training import model_training_pipeline
from src.monitoring import monitoring_flow
from src.utils import load_config

def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

@flow(name="main_pipeline")
def main_pipeline(config):
    storage_path = config['storage']['basepath']

    # Data profiling
    data_profiling_pipeline(
        storage_basepath=storage_path,
        input_file="raw/Earthquakes-1990-2023.csv",
        output_file="processed/earthquake_data_profile.html"
    )

    # Data preparation
    processed_data_path = data_preparation_pipeline(storage_basepath=storage_path)

    # Model training
    model_path = model_training_pipeline(
        input_data_path=processed_data_path,
        model_output_path=os.path.join(storage_path, "models/random_forest_model.joblib")
    )

    # Model monitoring
    monitoring_flow(
        reference_data_path=os.path.join(storage_path, "processed/reference_data.csv"),
        current_data_path=processed_data_path,
        output_path=os.path.join(storage_path, "reports/model_monitoring_report.html")
    )

    return model_path

def register_deployment():
    config = load_config()

    storage_path = config['storage']['basepath']
    if not os.path.exists(storage_path):
        raise FileNotFoundError(f"Storage path {storage_path} does not exist.")
    
    storage = LocalFileSystem(basepath=storage_path)
    storage.save("prefect-storage")

    schedule = IntervalSchedule(interval=timedelta(days=1))
    deployment = Deployment.build_from_flow(
        flow=main_pipeline,
        name="earthquake_prediction_pipeline",
        schedules=[schedule],
        storage=storage,
        parameters={"config": config},
        ignore_file=".prefectignore"
    )
    deployment.apply()

    print("Main pipeline 'earthquake_prediction_pipeline' was registered successfully.")

if __name__ == "__main__":
    register_deployment()