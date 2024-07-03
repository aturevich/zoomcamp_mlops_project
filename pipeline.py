import os
import yaml
from prefect.deployments import Deployment
from prefect.filesystems import LocalFileSystem
from prefect.client.schemas.schedules import IntervalSchedule
from datetime import timedelta
from ds_profiling import data_profiling_pipeline
from data_prep import data_preparation_pipeline
from utils import load_config

def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def register_deployment():
    config = load_config()

    # Dynamically set up LocalFileSystem storage using config file
    storage_path = config['storage']['basepath']
    if not os.path.exists(storage_path):
        raise FileNotFoundError(f"Storage path {storage_path} does not exist.")
    
    storage = LocalFileSystem(basepath=storage_path)
    storage.save("prefect-storage")

    # Schedule the data profiling flow
    profiling_schedule = IntervalSchedule(interval=timedelta(days=1))
    profiling_deployment = Deployment.build_from_flow(
        flow=data_profiling_pipeline,
        name="data_profiling_schedule",
        schedules=[profiling_schedule],  # Use schedules list
        storage=storage,
        parameters={
            "storage_basepath": storage_path,
            "input_file": os.path.join(storage_path, "raw/Earthquakes-1990-2023.csv"),
            "output_file": os.path.join(storage_path, "processed/earthquake_data_profile.html")
        },
        ignore_file=".prefectignore"
    )
    profiling_deployment.apply()

    # Schedule the data preparation flow
    preparation_schedule = IntervalSchedule(interval=timedelta(days=1))
    preparation_deployment = Deployment.build_from_flow(
        flow=data_preparation_pipeline,
        name="data_preparation_schedule",
        schedules=[preparation_schedule],  # Use schedules list
        storage=storage,
        parameters={
            "storage_basepath": storage_path
        },
        ignore_file=".prefectignore"
    )
    preparation_deployment.apply()

    print("Flows 'data_profiling_pipeline' and 'data_preparation_pipeline' were registered successfully.")

if __name__ == "__main__":
    register_deployment()
