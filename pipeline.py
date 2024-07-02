from prefect import task, flow
from prefect_dask.task_runners import DaskTaskRunner
from prefect.deployments import Deployment
from prefect.client.schemas.schedules import IntervalSchedule
from datetime import timedelta
import subprocess

@task
def run_ds_profiling():
    """Run the data profiling script."""
    subprocess.run(["python", "ds_profiling.py"])

@task
def run_data_prep():
    """Run the data preparation script."""
    subprocess.run(["python", "data_prep.py"])

@task
def run_model_training():
    """Run the model training script."""
    subprocess.run(["python", "model_training.py"])

@flow(name="earthquake_data_pipeline", task_runner=DaskTaskRunner())
def earthquake_data_pipeline():
    profiling_task = run_ds_profiling()
    data_prep_task = run_data_prep()
    model_training_task = run_model_training()

# Function to register the deployment
def register_deployment():
    # Schedule the flow to run daily
    schedule = IntervalSchedule(interval=timedelta(days=1))

    # Create a deployment with the schedules list
    deployment = Deployment.build_from_flow(
        flow=earthquake_data_pipeline,
        name="daily-schedule",
        schedules=[schedule]
    )

    # Register the deployment
    deployment.apply()

    # Print confirmation
    print("Flow 'earthquake_data_pipeline' with deployment 'daily-schedule' was registered successfully.")

if __name__ == "__main__":
    register_deployment()
