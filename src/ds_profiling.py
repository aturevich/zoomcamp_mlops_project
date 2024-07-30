import os
import pandas as pd
from ydata_profiling import ProfileReport
from prefect import task, flow
from prefect.artifacts import create_markdown_artifact
import logging
from src.utils import load_config


@task
def load_data(storage_basepath: str, file_path: str) -> pd.DataFrame:
    full_path = os.path.join(storage_basepath, file_path)
    return pd.read_csv(full_path)


@task
def generate_report(data: pd.DataFrame, report_title: str) -> ProfileReport:
    return ProfileReport(data, title=report_title)


@task
def save_report(report: ProfileReport, storage_basepath: str, file_path: str):
    full_path = os.path.join(storage_basepath, file_path)
    report.to_file(full_path)


@task
def log_artifact(storage_basepath: str, file_path: str):
    full_path = os.path.join(storage_basepath, file_path)
    with open(full_path, "r") as f:
        report_content = f.read()
    create_markdown_artifact(
        key="data_profiling_report",
        markdown=f"# Data Profiling Report\n\n{report_content}",
    )
    logging.info(f"Report saved and logged to Prefect artifact: {file_path}")


@flow(name="data_profiling_flow")
def data_profiling_pipeline(storage_basepath: str, input_file: str, output_file: str):
    data = load_data(storage_basepath, input_file).result()
    report = generate_report(data, "Earthquake Data Profiling Report").result()
    save_report(report, storage_basepath, output_file).result()
    log_artifact(storage_basepath, output_file).result()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    storage_basepath = config["storage"]["basepath"]
    data_profiling_pipeline(
        storage_basepath,
        "raw/Earthquakes-1990-2023.csv",
        "processed/earthquake_data_profile.html",
    )
