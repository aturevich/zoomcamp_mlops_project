# src/monitoring.py
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, RegressionPerformanceTab
from evidently.pipeline.column_mapping import ColumnMapping
import pandas as pd
from prefect import task, flow


@task
def load_reference_data(filepath):
    return pd.read_csv(filepath)


@task
def load_current_data(filepath):
    return pd.read_csv(filepath)


@task
def generate_evidently_report(reference_data, current_data, output_path):
    column_mapping = ColumnMapping()
    column_mapping.target = "mag"  # Assuming 'mag' is your target column
    column_mapping.prediction = (
        "mag"  # For this test, we'll use the same column as both target and prediction
    )
    column_mapping.numerical_features = ["latitude", "longitude", "depth"]

    dashboard = Dashboard(tabs=[DataDriftTab(), RegressionPerformanceTab()])
    dashboard.calculate(reference_data, current_data, column_mapping=column_mapping)
    dashboard.save(output_path)


@flow
def monitoring_flow(reference_data_path, current_data_path, output_path):
    reference_data = load_reference_data(reference_data_path)
    current_data = load_current_data(current_data_path)
    generate_evidently_report(reference_data, current_data, output_path)


if __name__ == "__main__":
    monitoring_flow(
        "data/processed/reference_data.csv",
        "data/processed/current_data.csv",
        "reports/model_monitoring_report.html",
    )
