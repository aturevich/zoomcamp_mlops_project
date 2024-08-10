import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import mlflow
import mlflow.catboost
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
                    metric_value = metric_value[
                        -1
                    ]  # Take the last value if it's a list
                mlflow.log_metric(
                    f"train_{metric_name}", metric_value, step=info.iteration
                )

            if "validation" in info.metrics:
                for metric_name, metric_value in info.metrics["validation"].items():
                    if isinstance(metric_value, list):
                        metric_value = metric_value[
                            -1
                        ]  # Take the last value if it's a list
                    mlflow.log_metric(
                        f"val_{metric_name}", metric_value, step=info.iteration
                    )
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
    date_column = (
        "date"
        if "date" in data.columns
        else "datetime" if "datetime" in data.columns else None
    )
    if date_column is None:
        raise ValueError("No date or datetime column found in the dataset.")

    data[date_column] = pd.to_datetime(data[date_column], format="mixed", utc=True)
    data = data.sort_values(date_column)

    data["day_of_year"] = data[date_column].dt.dayofyear
    data["month"] = data[date_column].dt.month
    data["year"] = data[date_column].dt.year
    data["week_of_year"] = data[date_column].dt.isocalendar().week

    # Create a binary column for significant earthquakes (e.g., magnitude >= 5.0)
    data["significant_event"] = (data["mag"] >= 5.0).astype(int)

    logger.info("Preprocessing completed successfully.")
    return data


@task
def prepare_features(data):
    logger.info("Preparing features...")
    features = [
        "latitude",
        "longitude",
        "day_of_year",
        "month",
        "year",
        "week_of_year",
    ]
    X = data[features]
    y_magnitude = data["mag"]
    y_depth = data["depth"]
    y_significant = data["significant_event"]
    return X, y_magnitude, y_depth, y_significant


@task
def train_model(X_train, y_train, X_val=None, y_val=None, target="magnitude"):
    logger.info(f"Training CatBoost model for {target}...")

    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_val, y_val) if X_val is not None and y_val is not None else None

    if target in ["magnitude", "depth"]:
        model = CatBoostRegressor(
            iterations=1000, learning_rate=0.1, depth=6, loss_function="RMSE", verbose=0
        )
    else:  # For event occurrence
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function="Logloss",
            verbose=0,
        )

    model.fit(
        train_pool,
        eval_set=eval_pool,
        use_best_model=True if eval_pool else False,
        callbacks=[MLflowCallback()],
    )

    return model


@task
def evaluate_model(model, X_test, y_test, task="regression"):
    y_pred = model.predict(X_test)
    if task == "regression":
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }
    else:  # For classification
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


@flow
def train_earthquake_model(input_data_path, model_output_path, sample_size=None):
    logger.info("Starting earthquake model training flow...")
    mlflow.set_experiment("Earthquake Prediction")

    try:
        with mlflow.start_run(run_name="earthquake_model_catboost") as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")

            data = load_data(input_data_path, sample_size)
            data = preprocess_data(data)
            X, y_magnitude, y_depth, y_significant = prepare_features(data)

            mlflow.log_param("input_rows", X.shape[0])
            mlflow.log_param("input_columns", X.shape[1])
            mlflow.log_param("features", ", ".join(X.columns))

            (
                X_train,
                X_val,
                y_mag_train,
                y_mag_val,
                y_depth_train,
                y_depth_val,
                y_sig_train,
                y_sig_val,
            ) = train_test_split(
                X, y_magnitude, y_depth, y_significant, test_size=0.2, random_state=42
            )

            # Train magnitude model
            magnitude_model = train_model(
                X_train, y_mag_train, X_val, y_mag_val, target="magnitude"
            )
            mag_metrics = evaluate_model(
                magnitude_model, X_val, y_mag_val, task="regression"
            )
            for metric, value in mag_metrics.items():
                mlflow.log_metric(f"magnitude_{metric}", value)

            # Train depth model
            depth_model = train_model(
                X_train, y_depth_train, X_val, y_depth_val, target="depth"
            )
            depth_metrics = evaluate_model(
                depth_model, X_val, y_depth_val, task="regression"
            )
            for metric, value in depth_metrics.items():
                mlflow.log_metric(f"depth_{metric}", value)

            # Train significant event model
            significant_model = train_model(
                X_train, y_sig_train, X_val, y_sig_val, target="significant"
            )
            sig_metrics = evaluate_model(
                significant_model, X_val, y_sig_val, task="classification"
            )
            for metric, value in sig_metrics.items():
                mlflow.log_metric(f"significant_{metric}", value)

            # Save models
            os.makedirs(model_output_path, exist_ok=True)
            magnitude_model.save_model(
                os.path.join(model_output_path, "magnitude_model.cbm")
            )
            depth_model.save_model(os.path.join(model_output_path, "depth_model.cbm"))
            significant_model.save_model(
                os.path.join(model_output_path, "significant_model.cbm")
            )

            # Log models to MLflow
            mlflow.catboost.log_model(magnitude_model, "magnitude_model")
            mlflow.catboost.log_model(depth_model, "depth_model")
            mlflow.catboost.log_model(significant_model, "significant_model")

            logger.info("Models trained and saved successfully.")

            # Log feature importances
            for model_name, model in [
                ("magnitude", magnitude_model),
                ("depth", depth_model),
                ("significant", significant_model),
            ]:
                feature_importance = pd.DataFrame(
                    {"feature": X.columns, "importance": model.feature_importances_}
                ).sort_values("importance", ascending=False)

                mlflow.log_table(
                    feature_importance, f"{model_name}_feature_importances.json"
                )
                logger.info(
                    f"Top 5 important features for {model_name}: {', '.join(feature_importance['feature'].head().tolist())}"
                )

    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        mlflow.log_param("error", str(e))
        raise

    logger.info("Earthquake model training completed.")
    return model_output_path


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    input_data_path = "data/processed/earthquake_data.csv"
    model_output_path = "models"

    try:
        train_earthquake_model(
            input_data_path, model_output_path, sample_size=100000
        )  # Adjust sample size as needed
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}")
    finally:
        if mlflow.active_run():
            mlflow.end_run()

    logger.info("Script execution completed")
