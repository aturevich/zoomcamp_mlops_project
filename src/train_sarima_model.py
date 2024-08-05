import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
from prefect import task, flow
import logging
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from prefect.tasks import task_input_hash
from datetime import timedelta

# Custom MLflow callback
class MLflowCallback:
    def __init__(self, run_id=None):
        self.run_id = run_id or mlflow.active_run().info.run_id

    def log_metric(self, key, value, step=None):
        with mlflow.start_run(run_id=self.run_id, nested=True):
            mlflow.log_metric(key, value, step=step)

    def log_param(self, key, value):
        with mlflow.start_run(run_id=self.run_id, nested=True):
            mlflow.log_param(key, value)

    def log_artifact(self, local_path):
        with mlflow.start_run(run_id=self.run_id, nested=True):
            mlflow.log_artifact(local_path)

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def load_and_prepare_data(input_data_path, mlflow_callback):
    try:
        data = pd.read_csv(input_data_path)
        mlflow_callback.log_param("input_data_shape", data.shape)
        
        data['datetime'] = pd.to_datetime(data['datetime'], format='ISO8601')
        data = data.rename(columns={'mag': 'y'}) if 'mag' in data.columns else data
        data = data.groupby(data['datetime'].dt.date)['y'].mean().reset_index()
        data = data.set_index('datetime').sort_index().asfreq('D').interpolate()
        
        mlflow_callback.log_param("prepared_data_shape", data.shape)
        mlflow_callback.log_param("date_range", f"{data.index.min()} to {data.index.max()}")
        
        # Log data distribution curve
        plt.figure(figsize=(10, 6))
        data['y'].hist()
        plt.title('Data Distribution')
        plt.savefig('data_distribution.png')
        mlflow_callback.log_artifact('data_distribution.png')
        plt.close()

        logging.info(f"Data prepared. Shape: {data.shape}, Range: {data.index.min()} to {data.index.max()}")
        return data
    except Exception as e:
        logging.error(f"Error in load_and_prepare_data: {str(e)}")
        raise

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def analyze_data(data, mlflow_callback):
    result = adfuller(data['y'])
    adf_statistic, p_value = result[0], result[1]
    
    mlflow_callback.log_metric("adf_statistic", adf_statistic)
    mlflow_callback.log_metric("adf_p_value", p_value)
    
    # Log ACF and PACF plots
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    plt.figure(figsize=(12, 6))
    plot_acf(data['y'], ax=plt.gca())
    plt.savefig('acf_plot.png')
    mlflow_callback.log_artifact('acf_plot.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plot_pacf(data['y'], ax=plt.gca())
    plt.savefig('pacf_plot.png')
    mlflow_callback.log_artifact('pacf_plot.png')
    plt.close()

    logging.info(f"ADF Statistic: {adf_statistic}, p-value: {p_value}")
    return data

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def find_best_parameters(data, mlflow_callback):
    model = auto_arima(data['y'], seasonal=True, m=7,
                       start_p=1, start_q=1, max_p=3, max_q=3, max_d=2,
                       start_P=0, start_Q=0, max_P=2, max_Q=2, max_D=1,
                       trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    
    order = model.order
    seasonal_order = model.seasonal_order
    
    mlflow_callback.log_param("best_order", order)
    mlflow_callback.log_param("best_seasonal_order", seasonal_order)
    mlflow_callback.log_metric("auto_arima_aic", model.aic())
    
    logging.info(f"Best model: {order}, {seasonal_order}")
    return order, seasonal_order

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def train_and_evaluate_sarima_model(data, order, seasonal_order, mlflow_callback):
    model = SARIMAX(data['y'], order=order, seasonal_order=seasonal_order, 
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False, maxiter=500)
    
    mlflow_callback.log_metric("model_aic", model_fit.aic)
    mlflow_callback.log_metric("model_bic", model_fit.bic)
    
    # In-sample predictions
    in_sample_pred = model_fit.predict()
    in_sample_rmse = np.sqrt(mean_squared_error(data['y'], in_sample_pred))
    in_sample_mae = mean_absolute_error(data['y'], in_sample_pred)
    
    mlflow_callback.log_metric("in_sample_rmse", in_sample_rmse)
    mlflow_callback.log_metric("in_sample_mae", in_sample_mae)
    
    # Out-of-sample forecast
    forecast_steps = 30
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_series = pd.Series(forecast.predicted_mean, index=data.index[-forecast_steps:])
    
    # If we have actual data for the forecast period
    if len(data) >= forecast_steps:
        actual = data['y'][-forecast_steps:]
        out_sample_rmse = np.sqrt(mean_squared_error(actual, forecast_series))
        out_sample_mae = mean_absolute_error(actual, forecast_series)
        
        mlflow_callback.log_metric("out_sample_rmse", out_sample_rmse)
        mlflow_callback.log_metric("out_sample_mae", out_sample_mae)
    else:
        out_sample_rmse = np.nan
        out_sample_mae = np.nan

    # Log forecast plot
    plt.figure(figsize=(12, 6))
    data['y'].plot(label='Actual')
    in_sample_pred.plot(label='In-sample Prediction')
    forecast_series.plot(label='Forecast')
    plt.legend()
    plt.title('SARIMA Model Forecast')
    plt.savefig('forecast_plot.png')
    mlflow_callback.log_artifact('forecast_plot.png')
    plt.close()
    
    return model_fit, in_sample_rmse, in_sample_mae, out_sample_rmse, out_sample_mae

@flow
def train_sarima_model_flow(input_data_path: str, model_output_path: str):
    mlflow.set_experiment("SARIMA Model")
    with mlflow.start_run(run_name="sarima_model") as run:
        mlflow_callback = MLflowCallback(run.info.run_id)
        try:
            data = load_and_prepare_data(input_data_path, mlflow_callback)
            analyze_data(data, mlflow_callback)
            order, seasonal_order = find_best_parameters(data, mlflow_callback)
            
            model_fit, in_sample_rmse, in_sample_mae, out_sample_rmse, out_sample_mae = train_and_evaluate_sarima_model(data, order, seasonal_order, mlflow_callback)
            
            # Save the model
            model_fit.save(model_output_path)
            mlflow_callback.log_artifact(model_output_path)
            logging.info(f"Model saved to {model_output_path}")
            
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            mlflow_callback.log_param("error", str(e))
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_data_path = "data/processed/earthquake_data.csv"
    model_output_path = "models/sarima_model.pkl"
    train_sarima_model_flow(input_data_path, model_output_path)