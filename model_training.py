# model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)
    print(f"Data loaded. Shape: {data.shape}")
    return data

def train_model(X_train, y_train, params):
    print("Training Random Forest model...")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    print("Model training completed.")
    return model

def evaluate_model(model, X_test, y_test):
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared Score: {r2}")
    return rmse, r2

def save_model(model, filepath):
    print(f"Saving model to {filepath}...")
    joblib.dump(model, filepath)
    print("Model saved.")

def main():
    # Set the experiment name
    mlflow.set_experiment("Earthquake Magnitude Prediction")

    # Load the training and test data
    train_data = load_data('data/processed/train_data.csv')
    test_data = load_data('data/processed/test_data.csv')

    # Separate features and target
    X_train = train_data.drop(columns=['magnitudo'])
    y_train = train_data['magnitudo']
    X_test = test_data.drop(columns=['magnitudo'])
    y_test = test_data['magnitudo']

    # Define model parameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    }

    # Start an MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)

        # Train the model
        model = train_model(X_train, y_train, params)

        # Evaluate the model
        rmse, r2 = evaluate_model(model, X_test, y_test)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mlflow.log_table(feature_importance, "feature_importance.json")

        # Save the model locally
        save_model(model, 'models/random_forest_model.joblib')

if __name__ == "__main__":
    main()