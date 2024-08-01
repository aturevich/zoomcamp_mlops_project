# Earthquake Prediction System

This project aims to predict earthquake magnitudes using machine learning techniques, implemented with MLOps best practices.

## Setup and Installation

1. Prerequisites: Python 3.10+, pipenv, Terraform
2. Clone the repository: `git clone <repository-url>`
3. Navigate to project directory: `cd earthquake_prediction`
4. Install dependencies: `make setup`

## Project Structure

- `data/`: Contains raw and processed data
- `src/`: Source code for data processing, model training, and API
- `models/`: Saved model artifacts
- `tests/`: Unit and integration tests
- `terraform/`: Infrastructure as Code configurations

## Data Processing and Model Training

1. Data Profiling:
```ydata_profiling --title "Earthquake Data Profiling Report" data/raw/Earthquakes-1990-2023.csv earthquake_data_profile.html```

2. Start Prefect server:
```prefect server start```

3. In a new terminal, start Prefect agent:
```prefect agent start -q "default"```

4. Run the main pipeline:
```pipenv shell```
```prefect agent start -q "default```

5. Execute the Prefect deployment:
```prefect deployment run earthquake_data_pipeline/daily-schedule```

6. View MLflow experiments:
```mlflow ui```

Access the MLflow UI at `http://localhost:5000`

## Model Deployment

Use Docker to build and run the prediction service:

```docker-compose up --build```

- FastAPI service: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`

## Usage

Use these make commands to run different parts of the project:

- `make run`: Execute main pipeline
- `make test`: Run tests
- `make lint`: Run linter
- `make format`: Format code
- `make api`: Start API
- `make check`: Run linter, tests, formatter, and Terraform plan

## Infrastructure as Code

Terraform is used for managing cloud infrastructure:

1. Update `terraform/terraform.tfvars` with your GCP project ID
2. Initialize Terraform: `make tf-init`
3. Plan changes: `make tf-plan`
4. Apply changes: `make tf-apply`
5. Destroy infrastructure: `make tf-destroy`

TODO: GCP Setup
- [ ] Create GCP project
- [ ] Enable necessary APIs
- [ ] Create service account and download key
- [ ] Configure Terraform with service account

## Contributing

See CONTRIBUTING.md for details on submitting pull requests.

## License

This project is under the MIT License. See LICENSE.md for details.

## Data Source

Earthquake data from [Kaggle](https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023)
