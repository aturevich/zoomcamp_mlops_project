.PHONY: setup run test lint format clean api profile train monitor check create-sample-data pre-commit

# Set up the project
setup:
	pipenv install --dev

# Run the main pipeline
run:
	pipenv run python pipeline.py

# Create sample data
create-sample-data:
	pipenv run python create_sample_data.py

# Run tests
test: create-sample-data
	PYTHONPATH=.  pipenv run pytest --cov=src tests/

# Run linter
lint:
	pipenv run flake8 .

# Format code
format:
	pipenv run black .

# Clean up unnecessary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Start the API
api:
	pipenv run uvicorn src.api:app --host 0.0.0.0 --port 8000

# Run data profiling
profile:
	pipenv run python -c "from src.ds_profiling import data_profiling_pipeline; data_profiling_pipeline()"

# Run model training
train:
	pipenv run python -c "from src.model_training import model_training_pipeline; model_training_pipeline('data/processed', 'models/random_forest_model.joblib')"

# Run model monitoring
monitor:
	pipenv run python -c "from src.monitoring import monitoring_flow; monitoring_flow('data/processed/reference_data.csv', 'data/processed/current_data.csv', 'reports/model_monitoring_report.html')"

# All-in-one command to run lint, test, and format
check: pre-commit lint create-sample-data test format

# Run pre-commit checks
pre-commit:
	pipenv run pre-commit run --all-files