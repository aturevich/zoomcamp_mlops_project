## Setup and Installation

1. Prerequisites: Python 3.10+, pipenv, Terraform
2. Clone the repository: `git clone <repository-url>`
3. Navigate to project directory: `cd earthquake_prediction`
4. Install dependencies: `make setup`

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

## Docker

Build image: `docker build -t earthquake-prediction .`
Run container: `docker run -p 8000:8000 earthquake-prediction`

## Contributing

See CONTRIBUTING.md for details on submitting pull requests.

## License

This project is under the MIT License. See LICENSE.md for details.

----
using earthquake data, training 2 models
# Data
https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023


ydata_profiling --title "Earthquake Data Profiling Report" data/raw/Eartquakes-1990-2023.csv earthquake_data_profile.html


prefect server start

prefect agent start -q "default"

python pipeline.py

prefect deployment run earthquake_data_pipeline/daily-schedule