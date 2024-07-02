using earthquake data, training 2 models
# Data
https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023


ydata_profiling --title "Earthquake Data Profiling Report" data/raw/Eartquakes-1990-2023.csv earthquake_data_profile.html


prefect server start

prefect agent start -q "default"

python pipeline.py

prefect deployment run earthquake_data_pipeline/daily-schedule