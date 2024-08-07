import requests
import random
from datetime import datetime, timedelta
import pandas as pd

# API endpoint
API_URL = "http://localhost:8000/predict_earthquake"  # Update this with your actual API URL

def generate_inferred_predictions(n=1000):
    # Generate input data
    start_date = datetime.now() - timedelta(days=30)
    predictions = []

    for i in range(n):
        timestamp = start_date + timedelta(minutes=random.randint(1, 43200))
        latitude = random.uniform(-90, 90)
        longitude = random.uniform(-180, 180)
        
        # Prepare input data for API
        input_data = {
            "latitude": latitude,
            "longitude": longitude,
            "date": timestamp.isoformat()
        }

        try:
            # Make API request
            response = requests.post(API_URL, json=input_data)
            response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
            
            prediction = response.json()
            
            # Add input data and timestamp to the prediction
            prediction["latitude"] = latitude
            prediction["longitude"] = longitude
            prediction["timestamp"] = timestamp.isoformat()
            
            predictions.append(prediction)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} predictions")

        except requests.exceptions.RequestException as e:
            print(f"Error making API request: {e}")
            continue

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(predictions)
    
    print(f"Generated {len(df)} inferred predictions.")
    print(df.head())

    # You can save this DataFrame to a CSV file if needed
    df.to_csv("inferred_predictions.csv", index=False)
    print("Saved predictions to inferred_predictions.csv")

if __name__ == "__main__":
    generate_inferred_predictions(1000)  # Generate 1000 inferred predictions
