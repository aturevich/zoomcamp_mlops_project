import streamlit as st
import requests
from datetime import date

API_URL = "http://fastapi:8000"  # This will be the service name in docker-compose

st.title("Earthquake Prediction System")

prediction_type = st.radio(
    "What would you like to predict?",
    ("Earthquake Magnitude", "Potential Earthquake Locations"),
)

if prediction_type == "Earthquake Magnitude":
    st.subheader("Predict Earthquake Magnitude")

    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0)
    longitude = st.number_input(
        "Longitude", min_value=-180.0, max_value=180.0, value=0.0
    )
    prediction_date = st.date_input("Date", value=date.today())

    if st.button("Predict Magnitude"):
        input_data = {
            "latitude": latitude,
            "longitude": longitude,
            "date": prediction_date.isoformat(),
        }

        response = requests.post(f"{API_URL}/predict_magnitude", json=input_data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Magnitude: {result['predicted_magnitude']:.2f}")
        else:
            st.error("An error occurred during prediction. Please try again.")

else:
    st.subheader("Predict Potential Earthquake Locations")

    prediction_date = st.date_input("Date", value=date.today())

    if st.button("Predict Locations"):
        input_data = {"date": prediction_date.isoformat()}

        response = requests.post(f"{API_URL}/predict_locations", json=input_data)

        if response.status_code == 200:
            result = response.json()
            st.success("Predicted potential earthquake locations:")
            for location in result["predicted_locations"]:
                st.write(
                    f"Latitude: {location['latitude']}, Longitude: {location['longitude']}, Probability: {location['probability']:.2f}"
                )
        else:
            st.error("An error occurred during prediction. Please try again.")

# You can add more Streamlit components here, such as:
# - A map to visualize the predicted locations
# - Historical data visualization
# - Model performance metrics
