import streamlit as st
import requests
from datetime import date
import folium
from streamlit_folium import folium_static
import pandas as pd

API_URL = "http://localhost:8000"  # Update this if your API is hosted elsewhere

st.set_page_config(page_title="Earthquake Prediction System", layout="wide")
st.title("Earthquake Prediction System")

prediction_type = st.sidebar.radio(
    "What would you like to predict?",
    ("Earthquake Magnitude", "Potential Earthquake Locations")
)

if prediction_type == "Earthquake Magnitude":
    st.header("Predict Earthquake Magnitude")

    col1, col2, col3 = st.columns(3)
    with col1:
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0)
    with col2:
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0)
    with col3:
        prediction_date = st.date_input("Date", value=date.today())

    if st.button("Predict Magnitude"):
        input_data = {
            "latitude": latitude,
            "longitude": longitude,
            "date": prediction_date.isoformat()
        }

        response = requests.post(f"{API_URL}/predict_magnitude", json=input_data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Magnitude: {result['predicted_magnitude']:.2f}")
        else:
            st.error("An error occurred during prediction. Please try again.")

else:
    st.header("Predict Potential Earthquake Locations")

    prediction_date = st.date_input("Date", value=date.today())

    if st.button("Predict Locations"):
        input_data = {"date": prediction_date.isoformat()}

        response = requests.post(f"{API_URL}/predict_locations", json=input_data)

        if response.status_code == 200:
            result = response.json()
            st.success("Predicted potential earthquake locations:")

            # Create a map centered on the first predicted location
            m = folium.Map(location=[result['predicted_locations'][0]['latitude'], 
                                     result['predicted_locations'][0]['longitude']], 
                           zoom_start=4)

            # Add markers for each predicted location
            for location in result['predicted_locations']:
                folium.Marker(
                    [location['latitude'], location['longitude']],
                    popup=f"Probability: {location['probability']:.2f}"
                ).add_to(m)

            # Display the map
            folium_static(m)

            # Display the data in a table
            df = pd.DataFrame(result['predicted_locations'])
            st.dataframe(df)
        else:
            st.error("An error occurred during prediction. Please try again.")

# Add a section for model performance metrics
st.header("Model Performance Metrics")
st.write("Here you can add visualizations of your model's performance metrics.")
# Add your performance metric visualizations here

# Add a section for data drift monitoring
st.header("Data Drift Monitoring")
st.write("Here you can add visualizations of your data drift monitoring results.")
# Add your data drift monitoring visualizations here
