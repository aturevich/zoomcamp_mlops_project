import streamlit as st
import requests
from datetime import date
import folium
from streamlit_folium import st_folium

API_URL = "http://localhost:8000"  # Update this with your API URL

st.title("Earthquake Prediction System")

# Information about the dataset used
st.sidebar.info("This prediction system uses USGS Earthquake Data from 1990-2023.")

# Initialize session state
if 'latitude' not in st.session_state:
    st.session_state.latitude = 39.8283
if 'longitude' not in st.session_state:
    st.session_state.longitude = -98.5795

# Function to create map
def create_map():
    m = folium.Map(location=[st.session_state.latitude, st.session_state.longitude], zoom_start=4)
    folium.Marker(
        [st.session_state.latitude, st.session_state.longitude],
        popup="Selected Location",
        tooltip="Selected Location"
    ).add_to(m)
    return m

# Callback function for map clicks
def map_click_callback(change):
    st.session_state.latitude = change['lat']
    st.session_state.longitude = change['lng']

# Display the map and capture clicks
st.write("Click on the map to select a location:")
map_output = st_folium(create_map(), width=700, height=450, key="map")

# Update coordinates if a location was clicked
if map_output['last_clicked']:
    map_click_callback(map_output['last_clicked'])
    st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

# Display selected coordinates
st.header("Selected Location")
st.write(f"Latitude: {st.session_state.latitude:.4f}")
st.write(f"Longitude: {st.session_state.longitude:.4f}")

# Date input
prediction_date = st.date_input("Select date for prediction", value=date.today())

# Toggle for automatic prediction
auto_predict = st.toggle("Enable automatic prediction", value=True)

# Function to make prediction
def make_prediction():
    input_data = {
        "latitude": st.session_state.latitude,
        "longitude": st.session_state.longitude,
        "date": prediction_date.isoformat()
    }
    try:
        response = requests.post(f"{API_URL}/predict_earthquake", json=input_data)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        
        st.success("Prediction successful!")
        st.write(f"Predicted Magnitude: {result['predicted_magnitude']:.2f}")
        st.write(f"Predicted Depth: {result['predicted_depth']:.2f} km")
        st.write(f"Probability of Significant Earthquake: {result['significant_probability']:.2%}")
        
        # Interpret the significance probability
        if result['significant_probability'] > 0.7:
            st.warning("High probability of a significant earthquake!")
        elif result['significant_probability'] > 0.3:
            st.info("Moderate probability of a significant earthquake.")
        else:
            st.info("Low probability of a significant earthquake.")
    except requests.exceptions.RequestException as e:
        error_message = f"An error occurred during prediction: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            error_message += f"\nResponse status code: {e.response.status_code}"
            if hasattr(e.response, 'text'):
                error_message += f"\nResponse content: {e.response.text}"
        st.error(error_message)

# Prediction logic
if auto_predict:
    make_prediction()
else:
    if st.button("Predict Earthquake"):
        make_prediction()

st.sidebar.info("""
    This is a demo of the Earthquake Prediction System. 
    
    How to use:
    1. Click on the map to select a location.
    2. Select a date for the prediction.
    3. Toggle automatic prediction on/off.
    4. If automatic prediction is off, click 'Predict Earthquake' to get results.
    
    The system predicts the magnitude and depth of a potential earthquake, 
    as well as the probability of it being a significant event (magnitude >= 5.0).
""")
