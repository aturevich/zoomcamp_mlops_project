import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="Earthquake Prediction", layout="wide")


def main():
    st.markdown(
        "<h1 style='text-align: center;'>Earthquake Prediction</h1>",
        unsafe_allow_html=True,
    )

    # Input form
    st.markdown(
        "<h2 style='text-align: center;'>Enter Prediction Details</h2>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        latitude = st.number_input(
            "Latitude", min_value=-90.0, max_value=90.0, value=0.0
        )
        longitude = st.number_input(
            "Longitude", min_value=-180.0, max_value=180.0, value=0.0
        )
        date = st.date_input("Date")
        time = st.time_input("Time")

        if st.button("Predict"):
            # Combine date and time
            datetime_str = f"{date} {time}"
            datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

            # Prepare the data for the API request
            data = {
                "latitude": latitude,
                "longitude": longitude,
                "date": datetime_obj.isoformat(),
            }

            # Make the API request
            API_URL = "http://localhost:8000/predict_earthquake"  # Update this with your actual API URL
            try:
                response = requests.post(API_URL, json=data)
                response.raise_for_status()
                result = response.json()

                # Display the results
                st.markdown(
                    "<h2 style='text-align: center;'>Prediction Results</h2>",
                    unsafe_allow_html=True,
                )
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric(
                        "Predicted Magnitude", f"{result['predicted_magnitude']:.2f}"
                    )
                    st.metric("Predicted Depth", f"{result['predicted_depth']:.2f} km")
                    st.metric(
                        "Probability of Significant Earthquake",
                        f"{result['significant_probability']:.2%}",
                    )

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred while making the prediction: {str(e)}")

    # Add some information about the prediction model
    st.markdown("---")
    st.markdown(
        "<h3 style='text-align: center;'>About the Prediction Model</h3>",
        unsafe_allow_html=True,
    )
    st.write(
        """
    This earthquake prediction model uses machine learning algorithms trained on historical earthquake data. 
    It takes into account factors such as geographical location and time to predict the magnitude and depth of potential earthquakes, 
    as well as the probability of a significant seismic event.

    Please note that earthquake prediction is a complex field with inherent uncertainties. 
    This model provides estimates based on patterns in historical data, but cannot guarantee the occurrence or non-occurrence of seismic events.
    """
    )


if __name__ == "__main__":
    main()
