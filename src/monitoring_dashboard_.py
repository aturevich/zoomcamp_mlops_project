import streamlit as st
import pandas as pd
import plotly.express as px
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from evidently.pipeline.column_mapping import ColumnMapping
from monitoring import get_recent_predictions, get_data_drift_results
import os

st.set_page_config(page_title="Earthquake Prediction Monitoring", layout="wide")

# Custom CSS for centered plots
st.markdown("""
<style>
.centered-plot {
    display: flex;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    recent_predictions = get_recent_predictions(n=2000)  # Get more predictions to split into reference and current
    drift_results = get_data_drift_results(n=1000)
    return recent_predictions, drift_results

recent_predictions, drift_results = load_data()

st.markdown("<h1 style='text-align: center;'>Earthquake Prediction Monitoring Dashboard</h1>", unsafe_allow_html=True)

# Display data info
st.markdown("<h2 style='text-align: center;'>Data Information</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Recent Predictions</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.write(f"Number of recent predictions: {len(recent_predictions)}")
    st.write(recent_predictions.info())
    st.write(recent_predictions.head())

# Data Drift Results
st.markdown("<h2 style='text-align: center;'>Data Drift Results</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.write(f"Number of drift results: {len(drift_results)}")
    st.dataframe(drift_results)

# Visualize drift scores over time
if not drift_results.empty:
    drift_results['timestamp'] = pd.to_datetime(drift_results['timestamp'])
    fig = px.line(drift_results, x='timestamp', y='drift_score', title='Drift Score Over Time')
    st.markdown('<div class="centered-plot">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Calculate and display the percentage of drift detected
    drift_percentage = (drift_results['drift_detected'].sum() / len(drift_results)) * 100
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.metric("Percentage of Drift Detected", f"{drift_percentage:.2f}%")

# Recent Predictions
st.markdown("<h2 style='text-align: center;'>Recent Predictions</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.dataframe(recent_predictions.head(100))  # Show only the most recent 100 predictions

st.markdown("<p style='text-align: center;'>This visualization shows recent earthquake predictions. Each point represents a predicted earthquake:</p>", unsafe_allow_html=True)
st.markdown("<ul style='text-align: center; list-style-position: inside;'><li>The size of the point represents the absolute depth (larger points are deeper).</li><li>The color represents the actual depth (including negative values if any).</li><li>The animation shows how predictions change over time.</li><li>Hover over points to see detailed information.</li></ul>", unsafe_allow_html=True)

# Visualize predictions
if not recent_predictions.empty:
    recent_predictions['timestamp'] = pd.to_datetime(recent_predictions['timestamp'])
    
    # Use absolute value of depth for size, but keep original depth for color
    recent_predictions['abs_depth'] = recent_predictions['depth'].abs()
    
    fig = px.scatter_geo(recent_predictions.head(100), 
                         lat='latitude', 
                         lon='longitude', 
                         color='depth',  # Use depth for color
                         size='abs_depth',  # Use absolute depth for size
                         hover_name='timestamp',
                         hover_data=['magnitude', 'depth'],  # Add these to hover info
                         animation_frame='timestamp',
                         projection='natural earth',
                         color_continuous_scale='Viridis',  # You can change this color scale
                         size_max=50)  # Adjust this value to change the maximum marker size

    fig.update_layout(height=600, title='Earthquake Predictions Visualization')
    st.markdown('<div class="centered-plot">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Evidently AI Dashboard
st.markdown("<h2 style='text-align: center;'>Evidently AI Dashboard</h2>", unsafe_allow_html=True)

if len(recent_predictions) > 200:  # Reduced the threshold to 200
    # Split the data into reference (older) and current (newer) datasets
    split_index = len(recent_predictions) // 2
    reference_data = recent_predictions.iloc[split_index:]
    current_data = recent_predictions.iloc[:split_index]

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.write(f"Reference data size: {len(reference_data)}")
        st.write(f"Current data size: {len(current_data)}")

    # Ensure both datasets have the same columns
    common_columns = list(set(current_data.columns) & set(reference_data.columns))
    current_data = current_data[common_columns]
    reference_data = reference_data[common_columns]

    # Print column names for debugging
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.write("Columns used for drift analysis:", common_columns)

    # Define column mapping
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = current_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Create and calculate dashboard
    evidently_dashboard = Dashboard(tabs=[DataDriftTab()])
    evidently_dashboard.calculate(reference_data, current_data, column_mapping=column_mapping)

    # Save dashboard to HTML and display in Streamlit
    evidently_dashboard.save("evidently_dashboard.html")
    
    # Center the Evidently dashboard
    col1, col2, col3 = st.columns([1,6,1])
    with col2:
        with open("evidently_dashboard.html", "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), width=None, height=1000, scrolling=True)
else:
    st.warning(f"Not enough data to generate Evidently AI dashboard. We need at least 200 predictions, but we only have {len(recent_predictions)}.")

# Display the latest data drift report if it exists
if os.path.exists("reports/data_drift_report.html"):
    st.markdown("<h2 style='text-align: center;'>Latest Data Drift Report</h2>", unsafe_allow_html=True)
    
    # Center the data drift report
    col1, col2, col3 = st.columns([1,6,1])
    with col2:
        with open("reports/data_drift_report.html", "r") as f:
            st.components.v1.html(f.read(), width=None, height=600, scrolling=True)
else:
    st.info("No recent data drift report available.")
