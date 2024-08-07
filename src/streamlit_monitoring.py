import streamlit as st
import pandas as pd
import plotly.express as px
from src.monitoring import get_recent_predictions, get_data_drift_results

def show_monitoring_page():
    st.title("Model Monitoring Dashboard")

    st.header("Recent Predictions")
    recent_predictions = get_recent_predictions(n=100)
    st.dataframe(recent_predictions)

    st.header("Prediction Distribution")
    fig_magnitude = px.histogram(recent_predictions, x="magnitude", nbins=20, title="Magnitude Distribution")
    st.plotly_chart(fig_magnitude)

    fig_depth = px.histogram(recent_predictions, x="depth", nbins=20, title="Depth Distribution")
    st.plotly_chart(fig_depth)

    st.header("Data Drift Results")
    drift_results = get_data_drift_results(n=100)
    st.dataframe(drift_results)

    st.header("Data Drift Over Time")
    fig_drift = px.line(drift_results, x="timestamp", y="drift_score", title="Data Drift Score Over Time")
    st.plotly_chart(fig_drift)

    st.header("Drift Detection Frequency")
    drift_counts = drift_results['drift_detected'].value_counts()
    fig_drift_pie = px.pie(values=drift_counts.values, names=drift_counts.index, title="Drift Detection Frequency")
    st.plotly_chart(fig_drift_pie)

if __name__ == "__main__":
    show_monitoring_page()
