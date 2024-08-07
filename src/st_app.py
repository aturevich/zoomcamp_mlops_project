import streamlit as st
from prediction_app import main as prediction_main
from monitoring_dashboard import monitoring_dashboard

PAGES = {
    "Earthquake Prediction": prediction_main,
    "Monitoring Dashboard": monitoring_dashboard
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()
