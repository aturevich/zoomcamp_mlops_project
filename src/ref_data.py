import pandas as pd
from sklearn.model_selection import train_test_split
import os

REFERENCE_DATA_PATH = "data/reference_data.csv"

def create_reference_data(full_data_path, test_size=0.2, random_state=42):
    """
    Create reference data from the full dataset.
    
    :param full_data_path: Path to the full dataset CSV file
    :param test_size: Proportion of data to use as reference (default 0.2)
    :param random_state: Random state for reproducibility
    """

    full_data = pd.read_csv(full_data_path)

    features = ['latitude', 'longitude', 'day_of_year', 'month', 'year', 'week_of_year', 'mag', 'depth']

    reference_data, _ = train_test_split(full_data[features], test_size=1-test_size, random_state=random_state)

    os.makedirs(os.path.dirname(REFERENCE_DATA_PATH), exist_ok=True)

    reference_data.to_csv(REFERENCE_DATA_PATH, index=False)

    print(f"Reference data saved to '{REFERENCE_DATA_PATH}'. Shape: {reference_data.shape}")

def load_reference_data():
    """
    Load the reference data.
    
    :return: pandas DataFrame containing reference data
    """
    if not os.path.exists(REFERENCE_DATA_PATH):
        raise FileNotFoundError(f"Reference data file not found at {REFERENCE_DATA_PATH}. "
                                f"Please run create_reference_data() to generate it.")
    
    return pd.read_csv(REFERENCE_DATA_PATH)

if __name__ == "__main__":
    full_data_path = "data/processed/earthquake_data.csv"  
    create_reference_data(full_data_path)
