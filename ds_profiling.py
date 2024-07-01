import pandas as pd
from ydata_profiling import ProfileReport

def load_data(filepath):
    """Load the earthquake data from a CSV file."""
    return pd.read_csv(filepath)

def generate_report(data, report_title, output_file):
    """Generate a profiling report and save it to an HTML file."""
    profile = ProfileReport(data, title=report_title)
    profile.to_file(output_file)
    print(f"Report saved to {output_file}")

def main():
    # Filepath to the earthquake data
    filepath = 'data/raw/Eartquakes-1990-2023.csv'
    
    # Load the data
    data = load_data(filepath)
    
    # Generate and save the report
    report_title = "Earthquake Data Profiling Report"
    output_file = "earthquake_data_profile.html"
    generate_report(data, report_title, output_file)

if __name__ == "__main__":
    main()