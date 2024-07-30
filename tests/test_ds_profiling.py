import pandas as pd
import os
from src.ds_profiling import load_data, generate_report


def test_load_data():
    data = load_data("tests/sample_data", "sample_earthquake.csv")
    assert isinstance(data, pd.DataFrame)
    assert not data.empty


def test_generate_report(tmp_path):
    data = load_data("tests/sample_data", "sample_earthquake.csv")
    report = generate_report(data, "Test Report")
    assert report is not None
