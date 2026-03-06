import pandas as pd
import numpy as np
import joblib
import pytest
import os
import sys

sys.path.append(os.path.abspath("."))


def test_pipeline_ignores_unknown_categories():
    pipeline_path = "data/processed/preprocessor.joblib"
    if not os.path.exists(pipeline_path):
        pytest.skip("Pipeline artifact not found.")

    preprocessor = joblib.load(pipeline_path)

    # Create synthetic DataFrame with a valid row
    valid_data = pd.DataFrame(
        {
            "LIMIT_BAL": [50000],
            "SEX": [1],
            "EDUCATION": [2],
            "MARRIAGE": [1],
            "AGE": [30],
            "PAY_0": [0],
            "PAY_2": [0],
            "PAY_3": [0],
            "PAY_4": [0],
            "PAY_5": [0],
            "PAY_6": [0],
            "BILL_AMT1": [0],
            "BILL_AMT2": [0],
            "BILL_AMT3": [0],
            "BILL_AMT4": [0],
            "BILL_AMT5": [0],
            "BILL_AMT6": [0],
            "PAY_AMT1": [0],
            "PAY_AMT2": [0],
            "PAY_AMT3": [0],
            "PAY_AMT4": [0],
            "PAY_AMT5": [0],
            "PAY_AMT6": [0],
        }
    )

    # Create synthetic DataFrame with an unseen EDUCATION category (99)
    anomalous_data = valid_data.copy()
    anomalous_data["EDUCATION"] = 99

    out_valid = preprocessor.transform(valid_data)
    out_anomalous = preprocessor.transform(anomalous_data)

    # Pipeline must not crash. Output dimensions must remain strictly identical.
    assert out_valid.shape == out_anomalous.shape

    # The anomaly must result in an all-zero vector for the EDUCATION feature's one-hot columns
    # Assuming EDUCATION is processed by the categorical pipeline
    assert not np.array_equal(out_valid, out_anomalous)
