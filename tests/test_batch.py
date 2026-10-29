import pandas as pd
import pytest

from src.batch import BatchInputError, process_batch_dataframe
from src.config import FEATURES, SAMPLE_PROFILES
from src.prediction import load_bundle


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


def frame_for(*profiles):
    return pd.DataFrame([{**profile} for profile in profiles], columns=FEATURES)


def test_batch_processes_valid_rows_and_keeps_physical_csv_row_numbers(bundle):
    predictions, rejected = process_batch_dataframe(frame_for(SAMPLE_PROFILES["Strong student"], SAMPLE_PROFILES["At-risk student"]), bundle)
    assert len(predictions) == 2
    assert predictions["CSV row"].tolist() == [2, 3]
    assert set(predictions["Predicted result"]) <= {"Pass", "Fail"}
    assert predictions["P(pass)"].between(0, 1).all()
    assert rejected.empty


def test_batch_rejects_invalid_rows_without_sending_them_to_model(bundle):
    bad = {**SAMPLE_PROFILES["Average student"], "Attendance": 140}
    predictions, rejected = process_batch_dataframe(frame_for(SAMPLE_PROFILES["Strong student"], bad), bundle)
    assert len(predictions) == 1
    assert len(rejected) == 1
    assert rejected.iloc[0]["CSV row"] == 3
    assert "Attendance" in rejected.iloc[0]["Errors"]


def test_batch_rejects_missing_columns(bundle):
    with pytest.raises(BatchInputError, match="Missing required column"):
        process_batch_dataframe(pd.DataFrame([{ "Age": 20 }]), bundle)


def test_batch_rejects_empty_and_oversized_files(bundle):
    with pytest.raises(BatchInputError, match="no data rows"):
        process_batch_dataframe(pd.DataFrame(columns=FEATURES), bundle)
    one = frame_for(SAMPLE_PROFILES["Average student"])
    with pytest.raises(BatchInputError, match="limits each upload"):
        process_batch_dataframe(pd.concat([one, one], ignore_index=True), bundle, max_rows=1)


def test_batch_trims_categories_and_preserves_warnings(bundle):
    student = {**SAMPLE_PROFILES["Average student"], "Gender": " Male ", "Study_Hours": 0.2}
    predictions, rejected = process_batch_dataframe(frame_for(student), bundle)
    assert rejected.empty
    assert len(predictions) == 1
    assert predictions.iloc[0]["Gender"] == "Male"
    assert "unusual" in predictions.iloc[0]["Validation warnings"]


def test_safe_csv_export_neutralizes_spreadsheet_formulas():
    from src.exports import dataframe_to_safe_csv_bytes

    frame = pd.DataFrame({"student": ["=1+1", "+cmd", "  =HYPERLINK(\"x\")", "normal"], "score": [10, -2, 3, 4]})
    exported = dataframe_to_safe_csv_bytes(frame).decode("utf-8-sig")
    assert "'=1+1" in exported
    assert "'+cmd" in exported
    assert "'  =HYPERLINK" in exported
    # Numeric negative values remain numeric rather than being escaped as text.
    assert "\n'+cmd,-2\n" in exported
