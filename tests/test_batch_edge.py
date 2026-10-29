"""Batch CSV processing: invalid uploads, empty data, speed and consistency with single prediction."""
import io
import time

import numpy as np
import pandas as pd
import pytest

from src.batch import (BatchInputError, normalize_columns, process_batch_dataframe, summarize_batch,
                       PREDICTION_COLUMNS, REJECTED_COLUMNS)
from src.config import FEATURES, SAMPLE_PROFILES
from src.exports import dataframe_to_safe_csv_bytes
from src.prediction import load_bundle, p_pass_for


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


def frame(*profiles):
    return pd.DataFrame([{**p} for p in profiles], columns=FEATURES)


def test_missing_values_are_rejected_per_row_not_crash(bundle):
    bad = frame(*[SAMPLE_PROFILES["Average student"]] * 3).astype(object)
    bad.loc[0, "Attendance"] = np.nan
    bad.loc[1, "Gender"] = pd.NA
    bad.loc[2, "Study_Hours"] = float("inf")
    predictions, rejected = process_batch_dataframe(bad, bundle)
    assert predictions.empty and len(rejected) == 3
    assert list(predictions.columns) == PREDICTION_COLUMNS and list(rejected.columns) == REJECTED_COLUMNS


def test_all_rows_invalid_returns_empty_predictions_with_columns(bundle):
    predictions, rejected = process_batch_dataframe(frame({**SAMPLE_PROFILES["Average student"], "Age": 3}), bundle)
    assert predictions.empty and len(rejected) == 1


def test_duplicate_feature_columns_are_rejected_clearly(bundle):
    dup = pd.concat([frame(SAMPLE_PROFILES["Average student"]), pd.DataFrame({"Age": [30]})], axis=1)
    with pytest.raises(BatchInputError, match="more than once"):
        process_batch_dataframe(dup, bundle)


def test_excel_bom_and_header_spaces_are_normalised(bundle):
    csv = "\ufeff" + frame(SAMPLE_PROFILES["Average student"]).rename(columns={"Gender": " Gender "}).to_csv(index=False)
    predictions, rejected = process_batch_dataframe(pd.read_csv(io.StringIO(csv)), bundle)
    assert len(predictions) == 1 and rejected.empty
    assert "Gender" in normalize_columns(pd.DataFrame({"\ufeffGender ": [1]})).columns


def test_extra_columns_are_ignored_and_not_echoed(bundle):
    f = frame(SAMPLE_PROFILES["Average student"])
    f["Notes"] = ["=HYPERLINK(\"http://evil\")"]
    predictions, _ = process_batch_dataframe(f, bundle)
    assert "Notes" not in predictions.columns


def test_formula_text_in_category_is_rejected_and_export_stays_safe(bundle):
    f = frame(SAMPLE_PROFILES["Average student"]).astype(object)
    f.loc[0, "Gender"] = "=cmd|' /C calc'!A0"
    predictions, rejected = process_batch_dataframe(f, bundle)
    assert predictions.empty
    exported = dataframe_to_safe_csv_bytes(rejected).decode("utf-8-sig")
    assert "=cmd" not in exported.replace("'=cmd", "")  # every occurrence is neutralised


def test_batch_matches_single_prediction_and_is_reproducible(bundle):
    profiles = list(SAMPLE_PROFILES.values())
    first, _ = process_batch_dataframe(frame(*profiles), bundle)
    second, _ = process_batch_dataframe(frame(*profiles), bundle)
    expected = [round(p_pass_for(bundle, p), 4) for p in profiles]
    assert first["P(pass)"].tolist() == expected
    pd.testing.assert_frame_equal(first, second)


def test_five_thousand_rows_complete_quickly(bundle):
    rows = pd.concat([frame(*SAMPLE_PROFILES.values())] * 1667, ignore_index=True).iloc[:5000]
    start = time.time()
    predictions, rejected = process_batch_dataframe(rows, bundle)
    assert len(predictions) == 5000 and rejected.empty
    assert time.time() - start < 15


def test_non_dataframe_input_is_rejected(bundle):
    with pytest.raises(BatchInputError):
        process_batch_dataframe("not a table", bundle)


def test_summary_uses_only_real_predictions(bundle):
    predictions, _ = process_batch_dataframe(frame(*SAMPLE_PROFILES.values()), bundle)
    s = summarize_batch(predictions)
    assert s["n"] == 3 and sum(s["risk_counts"].values()) == 3
    assert 0 <= s["mean_p_pass"] <= 1
    assert summarize_batch(predictions.iloc[0:0])["n"] == 0
