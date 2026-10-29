import pandas as pd
import pytest

from src.calibration import calibration_table, expected_calibration_error
from src.exports import dataframe_to_safe_csv_bytes, student_result_to_frame
from types import SimpleNamespace


def test_perfectly_calibrated_has_zero_error():
    # 10 students at p=0.7, exactly 7 pass
    table = calibration_table([1] * 7 + [0] * 3, [0.7] * 10)
    assert len(table) == 1 and table.loc[0, "Students"] == 10
    assert table.loc[0, "Observed pass rate"] == pytest.approx(0.7)
    assert expected_calibration_error(table) == pytest.approx(0.0)


def test_overconfident_model_has_positive_error_and_p_equal_one_is_counted():
    table = calibration_table([1, 0, 0, 1], [1.0, 0.9, 0.1, 0.0])
    assert table["Students"].sum() == 4
    assert expected_calibration_error(table) > 0


def test_empty_bins_dropped_and_nan_ignored():
    table = calibration_table([1, 0, 1], [0.9, float("nan"), 0.95])
    assert table["Students"].sum() == 2 and len(table) == 1


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        calibration_table([1, 0], [0.5])


def test_student_result_csv_contains_inputs_and_prediction():
    res = SimpleNamespace(label="Fail", p_pass=0.0261, risk={"level": "High Risk"}, score=36.14)
    frame = student_result_to_frame({"Age": 21, "Gender": "Male"}, res, "=bad")
    text = dataframe_to_safe_csv_bytes(frame).decode("utf-8-sig")
    assert "Student_ID,Age,Gender,Prediction,P_Pass,Risk,Performance_Score" in text
    assert "'=bad" in text and "High Risk" in text and "0.0261" in text
