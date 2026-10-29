"""Input validation edge cases: nothing invalid may reach the model."""
import math

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES, SAMPLE_PROFILES
from src.validation import validate_student_input

OK = SAMPLE_PROFILES["Average student"]


@pytest.mark.parametrize("bad", [np.nan, pd.NA, None, "", "   ", float("inf"), -float("inf")])
def test_missing_or_non_finite_numeric_is_rejected_without_crashing(bad):
    result = validate_student_input({**OK, "Attendance": bad})
    assert not result.ok
    assert any("Attendance" in message for message in result.errors)


@pytest.mark.parametrize("bad", [np.nan, pd.NA, None, "", 5, True])
def test_missing_or_wrong_type_category_is_rejected(bad):
    result = validate_student_input({**OK, "Gender": bad})
    assert not result.ok


def test_boolean_is_not_accepted_as_a_number():
    assert not validate_student_input({**OK, "Study_Hours": True}).ok


def test_text_in_numeric_field_gives_friendly_error():
    result = validate_student_input({**OK, "Previous_Score": "abc"})
    assert result.errors == ["Previous score must be a number (got 'abc')."]


def test_boundaries_are_inclusive_and_just_outside_is_rejected():
    assert validate_student_input({**OK, "Attendance": 0}).ok
    assert validate_student_input({**OK, "Attendance": 100}).ok
    assert not validate_student_input({**OK, "Attendance": 100.01}).ok
    assert not validate_student_input({**OK, "Attendance": -0.01}).ok


def test_valid_input_is_not_modified_in_meaning():
    clean = validate_student_input(OK).clean
    for feature in FEATURES:
        assert clean[feature] == OK[feature]


def test_arabic_text_in_category_is_rejected_not_guessed():
    assert not validate_student_input({**OK, "Gender": "ذكر"}).ok
