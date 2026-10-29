"""Synthetic data generator: reproducibility, realism, leakage and edge cases."""
import numpy as np
import pandas as pd
import pytest

from generate_dataset import COLUMNS, generate_dataset
from src.config import CATEGORY_VALUES, FEATURES, INPUT_LIMITS, TARGET
from src.preprocessing import clean_data
from src.validation import validate_dataframe


@pytest.fixture(scope="module")
def df():
    return generate_dataset()


def test_same_seed_same_data_and_other_seed_differs(df):
    pd.testing.assert_frame_equal(df, generate_dataset())
    assert not df.equals(generate_dataset(seed=7))


def test_schema_matches_app_and_passes_validation(df):
    assert list(df.columns) == COLUMNS                              # original CSV layout
    assert set(df.columns) == set(FEATURES) | {TARGET}             # no extra / leaking column
    assert validate_dataframe(df) == []
    for col, allowed in CATEGORY_VALUES.items():
        assert set(df[col].dropna()) <= set(allowed)


def test_values_in_limits_and_nothing_lost_in_cleaning(df):
    for col, (lo, hi) in INPUT_LIMITS.items():
        assert df[col].dropna().between(lo, hi).all()
    assert ((df.Study_Hours + df.Sleep_Hours).dropna() <= 24).all()
    assert len(clean_data(df)) == len(df)                          # cleaning drops nothing


def test_realistic_class_balance_and_missingness(df):
    assert 0.60 <= (df[TARGET] == "Pass").mean() <= 0.78
    assert df[TARGET].notna().all() and not df.duplicated().any()
    assert 0.01 < df[FEATURES].isna().mean().max() < 0.08
    assert df[FEATURES].isna().sum(axis=1).max() <= 2


def test_logical_relationships():
    d = generate_dataset(n=3000, include_exam_score=True)
    corr = d.select_dtypes("number").corr()
    assert corr.loc["Previous_Score", "Exam_Score"] > 0.5
    assert corr.loc["Attendance", "Exam_Score"] > 0.3
    assert corr.loc["Study_Hours", "Exam_Score"] > 0.15
    assert corr.loc["Study_Hours", "Sleep_Hours"] < 0
    assert corr.loc["Previous_Score", "Assignment_Score"] > 0.3
    rate = d.assign(p=d[TARGET] == "Pass").groupby("Participation").p.mean()
    assert rate["Low"] < rate["Medium"] < rate["High"]
    sleep_ok = d.Sleep_Hours.between(6.5, 8.5)
    assert d.loc[sleep_ok, "Exam_Score"].mean() > d.loc[d.Sleep_Hours < 5.5, "Exam_Score"].mean()


def test_no_leakage_missingness_and_gender_independent_of_result():
    d = generate_dataset(n=5000)
    passed = d[TARGET] == "Pass"
    miss_rate = d[FEATURES].isna().any(axis=1)
    assert abs(miss_rate[passed].mean() - miss_rate[~passed].mean()) < 0.03
    gap = abs(passed[d.Gender == "Male"].mean() - passed[d.Gender == "Female"].mean())
    assert gap < 0.04


def test_exam_score_excluded_by_default_and_opt_in_only(df):
    assert "Exam_Score" not in df.columns
    assert "Exam_Score" in generate_dataset(include_exam_score=True).columns


@pytest.mark.parametrize("bad", [0, -5, 10, 49, 2.5, "100", None, True])
def test_bad_sizes_rejected(bad):
    with pytest.raises(ValueError):
        generate_dataset(n=bad)


@pytest.mark.parametrize("n", [50, 120, 5000])
def test_small_and_large_sizes_keep_both_classes(n):
    d = generate_dataset(n=n)
    assert len(d) == n and d[TARGET].nunique() == 2
