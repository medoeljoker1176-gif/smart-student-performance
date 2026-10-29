"""
validation.py
-------------
Defensive checks. Nothing invalid should ever reach the model.

  * validate_student_input()  - one student typed into the app
  * validate_dataframe()      - the training CSV (columns, types, categories)
"""
from dataclasses import dataclass, field
import math

import pandas as pd

from src.config import (CATEGORY_VALUES, CATEGORICAL_FEATURES, FEATURE_LABELS, FEATURES,
                        INPUT_LIMITS, NUMERIC_FEATURES, POSITIVE_LABEL, NEGATIVE_LABEL,
                        TARGET, UNUSUAL_LIMITS)


class DataError(Exception):
    """Raised when the dataset is missing or has an unexpected structure."""


@dataclass
class ValidationResult:
    clean: dict = field(default_factory=dict)      # values converted to proper types
    errors: list = field(default_factory=list)     # must be fixed -> no prediction
    warnings: list = field(default_factory=list)   # allowed but unusual

    @property
    def ok(self) -> bool:
        return not self.errors


def _label(col: str) -> str:
    return FEATURE_LABELS.get(col, col)


def _is_missing(value) -> bool:
    """True for None, NaN, pandas NA/NaT and blank strings. Never raises on pd.NA."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):          # array-like or other odd object: treat as present, type checks follow
        return False


def validate_student_input(raw: dict, training_ranges: dict | None = None) -> ValidationResult:
    """Check one student's inputs. Returns cleaned values + friendly messages.

    training_ranges: optional {feature: (min, max)} seen during training; values
    outside it get a warning because the model is extrapolating.
    """
    res = ValidationResult()

    # 1. every required field must be present
    for col in FEATURES:
        if col not in raw or _is_missing(raw[col]):
            res.errors.append(f"{_label(col)} is required.")
    if res.errors:
        return res

    # 2. numeric fields: must be finite numbers inside the hard limits
    for col in NUMERIC_FEATURES:
        lo, hi = INPUT_LIMITS[col]
        if isinstance(raw[col], bool):                       # True/False is not a measurement
            res.errors.append(f"{_label(col)} must be a number (got '{raw[col]}').")
            continue
        try:
            value = float(raw[col])
        except (TypeError, ValueError):
            res.errors.append(f"{_label(col)} must be a number (got '{raw[col]}').")
            continue
        if math.isnan(value) or math.isinf(value):
            res.errors.append(f"{_label(col)} must be a real number.")
            continue
        if not lo <= value <= hi:
            res.errors.append(f"{_label(col)} must be between {lo} and {hi} (got {value:g}).")
            continue
        if col == "Age":
            if value != int(value):
                res.errors.append("Age must be a whole number.")
                continue
            value = int(value)
        res.clean[col] = value

    # 3. categorical fields: must be one of the known categories
    for col in CATEGORICAL_FEATURES:
        if not isinstance(raw[col], str) or raw[col] not in CATEGORY_VALUES[col]:
            options = ", ".join(CATEGORY_VALUES[col])
            res.errors.append(f"{_label(col)} must be one of: {options}.")
        else:
            res.clean[col] = raw[col]

    if res.errors:
        return res

    # 4. cross-field logic: a day only has 24 hours
    total = res.clean["Study_Hours"] + res.clean["Sleep_Hours"]
    if total > 24:
        res.errors.append(f"Study hours + sleep hours = {total:g} h, which is more than 24 h per day.")
        return res
    if total > 20:
        res.warnings.append(f"Study + sleep = {total:g} h leaves very little time for anything else.")

    # 5. warnings: possible but unusual, or outside the range the model has seen
    for col, (lo, hi) in UNUSUAL_LIMITS.items():
        v = res.clean[col]
        if v < lo or v > hi:
            res.warnings.append(f"{_label(col)} = {v:g} is unusual (typical range {lo}-{hi}).")
    for col, (lo, hi) in (training_ranges or {}).items():
        if col in res.clean and not lo <= res.clean[col] <= hi:
            res.warnings.append(f"{_label(col)} = {res.clean[col]:g} is outside the range the model was "
                                f"trained on ({lo:g}-{hi:g}); the prediction is less reliable.")
    return res


def validate_dataframe(df: pd.DataFrame) -> list:
    """Return a list of problems found in the dataset (empty list = OK)."""
    problems = []
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        return [f"Missing required column(s): {', '.join(missing_cols)}"]
    if len(df) == 0:
        return ["The dataset is empty."]

    for col in NUMERIC_FEATURES:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = int((converted.isna() & df[col].notna()).sum())
        non_finite = int((converted.notna() & ~converted.map(math.isfinite)).sum())
        if bad:
            problems.append(f"Column '{col}' has {bad} non-numeric value(s).")
        if non_finite:
            problems.append(f"Column '{col}' has {non_finite} non-finite value(s) (infinity is not allowed).")
    for col in CATEGORICAL_FEATURES:
        unknown = set(df[col].dropna().unique()) - set(CATEGORY_VALUES[col])
        if unknown:
            problems.append(f"Column '{col}' has unexpected categories: {sorted(unknown)}")
    labels = set(df[TARGET].dropna().unique())
    if not labels <= {POSITIVE_LABEL, NEGATIVE_LABEL}:
        problems.append(f"Column '{TARGET}' must contain only Pass/Fail (found {sorted(labels)}).")
    if len(labels) < 2:
        problems.append("The target needs both Pass and Fail examples to train a classifier.")
    return problems
