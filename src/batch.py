"""Testable batch CSV validation and prediction helpers.

This module intentionally has no Streamlit dependency so CSV edge cases can be
unit-tested without launching the UI.
"""
from __future__ import annotations

import pandas as pd

from src.config import FEATURES
from src.prediction import assess_risk, predict_proba_df
from src.validation import validate_student_input

MAX_ROWS = 5000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024          # 5 MB: far above 5,000 rows of 10 columns

PREDICTION_COLUMNS = ["CSV row", *FEATURES, "Predicted result", "P(pass)", "Risk category", "Validation warnings"]
REJECTED_COLUMNS = ["CSV row", "Errors"]


class BatchInputError(ValueError):
    """Raised when an uploaded CSV cannot be processed as a batch."""


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose column names have no BOM or surrounding spaces.

    Only header text is touched; no student value is changed.
    """
    renamed = frame.copy()
    renamed.columns = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
    return renamed


def process_batch_dataframe(
    frame: pd.DataFrame,
    bundle: dict,
    max_rows: int = MAX_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate a CSV frame and return (predictions, rejected_rows).

    Additional columns are ignored. Input row numbers refer to physical CSV
    lines (header is line 1), so the first data row is reported as line 2.
    Invalid rows are never sent to the model. All valid rows are scored in a
    single vectorised call to the saved pipeline (the model is never retrained).
    """
    if not isinstance(frame, pd.DataFrame):
        raise BatchInputError("The uploaded data could not be interpreted as a table.")
    if frame.empty:
        raise BatchInputError("The uploaded CSV has no data rows.")
    if len(frame) > max_rows:
        raise BatchInputError(f"This demo limits each upload to {max_rows:,} rows. Split the file into smaller batches.")

    frame = normalize_columns(frame)
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise BatchInputError("Missing required column(s): " + ", ".join(missing))
    duplicated = [column for column in FEATURES if (frame.columns == column).sum() > 1]
    if duplicated:
        raise BatchInputError("Column(s) appear more than once: " + ", ".join(duplicated)
                              + ". Keep a single column for each feature.")

    training_ranges = bundle.get("training_ranges") if isinstance(bundle, dict) else None

    accepted_rows: list[int] = []
    accepted_clean: list[dict] = []
    accepted_warnings: list[str] = []
    rejected: list[dict] = []

    records = frame[FEATURES].to_dict("records")
    for position, record in enumerate(records, start=2):
        raw = {feature: (value.strip() if isinstance(value, str) else value) for feature, value in record.items()}
        checked = validate_student_input(raw, training_ranges)
        if not checked.ok:
            rejected.append({"CSV row": position, "Errors": "; ".join(checked.errors)})
            continue
        accepted_rows.append(position)
        accepted_clean.append(checked.clean)
        accepted_warnings.append("; ".join(checked.warnings))

    if accepted_clean:
        clean_frame = pd.DataFrame(accepted_clean, columns=FEATURES)
        probabilities = [float(p) for p in predict_proba_df(bundle, clean_frame)]
        out = clean_frame.copy()
        out.insert(0, "CSV row", accepted_rows)
        out["Predicted result"] = ["Pass" if p >= 0.5 else "Fail" for p in probabilities]
        out["P(pass)"] = [round(p, 4) for p in probabilities]
        out["Risk category"] = [assess_risk(p)["level"] for p in probabilities]
        out["Validation warnings"] = accepted_warnings
        predictions = out[PREDICTION_COLUMNS]
    else:
        predictions = pd.DataFrame(columns=PREDICTION_COLUMNS)

    return predictions.reset_index(drop=True), pd.DataFrame(rejected, columns=REJECTED_COLUMNS)


def summarize_batch(predictions: pd.DataFrame) -> dict:
    """Numbers shown above the batch charts (computed only from real predictions)."""
    n = len(predictions)
    if n == 0:
        return {"n": 0, "pass_share": None, "mean_p_pass": None, "risk_counts": {}, "result_counts": {}}
    return {
        "n": n,
        "pass_share": float((predictions["Predicted result"] == "Pass").mean()),
        "mean_p_pass": float(predictions["P(pass)"].mean()),
        "risk_counts": predictions["Risk category"].value_counts().to_dict(),
        "result_counts": predictions["Predicted result"].value_counts().to_dict(),
    }
