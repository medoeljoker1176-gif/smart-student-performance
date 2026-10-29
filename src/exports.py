"""Safe helpers for downloadable tabular exports."""
from __future__ import annotations

import pandas as pd


_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def dataframe_to_safe_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a DataFrame as UTF-8 CSV, neutralizing spreadsheet formulas in text.

    Numeric values remain numeric. String values beginning with characters commonly
    interpreted as spreadsheet formulas are prefixed with an apostrophe so opening a
    downloaded file in spreadsheet software does not execute them as formulas.
    """
    safe = frame.copy()
    for column in safe.columns:
        if pd.api.types.is_object_dtype(safe[column]) or pd.api.types.is_string_dtype(safe[column]):
            def neutralize(value):
                if not isinstance(value, str):
                    return value
                # Spreadsheet applications may ignore leading whitespace before formulas.
                candidate = value.lstrip(" \t\r\n")
                if candidate.startswith(_FORMULA_PREFIXES):
                    return "'" + value
                return value
            safe[column] = safe[column].map(neutralize)
    return safe.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def student_result_to_frame(student: dict, result, student_id: str = "") -> pd.DataFrame:
    """One-row table with a student's inputs and prediction, ready for dataframe_to_safe_csv_bytes."""
    row = {"Student_ID": student_id or "", **student,
           "Prediction": result.label, "P_Pass": round(float(result.p_pass), 4),
           "Risk": result.risk["level"], "Performance_Score": round(float(result.score), 1)}
    return pd.DataFrame([row])
