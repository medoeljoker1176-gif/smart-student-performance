"""
history.py
----------
Prediction history with a pluggable storage backend.

* ``csv`` (default) - a local CSV file; no setup, but on a managed cloud service the file system is
  ephemeral and not shared between instances, so the data is lost on restart or scale-out.
* ``firestore`` - Google Cloud Firestore; durable and shared by every instance of the service.
  Selected with the environment variable ``HISTORY_BACKEND=firestore`` (see CLOUD_DEPLOYMENT.md).

Repeated predictions with the same Student_ID let us monitor progress over time.
"""
import os
from datetime import datetime

import pandas as pd

from src.config import FEATURES, HISTORY_PATH

COLUMNS = ["Timestamp", "Student_ID"] + FEATURES + ["Prediction", "P_Pass", "Risk", "Performance_Score"]
FIRESTORE_COLLECTION = os.environ.get("HISTORY_COLLECTION", "prediction_history")
_FIRESTORE_CLIENT = None


def backend() -> str:
    """Active storage backend: 'csv' (default) or 'firestore'."""
    name = os.environ.get("HISTORY_BACKEND", "csv").strip().lower()
    return name if name in ("csv", "firestore") else "csv"


def _firestore():
    """Lazily create one Firestore client per process (the library is optional)."""
    global _FIRESTORE_CLIENT
    if _FIRESTORE_CLIENT is None:
        from google.cloud import firestore          # imported here so local runs need no extra package
        _FIRESTORE_CLIENT = firestore.Client()
    return _FIRESTORE_CLIENT


def add_record(student: dict, result, student_id: str = "") -> None:
    """Append one prediction. `result` is a prediction.PredictionResult."""
    row = {"Timestamp": datetime.now().isoformat(timespec="seconds"),
           "Student_ID": (student_id or "").strip(), **{f: student[f] for f in FEATURES},
           "Prediction": result.label, "P_Pass": round(result.p_pass, 4),
           "Risk": result.risk["level"], "Performance_Score": result.score}
    if backend() == "firestore":
        _firestore().collection(FIRESTORE_COLLECTION).add(row)
        return
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0
    pd.DataFrame([row], columns=COLUMNS).to_csv(HISTORY_PATH, mode="a", header=new_file, index=False)


def read_history() -> pd.DataFrame:
    """All saved predictions (empty DataFrame if none or if the storage is unreadable)."""
    if backend() == "firestore":
        try:
            docs = [d.to_dict() for d in _firestore().collection(FIRESTORE_COLLECTION).stream()]
        except Exception:
            return pd.DataFrame(columns=COLUMNS)
        df = pd.DataFrame(docs, columns=COLUMNS) if docs else pd.DataFrame(columns=COLUMNS)
        return _clean(df)
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        return _clean(pd.read_csv(HISTORY_PATH, dtype={"Student_ID": str}))
    except Exception:
        return pd.DataFrame(columns=COLUMNS)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only well-formed rows so a damaged record can never crash a page."""
    try:
        if not set(COLUMNS) <= set(df.columns):
            return pd.DataFrame(columns=COLUMNS)
        df["Student_ID"] = df["Student_ID"].fillna("").astype(str)
        # A hand-edited file must not crash the pages: keep only rows with usable numbers.
        for col in ("P_Pass", "Performance_Score"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["P_Pass", "Timestamp", "Risk", "Prediction"])
        df = df[df["P_Pass"].between(0, 1)]
        return df[COLUMNS].reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)


def clear_history() -> None:
    if backend() == "firestore":
        col = _firestore().collection(FIRESTORE_COLLECTION)
        for doc in col.stream():
            doc.reference.delete()
        return
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()


def summarize_history(df: pd.DataFrame) -> dict:
    """Headline numbers for the dashboard/history page (computed only from saved rows)."""
    if df is None or df.empty:
        return {"n": 0, "mean_p_pass": None, "predicted_pass_share": None, "risk_counts": {}, "recent": df}
    return {
        "n": int(len(df)),
        "mean_p_pass": float(df["P_Pass"].mean()),
        "predicted_pass_share": float((df["Prediction"] == "Pass").mean()),
        "risk_counts": df["Risk"].value_counts().to_dict(),
        "recent": df.sort_values("Timestamp", ascending=False).head(5),
    }
