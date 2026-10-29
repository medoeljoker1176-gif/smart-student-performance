"""
prediction.py
-------------
Load the saved model bundle, predict, assign a risk level, compute the
application-level Performance Score.

The .pkl file is a joblib "bundle" (a dict) holding the fitted Pipeline plus what
the app needs (feature lists, reference students for explanations, thresholds...).
Security note: joblib files are pickles, so only load model files you trained yourself.
"""
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.config import (FEATURES, HIGH_RISK_BELOW, LOW_RISK_FROM, MODEL_PATH, NEGATIVE_LABEL,
                        POSITIVE_LABEL)

REQUIRED_KEYS = {"pipeline", "model_name", "background", "training_ranges"}


class ModelLoadError(Exception):
    """Raised with a friendly message when the model cannot be loaded."""


def _minor(version: str) -> str:
    """'1.8.2' -> '1.8' (patch releases keep the pickle format, minor releases may not)."""
    return ".".join(str(version).split(".")[:2])


def load_bundle(path: Path = MODEL_PATH) -> dict:
    """Load and sanity-check the model bundle."""
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(f"Model file not found at '{path}'. Run `python train_model.py` first.")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(f"The model file could not be loaded ({type(exc).__name__}: {exc}). "
                             "Retrain it with `python train_model.py`.") from exc
    if not isinstance(bundle, dict) or not REQUIRED_KEYS <= set(bundle):
        raise ModelLoadError("The model file has an old or unexpected format. "
                             "Retrain it with `python train_model.py`.")
    if not hasattr(bundle["pipeline"], "predict_proba"):
        raise ModelLoadError("The saved model cannot output probabilities.")
    bundle["warnings"] = []
    trained_with = bundle.get("sklearn_version")
    if trained_with and _minor(trained_with) != _minor(sklearn.__version__):
        # A pickle from another scikit-learn minor version can load fine and then fail later
        # (e.g. "'SimpleImputer' object has no attribute '_fill_dtype'"), so refuse it up front.
        # The app reacts by retraining the model once in the current environment.
        raise ModelLoadError(f"The model was trained with scikit-learn {trained_with} but this environment has "
                             f"{sklearn.__version__}. Retrain it with `python train_model.py`.")
    return bundle


def predict_proba_df(bundle: dict, df: pd.DataFrame):
    """P(pass) for every row of a DataFrame (columns in the training order)."""
    return bundle["pipeline"].predict_proba(df[FEATURES])[:, 1]


def make_predict_fn(bundle: dict):
    """Function DataFrame -> P(pass); used by explanations and what-if checks."""
    return lambda df: predict_proba_df(bundle, df)


def p_pass_for(bundle: dict, student: dict) -> float:
    return float(predict_proba_df(bundle, pd.DataFrame([student]))[0])


def format_probability(p: float, decimals: int = 1) -> str:
    """Readable probability that never claims certainty ("<1%" / ">99%" instead of 0.0% / 100.0%)."""
    pct = float(p) * 100
    if pct < 1:
        return "<1%"
    if pct > 99:
        return ">99%"
    return f"{pct:.{decimals}f}%"


# ------------------------------------------------------------------ risk level
def assess_risk(p_pass: float) -> dict:
    """Map P(pass) to a risk level.

    High risk     : P(pass) <  0.50   (the model itself predicts Fail)
    Moderate risk : 0.50 <= P(pass) < 0.75   (predicted Pass, but not confident)
    Low risk      : P(pass) >= 0.75
    Thresholds are design choices for this prototype, not validated standards.
    """
    if p_pass < HIGH_RISK_BELOW:
        return {"level": "High Risk", "color": "#E5534B", "icon": "🔴",
                "message": "The model expects this student to fail. Act early on the priorities below."}
    if p_pass < LOW_RISK_FROM:
        return {"level": "Moderate Risk", "color": "#F5A524", "icon": "🟠",
                "message": "The model expects a pass, but with limited confidence. Small improvements matter."}
    return {"level": "Low Risk", "color": "#1FA774", "icon": "🟢",
            "message": "The model is confident this student will pass. Keep the good habits."}


# ------------------------------------------------------------ performance score
# Application-level indicator: a weighted average of inputs, each scaled to 0-100.
# The weights are chosen by us for readability. They are NOT learned by the model
# and NOT an official grade.
SCORE_WEIGHTS = {"Previous_Score": 0.25, "Attendance": 0.25, "Assignment_Score": 0.20,
                 "Study_Hours": 0.15, "Participation": 0.10, "Sleep_Hours": 0.05}
PARTICIPATION_POINTS = {"Low": 0, "Medium": 60, "High": 100}


def _component_scores(s: dict) -> dict:
    return {
        "Previous_Score": float(s["Previous_Score"]),
        "Attendance": float(s["Attendance"]),
        "Assignment_Score": float(s["Assignment_Score"]),
        "Study_Hours": min(float(s["Study_Hours"]) / 6.0, 1.0) * 100,     # 6 h/day or more = 100
        "Participation": float(PARTICIPATION_POINTS[s["Participation"]]),
        "Sleep_Hours": max(0.0, 1 - abs(float(s["Sleep_Hours"]) - 7.5) / 3.5) * 100,  # best at 7.5 h
    }


def performance_score(student: dict):
    """Returns (score 0-100, per-component table)."""
    parts = _component_scores(student)
    score = sum(SCORE_WEIGHTS[k] * parts[k] for k in SCORE_WEIGHTS)
    rows = [{"Component": k, "Weight": SCORE_WEIGHTS[k], "Component score": round(parts[k], 1),
             "Points": round(SCORE_WEIGHTS[k] * parts[k], 1)} for k in SCORE_WEIGHTS]
    return round(score, 1), pd.DataFrame(rows)


# ------------------------------------------------------------------ main entry
@dataclass
class PredictionResult:
    label: str            # "Pass" / "Fail"
    p_pass: float
    risk: dict
    score: float
    score_table: pd.DataFrame


def predict_student(bundle: dict, student: dict) -> PredictionResult:
    """`student` must already be validated (see validation.validate_student_input)."""
    p = p_pass_for(bundle, student)
    score, table = performance_score(student)
    return PredictionResult(POSITIVE_LABEL if p >= 0.5 else NEGATIVE_LABEL, p, assess_risk(p), score, table)


def analyze_student(bundle: dict, student: dict) -> dict:
    """Everything the dashboard shows for one validated student:
    prediction + risk + score, Shapley explanation, prioritised recommendations."""
    from src.explainability import explain_student          # imported here to keep module load light
    from src.recommendations import get_prioritized_recommendations
    result = predict_student(bundle, student)
    explanation = explain_student(make_predict_fn(bundle), student, bundle["background"])
    recs = get_prioritized_recommendations(student, lambda s: p_pass_for(bundle, s))
    return {"student": dict(student), "result": result, "explanation": explanation, "recs": recs,
            "model_name": bundle.get("model_name", "Saved model")}
