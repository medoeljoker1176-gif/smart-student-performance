"""
explainability.py
-----------------
Explains WHY the model gave a certain probability, using Shapley values.

What is a Shapley value?  Imagine the 10 features as a team that together move
the prediction away from the "average student". A feature's Shapley value is its
fair share of that movement: the average change in P(pass) when the feature is
added to every possible subset of the other features. This is exactly the idea
behind the SHAP method.

Implementation note: we have only 10 features, so we can compute the values
EXACTLY (2^10 = 1024 subsets) with our own code instead of approximating them.
A feature that is "switched off" is replaced by values from reference students
(the background set) and the predictions are averaged. Works for ANY model that
gives P(pass), so the same code explains Logistic Regression and Random Forest.

Sanity check (used in tests): contributions always add up exactly to
    P(pass of this student) - average P(pass of the reference students).
"""
from math import factorial

import numpy as np
import pandas as pd

from src.config import FEATURES, FEATURE_LABELS, FEATURE_UNITS, NUMERIC_FEATURES


def shapley_values(predict_fn, x: pd.Series, background: pd.DataFrame, features=None):
    """Exact Shapley values of one student, in probability units.

    predict_fn : DataFrame -> array of P(pass)
    x          : the student (Series indexed by feature name)
    background : reference students (DataFrame with the same columns)
    Returns (phi, base_value, prediction): phi is a Series, base_value = mean P(pass)
    of the background, prediction = P(pass) of the student.
    """
    features = list(features or x.index)
    d, n_bg = len(features), len(background)
    n_masks = 2 ** d

    # masks[m, j] is True when feature j is "present" (uses the student's value) in subset m
    ids = np.arange(n_masks)
    masks = ((ids[:, None] >> np.arange(d)) & 1).astype(bool)

    x_arr = x[features].to_numpy(dtype=object)
    bg_arr = background[features].to_numpy(dtype=object)
    mixed = np.where(masks[:, None, :], x_arr[None, None, :], bg_arr[None, :, :])
    frame = pd.DataFrame(mixed.reshape(-1, d), columns=features)
    for col in features:                                   # restore numeric dtypes
        if col in NUMERIC_FEATURES:
            frame[col] = frame[col].astype(float)

    # v[m] = average prediction when only the features in subset m are "known"
    v = np.asarray(predict_fn(frame), dtype=float).reshape(n_masks, n_bg).mean(axis=1)

    popcount = masks.sum(axis=1)
    weights = np.array([factorial(k) * factorial(d - k - 1) / factorial(d) for k in range(d)])
    phi = np.zeros(d)
    for j in range(d):
        without_j = ids[~masks[:, j]]
        phi[j] = np.sum(weights[popcount[without_j]] * (v[without_j | (1 << j)] - v[without_j]))
    return pd.Series(phi, index=features), float(v[0]), float(v[-1])


def _describe_value(feature: str, value) -> str:
    if feature in NUMERIC_FEATURES:
        return f"{value:g}{FEATURE_UNITS.get(feature, '')}"
    return str(value)


def explain_student(predict_fn, student: dict, background: pd.DataFrame) -> dict:
    """Shapley explanation packaged for the UI.

    Returns {"base", "prediction", "table"} where table has one row per feature,
    sorted by absolute impact: Feature, Value, Contribution (points of P(pass)), Sentence.
    """
    x = pd.Series({f: student[f] for f in FEATURES})
    phi, base, pred = shapley_values(predict_fn, x, background, FEATURES)
    rows = []
    for feat, contrib in phi.items():
        points = contrib * 100
        value_text = _describe_value(feat, x[feat])
        if abs(points) < 0.5:
            sentence = f"{FEATURE_LABELS[feat]} ({value_text}) had almost no effect."
        elif points > 0:
            sentence = f"{FEATURE_LABELS[feat]} ({value_text}) increased the chance of passing by {points:.1f} points."
        else:
            sentence = f"{FEATURE_LABELS[feat]} ({value_text}) decreased the chance of passing by {abs(points):.1f} points."
        rows.append({"Feature": feat, "Label": FEATURE_LABELS[feat], "Value": value_text,
                     "Contribution": points, "Sentence": sentence})
    table = pd.DataFrame(rows).sort_values("Contribution", key=np.abs, ascending=False).reset_index(drop=True)
    return {"base": base, "prediction": pred, "table": table}


def global_importance(predict_fn, students: pd.DataFrame, background: pd.DataFrame) -> dict:
    """Mean |Shapley value| per feature over a representative student sample.

    Shapley values are exact over all feature subsets for each sampled student.
    The global summary uses a bounded student/background sample so model training
    remains practical on laptops. Individual explanations still use the full
    saved background set.
    """
    if students.empty:
        return {feature: 0.0 for feature in FEATURES}
    total = pd.Series(0.0, index=FEATURES)
    for _, row in students[FEATURES].iterrows():
        phi, _, _ = shapley_values(predict_fn, row, background, FEATURES)
        total += phi.abs()
    return (total / len(students) * 100).sort_values(ascending=False).to_dict()
