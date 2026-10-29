"""
fairness.py
-----------
Basic group-fairness check on out-of-fold predictions (every student was predicted
by a model that never saw that student). Positive outcome = "Pass".

Metrics per group:
  predicted pass rate  - share of students the model predicts to pass
  recall (TPR)         - of the students who really passed, share predicted Pass
  FPR                  - of the students who really failed, share wrongly predicted Pass
  FNR                  - of the students who really passed, share wrongly predicted Fail
"""
from math import erfc, sqrt

import pandas as pd

MIN_GROUP_SIZE = 30


def _safe_div(a, b):
    return float(a / b) if b else float("nan")


def group_metrics(df: pd.DataFrame, group_col: str = "Gender") -> pd.DataFrame:
    """df needs columns: group_col, y_true (1=Pass), y_pred (1=Pass)."""
    rows = []
    for group, g in df.groupby(group_col):
        tp = int(((g.y_true == 1) & (g.y_pred == 1)).sum())
        fn = int(((g.y_true == 1) & (g.y_pred == 0)).sum())
        fp = int(((g.y_true == 0) & (g.y_pred == 1)).sum())
        tn = int(((g.y_true == 0) & (g.y_pred == 0)).sum())
        rows.append({
            group_col: group, "Students": len(g),
            "Actual pass rate": _safe_div(tp + fn, len(g)),
            "Predicted pass rate": _safe_div(tp + fp, len(g)),
            "Accuracy": _safe_div(tp + tn, len(g)),
            "Recall (TPR)": _safe_div(tp, tp + fn),
            "False positive rate": _safe_div(fp, fp + tn),
            "False negative rate": _safe_div(fn, fn + tp),
        })
    return pd.DataFrame(rows)


def two_proportion_pvalue(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided z-test p-value for 'are these two rates different?'."""
    if min(n1, n2) == 0:
        return float("nan")
    p1, p2, pooled = x1 / n1, x2 / n2, (x1 + x2) / (n1 + n2)
    se = sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return float(erfc(abs(p1 - p2) / se / sqrt(2)))


def fairness_summary(df: pd.DataFrame, group_col: str = "Gender") -> dict:
    """Table + gaps between the two groups + a plain-language conclusion."""
    table = group_metrics(df, group_col)
    out = {"table": table, "small_groups": bool((table["Students"] < MIN_GROUP_SIZE).any())}
    if len(table) == 2:
        a, b = table.iloc[0], table.iloc[1]
        out["gaps"] = {
            "Predicted pass rate": abs(a["Predicted pass rate"] - b["Predicted pass rate"]),
            "Recall (TPR)": abs(a["Recall (TPR)"] - b["Recall (TPR)"]),
            "False positive rate": abs(a["False positive rate"] - b["False positive rate"]),
            "False negative rate": abs(a["False negative rate"] - b["False negative rate"]),
        }
        pa, pb = df[df[group_col] == a[group_col]], df[df[group_col] == b[group_col]]
        out["p_value_pass_rate"] = two_proportion_pvalue(int(pa.y_pred.sum()), len(pa), int(pb.y_pred.sum()), len(pb))
    return out
