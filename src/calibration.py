"""Probability calibration check: do predicted probabilities match observed pass rates?"""
from __future__ import annotations

import numpy as np
import pandas as pd

BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def calibration_table(y_true, p_pass, edges=BIN_EDGES) -> pd.DataFrame:
    """Group students by predicted P(pass) and compare it with the share who really passed.

    Returns one row per non-empty bin: Bin, Students, Mean predicted, Observed pass rate.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pass, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and p_pass must have the same length")
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    # include 1.0 in the last bin
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({"Bin": f"{edges[b]:.0%}-{edges[b + 1]:.0%}", "Students": n,
                     "Mean predicted": float(p[mask].mean()), "Observed pass rate": float(y[mask].mean())})
    return pd.DataFrame(rows, columns=["Bin", "Students", "Mean predicted", "Observed pass rate"])


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Weighted average gap between predicted and observed rates (0 = perfectly calibrated)."""
    if table.empty:
        return float("nan")
    w = table["Students"] / table["Students"].sum()
    return float((w * (table["Mean predicted"] - table["Observed pass rate"]).abs()).sum())
