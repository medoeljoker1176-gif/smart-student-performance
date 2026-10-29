"""Builds a small file in the exact UCI student-mat.csv layout (';'-separated, quoted text) for tests.
It is made-up data - the real UCI files are not bundled here. Only the columns the adapter needs
plus the merge keys are produced."""
import numpy as np
import pandas as pd


def make_uci_frame(n=300, seed=3, subject_offset=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, n)
    g1 = np.clip(np.round(10.5 + 3 * ability + rng.normal(0, 1.5, n)), 0, 20)
    g2 = np.clip(np.round(g1 + rng.normal(0, 1.3, n)), 0, 20)
    g3 = np.clip(np.round(g2 + rng.normal(0, 1.5, n)), 0, 20)
    g3 = np.where(rng.random(n) < 0.04, 0, g3)                       # drop-outs score 0 in UCI
    return pd.DataFrame({
        "school": rng.choice(["GP", "MS"], n), "sex": rng.choice(["F", "M"], n),
        "age": rng.integers(15, 23, n), "address": rng.choice(["U", "R"], n),
        "famsize": rng.choice(["GT3", "LE3"], n), "Pstatus": rng.choice(["T", "A"], n),
        "Medu": rng.integers(0, 5, n), "Fedu": rng.integers(0, 5, n),
        "Mjob": rng.choice(["at_home", "health", "other", "services", "teacher"], n),
        "Fjob": rng.choice(["at_home", "health", "other", "services", "teacher"], n),
        "reason": rng.choice(["course", "home", "reputation", "other"], n),
        "nursery": rng.choice(["yes", "no"], n), "internet": rng.choice(["yes", "no"], n, p=[.8, .2]),
        "activities": rng.choice(["yes", "no"], n),
        "studytime": np.clip(np.round(2 + 0.4 * ability + rng.normal(0, 0.8, n)), 1, 4).astype(int),
        "absences": np.clip(np.round(rng.gamma(1.5, 3, n)), 0, 75).astype(int),
        "G1": g1.astype(int), "G2": g2.astype(int), "G3": g3.astype(int),
    })


def write_uci_file(path, frame) -> None:
    frame.to_csv(path, sep=";", index=False, quoting=1)             # quote everything like the original
