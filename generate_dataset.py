"""
generate_dataset.py
-------------------
Creates a synthetic (made-up but realistic) student performance dataset.

Why synthetic?  No public dataset has exactly the features we want, and a
generated one lets everybody reproduce the same results (fixed random seed).

How the data is built (in simple words)
  1. Every student gets a few HIDDEN traits that are never saved: academic
     ability, conscientiousness (self-discipline) and socio-economic status.
  2. The visible features are generated FROM those traits, so they are
     correlated the way they are in real classes:
       * disciplined students study more, attend more and take part more;
       * able students have higher previous and assignment scores;
       * students with internet at home tend to come from better-off homes;
       * heavy study slightly reduces sleep.
  3. A hidden EXAM SCORE (0-100) is computed from those causes with
     diminishing returns for study hours, a sweet-spot for sleep and noise
     that is larger for students with an irregular profile.
  4. Result = "Pass" when the exam score reaches the pass mark (50). The pass
     rate is therefore an outcome of the data, not forced to a fixed number.

Data-leakage rules
  * The exam score is the thing we predict, so it is NOT written to the CSV
    (pass `include_exam_score=True` only for analysis; never train on it).
  * Assignment_Score and Previous_Score come only from things that happen
    BEFORE the exam (ability, effort, attendance), never from the exam/Result.
  * Missing values depend only on observed, non-target information, and the
    missing-value mask is drawn independently of the label.
  * Gender has no causal effect on the outcome (useful for the fairness page).

Edge cases handled: every value stays inside the app's hard input limits,
study + sleep <= 24 h, no duplicated rows, no row with too many gaps, the
label is never missing, and both classes are always present.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_STUDENTS = 1000
PASS_MARK = 50.0
MISSING_RATE = 0.03                      # about 3 % of values in the columns below
MISSING_COLUMNS = ["Study_Hours", "Attendance", "Sleep_Hours", "Assignment_Score", "Participation"]
MAX_MISSING_PER_ROW = 2
MIN_N = 50                               # below this, class balance is too unstable
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "student_performance.csv"

COLUMNS = ["Age", "Gender", "Study_Hours", "Attendance", "Previous_Score", "Assignment_Score",
           "Sleep_Hours", "Participation", "Internet_Access", "Extracurricular", "Result"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(n=N_STUDENTS, seed=RANDOM_STATE, include_exam_score=False) -> pd.DataFrame:
    """Return a synthetic dataset with the same 10 features + `Result` as always.

    Same (n, seed) -> identical DataFrame. `include_exam_score=True` adds the
    hidden exam score as an extra column for analysis ONLY (it is the target in
    disguise, so using it as a feature would be data leakage).
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < MIN_N:
        raise ValueError(f"n must be an integer >= {MIN_N} (got {n!r}).")
    rng = np.random.default_rng(seed)

    # ---- 1. Hidden traits (never saved) ----------------------------------
    ability = rng.normal(0, 1, n)                       # academic aptitude
    discipline = rng.normal(0, 1, n)                    # conscientiousness / motivation
    ses = rng.normal(0, 1, n)                           # socio-economic background

    # ---- 2. Demographics --------------------------------------------------
    # Mostly first-to-fourth-year undergraduates; a few older (mature) students.
    age = np.clip(np.round(rng.gamma(shape=9.0, scale=0.55, size=n) + 16.5), 17, 25).astype(int)
    gender = rng.choice(["Male", "Female"], n, p=[0.48, 0.52])
    internet = np.where(rng.random(n) < _sigmoid(1.9 + 0.8 * ses), "Yes", "No")
    extra = np.where(rng.random(n) < _sigmoid(-0.2 + 0.3 * ses + 0.2 * discipline), "Yes", "No")

    # ---- 3. Behaviour -----------------------------------------------------
    # Study hours: right-skewed (a few very hard workers, some who barely study).
    study = np.exp(np.log(3.4) + 0.38 * discipline + 0.08 * ses + rng.normal(0, 0.35, n))
    study = np.where(rng.random(n) < 0.02, rng.uniform(0, 0.8, n), study)   # near-zero effort
    study = np.clip(study, 0, 12)

    # Sleep: ~7 h on average, shrinks when studying a lot, slightly shorter for older students.
    sleep = 7.3 - 0.12 * (study - 3.5) - 0.04 * (age - 21) + rng.normal(0, 0.95, n)
    sleep = np.clip(sleep, 3, 10)
    study = np.minimum(study, 24 - sleep)                # physical limit (never binding in practice)

    # Attendance: bunched near the top with a long tail of frequent absentees.
    attendance = 100 - np.exp(2.95 - 0.38 * discipline - 0.18 * ability
                              - 0.10 * (sleep - 7) + rng.normal(0, 0.45, n))
    attendance = np.clip(attendance, 25, 100)

    # Participation: ordered category driven by discipline + ability + noise.
    part_latent = 0.7 * discipline + 0.4 * ability + rng.normal(0, 0.9, n)
    participation = np.select([part_latent < -0.45, part_latent < 0.75], ["Low", "Medium"], "High")

    # ---- 4. Past performance (all BEFORE the exam) ------------------------
    previous = 62 + 11 * ability + 3.5 * discipline + 1.5 * ses + rng.normal(0, 6.5, n)
    assignment = (19 + 0.30 * previous + 12.0 * np.log1p(study) + 0.16 * attendance
                  + 3.0 * discipline + rng.normal(0, 7.5, n))
    previous = np.clip(previous, 0, 100)
    assignment = np.clip(assignment, 0, 100)

    # ---- 5. Hidden exam score -> Pass / Fail ------------------------------
    study_gain = 9.0 * (1 - np.exp(-study / 3.0))          # diminishing returns
    sleep_effect = -1.1 * (sleep - 7.5) ** 2 + 1.0         # sweet spot ~7-8 h
    part_effect = pd.Series(participation).map({"Low": -2.5, "Medium": 0.0, "High": 2.5}).to_numpy()
    # Irregular profiles (very high effort but poor attendance, etc.) are harder to call.
    noise_sd = 6.0 + 2.0 * (np.abs(study - 4) > 3) + 2.0 * (attendance < 60)
    exam = (8
            + 0.38 * previous
            + 0.16 * assignment
            + 0.10 * attendance
            + study_gain
            + sleep_effect
            + part_effect
            + np.where(internet == "Yes", 1.2, 0.0)
            - np.where(extra == "Yes", 0.5, 0.0)            # tiny time cost
            + 3.0 * ability                                  # ability beyond past scores
            + rng.normal(0, 1, n) * noise_sd)
    exam = np.clip(exam, 0, 100)
    result = np.where(exam >= PASS_MARK, "Pass", "Fail")

    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Study_Hours": study.round(1),
        "Attendance": attendance.round(1),
        "Previous_Score": previous.round(1),
        "Assignment_Score": assignment.round(1),
        "Sleep_Hours": sleep.round(1),
        "Participation": participation,
        "Internet_Access": internet,
        "Extracurricular": extra,
        "Result": result,
    })

    # ---- 6. Missing values (never depend on the label) --------------------
    # Slightly more gaps for students without internet (surveys are harder to fill in).
    p_miss = MISSING_RATE * np.where(internet == "No", 1.8, 0.9)
    miss_draw = {col: rng.random(n) < p_miss for col in MISSING_COLUMNS}
    mask = pd.DataFrame(miss_draw)
    too_many = mask.sum(axis=1) > MAX_MISSING_PER_ROW
    mask.loc[too_many, :] = False                           # keep rows usable
    for col in MISSING_COLUMNS:
        df.loc[mask[col].to_numpy(), col] = np.nan

    df = df.astype({"Participation": object})
    _check_dataset(df)
    if include_exam_score:
        df["Exam_Score"] = exam.round(1)                    # analysis only - NOT a feature
    return df


def _check_dataset(df: pd.DataFrame) -> None:
    """Fail loudly if the generator ever produces something the app would reject."""
    from_limits = {"Age": (15, 40), "Study_Hours": (0, 16), "Attendance": (0, 100),
                   "Previous_Score": (0, 100), "Assignment_Score": (0, 100), "Sleep_Hours": (0, 14)}
    for col, (lo, hi) in from_limits.items():
        if not df[col].dropna().between(lo, hi).all():
            raise AssertionError(f"{col} outside {lo}-{hi}")
    if ((df["Study_Hours"] + df["Sleep_Hours"]) > 24).any():
        raise AssertionError("study + sleep exceeds 24 hours")
    if df["Result"].isna().any() or df["Result"].nunique() < 2:
        raise AssertionError("target missing or only one class generated")
    if df.duplicated().any():
        raise AssertionError("duplicate rows generated")


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = generate_dataset()
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(data)} rows to {OUTPUT_PATH}")
    print(data["Result"].value_counts())
