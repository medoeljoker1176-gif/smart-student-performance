"""
prepare_uci.py
--------------
Converts the real UCI "Student Performance" files (Cortez & Silva, 2008) into the 10-feature
schema this app uses, and writes data/uci_student_performance.csv.

Get the data (free, CC BY 4.0):  https://archive.ics.uci.edu/dataset/320/student+performance
Unzip it and put  student-mat.csv  and/or  student-por.csv  in  data/raw/  (they use ';' as separator).

Run:   python prepare_uci.py                      (uses every file found in data/raw/)
       python prepare_uci.py path/to/student-mat.csv
Then:  python train_model.py --dataset uci

How UCI columns are mapped (approximations are listed honestly - see docs/DATA_DICTIONARY.md):
  age                 -> Age
  sex (F/M)           -> Gender (Female/Male)
  studytime (1-4)     -> Study_Hours      weekly band midpoint (1, 3.5, 7.5, 12 h/week) / 7 = hours per day
  absences            -> Attendance       100 * (1 - absences / 90), clipped to 0-100 (assumes ~90 lessons)
  G1 (0-20)           -> Previous_Score   x5  (period-1 grade)
  G2 (0-20)           -> Assignment_Score x5  (period-2 grade, used as the "recent work" signal)
  internet (yes/no)   -> Internet_Access
  activities (yes/no) -> Extracurricular
  (no source)         -> Sleep_Hours, Participation  left EMPTY on purpose (never invented)
  G3 (final, 0-20)    -> Result           Pass when G3 >= 10, otherwise Fail (the Portuguese pass mark)
G3 itself is never a feature (that would be data leakage). G1/G2 happen BEFORE the final grade.

Students who appear in both subject files (about 382) are kept ONCE, so the same person can never
land in both the training and the test set.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "uci_student_performance.csv"
PASS_GRADE = 10
ASSUMED_LESSONS = 90
STUDY_BAND_HOURS_PER_WEEK = {1: 1.0, 2: 3.5, 3: 7.5, 4: 12.0}
# Attributes the dataset authors use to recognise the same student in both subject files.
MERGE_KEYS = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
              "Mjob", "Fjob", "reason", "nursery", "internet"]
REQUIRED = ["sex", "age", "studytime", "absences", "G1", "G2", "G3", "internet", "activities"]
OUT_COLUMNS = ["Age", "Gender", "Study_Hours", "Attendance", "Previous_Score", "Assignment_Score",
               "Sleep_Hours", "Participation", "Internet_Access", "Extracurricular", "Result"]


class UCIFormatError(Exception):
    """The input file is not the UCI student-performance format."""


def read_uci_file(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise UCIFormatError(f"File not found: {path}")
    try:
        df = pd.read_csv(path, sep=";", skipinitialspace=True)
        df.columns = [str(c).strip().strip('"') for c in df.columns]
    except Exception as exc:                                    # noqa: BLE001
        raise UCIFormatError(f"Could not read {path.name}: {exc}") from exc
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise UCIFormatError(f"{path.name} is missing UCI column(s): {', '.join(missing)}. "
                             "Use the original student-mat.csv / student-por.csv (';'-separated).")
    if df.empty:
        raise UCIFormatError(f"{path.name} has no rows.")
    return df


def combine(frames: list) -> pd.DataFrame:
    """Stack the subject files and keep each student once (first file wins)."""
    df = pd.concat(frames, ignore_index=True)
    keys = [k for k in MERGE_KEYS if k in df.columns]
    return df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True) if len(frames) > 1 else df


def convert(df: pd.DataFrame) -> pd.DataFrame:
    """UCI rows -> the app's schema. Rows with unusable values are dropped, never guessed."""
    d = df.copy()
    for col in ["age", "studytime", "absences", "G1", "G2", "G3"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    for col in ["sex", "internet", "activities"]:
        d[col] = d[col].astype(str).str.strip().str.strip('"').str.lower()
    ok = (d[["age", "studytime", "absences", "G1", "G2", "G3"]].notna().all(axis=1)
          & d["sex"].isin(["f", "m"]) & d["internet"].isin(["yes", "no"])
          & d["activities"].isin(["yes", "no"])
          & d["studytime"].isin(STUDY_BAND_HOURS_PER_WEEK)
          & d[["G1", "G2", "G3"]].apply(lambda s: s.between(0, 20)).all(axis=1)
          & (d["absences"] >= 0) & d["age"].between(15, 40))
    d = d.loc[ok]
    out = pd.DataFrame({
        "Age": d["age"].astype(int),
        "Gender": np.where(d["sex"] == "f", "Female", "Male"),
        "Study_Hours": (d["studytime"].map(STUDY_BAND_HOURS_PER_WEEK) / 7).round(2),
        "Attendance": np.clip(100 * (1 - d["absences"] / ASSUMED_LESSONS), 0, 100).round(1),
        "Previous_Score": (d["G1"] * 5).astype(float),
        "Assignment_Score": (d["G2"] * 5).astype(float),
        "Sleep_Hours": np.nan,                       # not in UCI: left empty on purpose
        "Participation": np.nan,                     # not in UCI: left empty on purpose
        "Internet_Access": np.where(d["internet"] == "yes", "Yes", "No"),
        "Extracurricular": np.where(d["activities"] == "yes", "Yes", "No"),
        "Result": np.where(d["G3"] >= PASS_GRADE, "Pass", "Fail"),
    })
    out = out[OUT_COLUMNS].reset_index(drop=True)
    if out["Result"].nunique() < 2:
        raise UCIFormatError("After conversion only one class (Pass or Fail) is left - cannot train.")
    return out


def prepare(paths=None, output=OUTPUT_PATH) -> pd.DataFrame:
    if not paths:
        paths = sorted(RAW_DIR.glob("student-*.csv"))
    if not paths:
        raise UCIFormatError(f"No UCI files found. Put student-mat.csv and/or student-por.csv in {RAW_DIR}.")
    out = convert(combine([read_uci_file(p) for p in paths]))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return out


if __name__ == "__main__":
    try:
        result = prepare([Path(p) for p in sys.argv[1:]] or None)
    except UCIFormatError as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"Saved {len(result)} students to {OUTPUT_PATH}")
    print(result["Result"].value_counts())
