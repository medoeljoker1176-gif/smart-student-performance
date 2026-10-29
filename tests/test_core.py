"""
Core tests (no Streamlit needed).  Run from the project root:

    python -m tests.test_core          # plain Python
    pytest -q                          # if you have pytest installed

They check the parts that must be right: validation, leakage-free training,
risk levels, Shapley maths, recommendations, history, fairness, model loading.
"""
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import config, history
from src.config import FEATURES, SAMPLE_PROFILES
from src.explainability import explain_student, shapley_values
from src.fairness import fairness_summary
from src.model_training import select_final_model
from src.prediction import (ModelLoadError, analyze_student, assess_risk, load_bundle, make_predict_fn,
                            p_pass_for, performance_score, predict_student)
from src.preprocessing import clean_data, data_quality_report, load_dataset
from src.recommendations import get_prioritized_recommendations, get_recommendations
from src.validation import DataError, validate_dataframe, validate_student_input

AVERAGE, STRONG, RISKY = (SAMPLE_PROFILES[k] for k in ("Average student", "Strong student", "At-risk student"))


def _bundle():
    return load_bundle()          # requires `python train_model.py` to have been run


# ------------------------------------------------------------- validation
def test_valid_student_passes_validation():
    assert validate_student_input(AVERAGE).ok


def test_invalid_values_are_rejected():
    for field, bad in [("Age", 5), ("Attendance", 120), ("Previous_Score", -3),
                       ("Study_Hours", 30), ("Sleep_Hours", "abc"), ("Participation", "Huge")]:
        res = validate_student_input({**AVERAGE, field: bad})
        assert not res.ok and res.errors, f"{field}={bad} should be rejected"


def test_study_plus_sleep_cannot_exceed_24h():
    res = validate_student_input({**AVERAGE, "Study_Hours": 14, "Sleep_Hours": 12})
    assert not res.ok and any("24" in e for e in res.errors)


def test_missing_field_and_nan_rejected():
    incomplete = {k: v for k, v in AVERAGE.items() if k != "Attendance"}
    assert not validate_student_input(incomplete).ok
    assert not validate_student_input({**AVERAGE, "Attendance": float("nan")}).ok


def test_out_of_training_range_gives_warning_not_error():
    res = validate_student_input({**AVERAGE, "Age": 38}, {"Age": (17, 25)})
    assert res.ok and res.warnings


def test_dataset_validation_detects_problems():
    df = load_dataset()
    assert validate_dataframe(df) == []
    assert validate_dataframe(df.drop(columns=["Attendance"]))
    assert validate_dataframe(df.assign(Gender="Robot"))


def test_dataset_validation_rejects_non_finite_numeric_values():
    df = load_dataset().copy()
    df.loc[df.index[0], "Attendance"] = float("inf")
    assert any("non-finite" in issue for issue in validate_dataframe(df))


def test_missing_dataset_gives_friendly_error():
    try:
        load_dataset(Path("does/not/exist.csv"))
    except DataError as e:
        assert "not found" in str(e)
    else:
        raise AssertionError("expected DataError")


# ------------------------------------------------------------- data
def test_clean_data_keeps_missing_feature_values_for_the_pipeline():
    raw = load_dataset()
    clean = clean_data(raw)
    assert clean.isna().sum().sum() == raw.isna().sum().sum(), "imputation must happen inside the Pipeline"
    report = data_quality_report(raw)
    assert report["n_records"] == len(raw) and report["duplicates"] == 0


def test_clean_data_drops_impossible_values_instead_of_silently_clipping():
    raw = load_dataset().head(3).copy()
    valid = raw.iloc[[0]].copy()
    invalid_score = raw.iloc[[1]].copy()
    invalid_score.loc[:, "Attendance"] = 150
    impossible_day = raw.iloc[[2]].copy()
    impossible_day.loc[:, "Study_Hours"] = 16
    impossible_day.loc[:, "Sleep_Hours"] = 14
    combined = pd.concat([valid, invalid_score, impossible_day], ignore_index=True)
    cleaned = clean_data(combined)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["Attendance"] == valid.iloc[0]["Attendance"]


# ------------------------------------------------------------- model + prediction
def test_model_bundle_loads_and_predicts():
    b = _bundle()
    for student in (AVERAGE, STRONG, RISKY):
        assert 0 <= p_pass_for(b, student) <= 1
    assert p_pass_for(b, STRONG) > p_pass_for(b, AVERAGE) > p_pass_for(b, RISKY)


def test_normal_and_high_risk_students():
    b = _bundle()
    assert predict_student(b, STRONG).risk["level"] == "Low Risk"
    high = predict_student(b, RISKY)
    assert high.risk["level"] == "High Risk" and high.label == "Fail"


def test_missing_and_corrupt_model_files_are_handled():
    tmp = Path(tempfile.mkdtemp())
    try:
        for path in (tmp / "nope.pkl",):
            try:
                load_bundle(path)
            except ModelLoadError as e:
                assert "not found" in str(e)
            else:
                raise AssertionError("expected ModelLoadError")
        bad = tmp / "bad.pkl"; bad.write_bytes(b"this is not a model")
        try:
            load_bundle(bad)
        except ModelLoadError:
            pass
        else:
            raise AssertionError("expected ModelLoadError")
    finally:
        shutil.rmtree(tmp)


def test_risk_thresholds():
    levels = [assess_risk(p)["level"] for p in (0.0, 0.49, 0.50, 0.74, 0.75, 1.0)]
    assert levels == ["High Risk", "High Risk", "Moderate Risk", "Moderate Risk", "Low Risk", "Low Risk"]


def test_performance_score_range_and_order():
    scores = [performance_score(s)[0] for s in (RISKY, AVERAGE, STRONG)]
    assert all(0 <= s <= 100 for s in scores) and scores == sorted(scores)
    best = {**STRONG, "Previous_Score": 100, "Attendance": 100, "Assignment_Score": 100,
            "Study_Hours": 8, "Participation": "High", "Sleep_Hours": 7.5}
    assert performance_score(best)[0] == 100.0


def test_model_selection_prefers_simpler_model_when_tied():
    cv = {"A": {"roc_auc": {"mean": 0.90, "se": 0.01}}, "B": {"roc_auc": {"mean": 0.905, "se": 0.01}},
          "C": {"roc_auc": {"mean": 0.80, "se": 0.01}}}
    assert select_final_model(cv, ["A", "B", "C"])[0] == "A"
    assert select_final_model(cv, ["B", "A", "C"])[0] == "B"


def test_analyze_student_end_to_end():
    out = analyze_student(_bundle(), RISKY)
    assert out["result"].risk["level"] == "High Risk" and out["recs"] and len(out["explanation"]["table"]) == len(FEATURES)


# ------------------------------------------------------------- explanations
def test_shapley_values_add_up_to_prediction_minus_baseline():
    b = _bundle()
    fn = make_predict_fn(b)
    for student in (AVERAGE, RISKY):
        exp = explain_student(fn, student, b["background"])
        assert np.isclose(exp["table"]["Contribution"].sum() / 100, exp["prediction"] - exp["base"], atol=1e-9)
        assert np.isclose(exp["prediction"], p_pass_for(b, student))
        assert set(exp["table"]["Feature"]) == set(FEATURES)


def test_shapley_is_exact_for_an_additive_model():
    import src.explainability as ex
    feats = ["a", "b", "c"]
    old, ex.NUMERIC_FEATURES = ex.NUMERIC_FEATURES, feats
    try:
        bg = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 3)), columns=feats)
        x = pd.Series({"a": 1.0, "b": 2.0, "c": -1.0})
        phi, base, pred = shapley_values(lambda d: 0.1 * d.a + 0.2 * d.b - 0.3 * d.c, x, bg, feats)
        assert np.allclose(phi, [0.1 * (1 - bg.a.mean()), 0.2 * (2 - bg.b.mean()), -0.3 * (-1 - bg.c.mean())])
    finally:
        ex.NUMERIC_FEATURES = old


def test_low_attendance_pushes_probability_down():
    b = _bundle()
    exp = explain_student(make_predict_fn(b), RISKY, b["background"])
    row = exp["table"].set_index("Feature").loc["Attendance"]
    assert row["Contribution"] < 0


# ------------------------------------------------------------- recommendations
def test_recommendations_depend_on_inputs():
    b = _bundle()
    fn = lambda s: p_pass_for(b, s)
    assert get_prioritized_recommendations(STRONG, fn) == []
    recs = get_prioritized_recommendations(RISKY, fn)
    assert len(recs) >= 4 and all(r.reason for r in recs)
    assert recs[0].rank_score >= recs[-1].rank_score
    only_attendance = get_prioritized_recommendations({**STRONG, "Attendance": 50}, fn)
    assert [r.feature for r in only_attendance] == ["Attendance"] and only_attendance[0].gain > 0
    assert get_recommendations(STRONG)          # backward-compatible helper still returns text


# ------------------------------------------------------------- history
def test_history_add_read_clear():
    tmp = Path(tempfile.mkdtemp())
    old = history.HISTORY_PATH
    history.HISTORY_PATH = tmp / "h.csv"
    try:
        assert history.read_history().empty
        result = SimpleNamespace(label="Pass", p_pass=0.8, risk={"level": "Low Risk"}, score=70.0)
        history.add_record(AVERAGE, result, "Sara"); history.add_record(RISKY, result, "")
        df = history.read_history()
        assert len(df) == 2 and df.Student_ID.tolist() == ["Sara", ""]
        history.HISTORY_PATH.write_text("garbage,not,a,history\n1,2,3,4\n")
        assert history.read_history().empty            # corrupt file must not crash
        history.clear_history()
        assert not history.HISTORY_PATH.exists()
    finally:
        history.HISTORY_PATH = old
        shutil.rmtree(tmp)


# ------------------------------------------------------------- fairness
def test_fairness_summary_on_saved_predictions():
    oof = pd.read_csv(config.OOF_PATH)
    out = fairness_summary(oof, "Gender")
    assert set(out["table"]["Gender"]) == {"Male", "Female"} and "gaps" in out
    assert (out["table"]["Students"].sum()) == len(oof)


def test_fairness_metrics_are_correct_on_a_toy_example():
    toy = pd.DataFrame({"Gender": ["M"] * 4 + ["F"] * 4,
                        "y_true": [1, 1, 0, 0, 1, 1, 0, 0], "y_pred": [1, 0, 1, 0, 1, 1, 0, 0]})
    t = fairness_summary(toy).get("table").set_index("Gender")
    assert t.loc["M", "Recall (TPR)"] == 0.5 and t.loc["M", "False positive rate"] == 0.5
    assert t.loc["F", "Recall (TPR)"] == 1.0 and t.loc["F", "False positive rate"] == 0.0


def test_no_data_leakage_pipeline_learns_only_from_training_rows():
    """The imputer inside the saved pipeline must equal the median of TRAINING rows only."""
    from sklearn.model_selection import train_test_split
    from src.preprocessing import split_xy
    X, y = split_xy(clean_data(load_dataset()))
    X_tr, X_te, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)
    imputer = _bundle()["pipeline"].named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
    expected = X_tr[config.NUMERIC_FEATURES].median().to_numpy()
    assert np.allclose(imputer.statistics_, expected)
    assert not np.allclose(X[config.NUMERIC_FEATURES].median().to_numpy(), expected)   # full-data medians differ


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as exc:                       # noqa: BLE001
            failed += 1; print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
