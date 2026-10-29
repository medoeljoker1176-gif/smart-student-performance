"""UCI adapter, dataset profiles and training on a second dataset (fixture data, not the real UCI files)."""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from uci_fixture import make_uci_frame, write_uci_file          # noqa: E402

import prepare_uci                                               # noqa: E402
from src import config                                           # noqa: E402
from src.config import FEATURES, INPUT_LIMITS                    # noqa: E402
from src.validation import validate_dataframe                    # noqa: E402


@pytest.fixture(scope="module")
def uci_csv(tmp_path_factory):
    d = tmp_path_factory.mktemp("uci")
    write_uci_file(d / "student-mat.csv", make_uci_frame(300, seed=3))
    prepare_uci.prepare([d / "student-mat.csv"], output=d / "out.csv")
    return d / "out.csv"


def test_dataset_registry_keeps_default_paths():
    p = config.dataset_paths("synthetic")
    assert p["data"] == config.DATA_PATH and p["model"] == config.MODEL_PATH and p["metrics"] == config.METRICS_PATH
    assert config.dataset_paths("uci")["model"] != config.MODEL_PATH
    with pytest.raises(ValueError):
        config.dataset_paths("nope")


def test_conversion_matches_app_schema_and_mapping(uci_csv):
    out = pd.read_csv(uci_csv)
    raw = make_uci_frame(300, seed=3)
    assert set(out.columns) == set(FEATURES) | {"Result"}
    assert validate_dataframe(out) == []
    for col, (lo, hi) in INPUT_LIMITS.items():
        assert out[col].dropna().between(lo, hi).all()
    assert out["Sleep_Hours"].isna().all() and out["Participation"].isna().all()   # never invented
    assert (out["Previous_Score"] == raw["G1"] * 5).all()
    assert (out["Result"] == ("Pass")).sum() == (raw["G3"] >= 10).sum()          # pass mark = 10/20
    assert set(out["Gender"]) <= {"Female", "Male"}


def test_final_grade_is_never_a_feature(uci_csv):
    out = pd.read_csv(uci_csv)
    assert not any(c in out.columns for c in ("G3", "G2", "G1", "Final_Grade"))


def test_students_in_both_subject_files_are_kept_once(tmp_path):
    mat = make_uci_frame(200, seed=5)
    por = pd.concat([mat.head(120).assign(G1=lambda d: d.G1), make_uci_frame(80, seed=9)], ignore_index=True)
    write_uci_file(tmp_path / "student-mat.csv", mat)
    write_uci_file(tmp_path / "student-por.csv", por)
    out = prepare_uci.prepare([tmp_path / "student-mat.csv", tmp_path / "student-por.csv"], output=tmp_path / "o.csv")
    assert 200 < len(out) <= 280                         # 200 + 80 new students; the 120 overlapping ones counted once


@pytest.mark.parametrize("breaker", ["missing_col", "empty", "one_class", "missing_file"])
def test_bad_inputs_give_friendly_errors(tmp_path, breaker):
    f = make_uci_frame(60, seed=1)
    if breaker == "missing_col":
        f = f.drop(columns=["G2"])
    elif breaker == "one_class":
        f["G3"] = 15
    elif breaker == "empty":
        f = f.head(0)
    path = tmp_path / ("nope.csv" if breaker == "missing_file" else "student-mat.csv")
    if breaker != "missing_file":
        write_uci_file(path, f)
    with pytest.raises(prepare_uci.UCIFormatError):
        prepare_uci.prepare([path], output=tmp_path / "o.csv")


def test_rows_with_impossible_values_are_dropped_not_guessed(tmp_path):
    f = make_uci_frame(100, seed=2)
    f.loc[0, "G2"] = 99
    f.loc[1, "studytime"] = 9
    f.loc[2, "sex"] = "X"
    write_uci_file(tmp_path / "student-mat.csv", f)
    out = prepare_uci.prepare([tmp_path / "student-mat.csv"], output=tmp_path / "o.csv")
    assert len(out) == 97


def test_training_and_prediction_work_on_second_dataset(uci_csv, tmp_path, monkeypatch):
    monkeypatch.setitem(config.DATASETS, "uci", {**config.DATASETS["uci"], "data": uci_csv,
                        "model": tmp_path / "m" / "model.pkl", "reports": tmp_path / "rep"})
    from src.model_training import main
    from src.prediction import analyze_student, load_bundle
    metrics = main(quiet=True, dataset="uci")
    assert metrics["dataset"] == "uci" and set(metrics["unavailable_features"]) == {"Sleep_Hours", "Participation"}
    assert (tmp_path / "rep" / "metrics.json").exists() and (tmp_path / "rep" / "oof_predictions.csv").exists()
    assert 0.5 < metrics["test_metrics"][metrics["final_model"]]["roc_auc"] <= 1
    bundle = load_bundle(tmp_path / "m" / "model.pkl")
    strong = analyze_student(bundle, config.SAMPLE_PROFILES["Strong student"])
    weak = analyze_student(bundle, config.SAMPLE_PROFILES["At-risk student"])
    assert strong["result"].p_pass > weak["result"].p_pass
    # the model ignores features the dataset does not have
    a = dict(config.SAMPLE_PROFILES["Average student"]); b = {**a, "Sleep_Hours": 3.0, "Participation": "Low"}
    from src.prediction import p_pass_for
    assert p_pass_for(bundle, a) == pytest.approx(p_pass_for(bundle, b))
    # the default dataset's real artifacts were not touched
    assert json.loads(config.METRICS_PATH.read_text())["final_model"]
