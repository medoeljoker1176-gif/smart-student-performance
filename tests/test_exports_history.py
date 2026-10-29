"""CSV export safety and prediction-history persistence (existing local-CSV mechanism)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from src import history
from src.config import SAMPLE_PROFILES
from src.exports import dataframe_to_safe_csv_bytes

RESULT = SimpleNamespace(label="Pass", p_pass=0.81, risk={"level": "Low Risk"}, score=70.0)
AVERAGE = SAMPLE_PROFILES["Average student"]


@pytest.fixture()
def hist(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_PATH", tmp_path / "h.csv")
    return history.HISTORY_PATH


def test_export_keeps_numbers_numeric_and_unicode_intact():
    frame = pd.DataFrame({"name": ["أحمد", "-5 apples", "@SUM(1)", "ok"], "n": [-1, 2, 3, 4]})
    text = dataframe_to_safe_csv_bytes(frame).decode("utf-8-sig")
    assert "أحمد" in text and "'-5 apples" in text and "'@SUM(1)" in text
    assert ",-1\n" in text
    assert dataframe_to_safe_csv_bytes(frame).startswith(b"\xef\xbb\xbf")     # Excel opens UTF-8 correctly


def test_export_of_empty_frame_is_just_a_header():
    assert dataframe_to_safe_csv_bytes(pd.DataFrame(columns=["a", "b"])).decode("utf-8-sig").strip() == "a,b"


def test_history_missing_file_is_empty(hist):
    assert not hist.exists()
    df = history.read_history()
    assert df.empty and list(df.columns) == history.COLUMNS
    assert history.summarize_history(df)["n"] == 0


def test_history_roundtrip_with_arabic_id_and_summary(hist):
    history.add_record(AVERAGE, RESULT, "طالب-١")
    history.add_record(AVERAGE, SimpleNamespace(label="Fail", p_pass=0.2, risk={"level": "High Risk"}, score=40.0), "")
    df = history.read_history()
    assert df["Student_ID"].tolist() == ["طالب-١", ""]
    s = history.summarize_history(df)
    assert s["n"] == 2 and s["risk_counts"] == {"Low Risk": 1, "High Risk": 1}
    assert s["predicted_pass_share"] == 0.5 and len(s["recent"]) == 2


def test_history_formula_id_is_neutralised_on_export(hist):
    history.add_record(AVERAGE, RESULT, "=HYPERLINK(\"http://x\")")
    exported = dataframe_to_safe_csv_bytes(history.read_history()).decode("utf-8-sig")
    assert "'=HYPERLINK" in exported


def test_history_ignores_unusable_rows_instead_of_crashing(hist):
    history.add_record(AVERAGE, RESULT, "a")
    with open(hist, "a", encoding="utf-8") as fh:
        fh.write("2026-01-01T00:00:00,b,20,Male,4,80,60,60,7,Medium,Yes,Yes,Pass,not-a-number,Low Risk,70\n")
        fh.write("2026-01-01T00:00:00,c,20,Male,4,80,60,60,7,Medium,Yes,Yes,Pass,7.5,Low Risk,70\n")
    df = history.read_history()
    assert df["Student_ID"].tolist() == ["a"]


def test_clear_history_when_nothing_exists_is_safe(hist):
    history.clear_history()
    assert not hist.exists()


# ---------------------------------------------------------------- Firestore backend (fake client)
class _FakeDoc:
    def __init__(self, store, key):
        self._store, self._key = store, key
        self.reference = self

    def to_dict(self):
        return dict(self._store[self._key])

    def delete(self):
        del self._store[self._key]


class _FakeCollection:
    def __init__(self, store):
        self._store = store

    def add(self, row):
        self._store[len(self._store)] = dict(row)

    def stream(self):
        return [_FakeDoc(self._store, k) for k in list(self._store)]


class _FakeClient:
    def __init__(self):
        self.store = {}

    def collection(self, name):
        return _FakeCollection(self.store)


@pytest.fixture()
def fake_firestore(monkeypatch):
    client = _FakeClient()
    monkeypatch.setenv("HISTORY_BACKEND", "firestore")
    monkeypatch.setattr(history, "_firestore", lambda: client)
    return client


def test_backend_defaults_to_csv_and_ignores_unknown_values(monkeypatch):
    monkeypatch.delenv("HISTORY_BACKEND", raising=False)
    assert history.backend() == "csv"
    monkeypatch.setenv("HISTORY_BACKEND", "nonsense")
    assert history.backend() == "csv"


def test_firestore_backend_roundtrip_and_clear(fake_firestore):
    assert history.read_history().empty
    history.add_record(AVERAGE, RESULT, "S-1")
    history.add_record(AVERAGE, RESULT, "S-2")
    df = history.read_history()
    assert list(df["Student_ID"]) == ["S-1", "S-2"] and list(df.columns) == history.COLUMNS
    history.clear_history()
    assert history.read_history().empty


def test_firestore_read_failure_returns_empty_frame(monkeypatch):
    class Broken:
        def collection(self, name):
            raise RuntimeError("unreachable")
    monkeypatch.setenv("HISTORY_BACKEND", "firestore")
    monkeypatch.setattr(history, "_firestore", lambda: Broken())
    assert history.read_history().empty
