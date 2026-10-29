"""Headless UI checks with Streamlit's official AppTest framework.

NOT EXECUTED in the environment where this project was prepared (Streamlit could not be installed there).
They are skipped automatically when Streamlit is missing. If one fails on your machine, please treat the
failure as real information and see docs/TEST_REPORT.md for the manual UI checklist.
"""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest          # noqa: E402
from pathlib import Path                          # noqa: E402

_APP = Path(__file__).resolve().parents[1] / "app.py"

TIMEOUT = 120


def _app():
    return AppTest.from_file(str(_APP), default_timeout=TIMEOUT)


def test_app_starts_without_exception():
    at = _app().run()
    assert not at.exception


def _prediction_script(root):
    import sys
    sys.path.insert(0, root)
    from views import prediction
    prediction.render()


def test_valid_prediction_shows_result_and_saves_history(tmp_path, monkeypatch):
    from src import history
    monkeypatch.setattr(history, "HISTORY_PATH", tmp_path / "h.csv")
    # The Prediction page is rendered on its own: AppTest cannot switch to st.Page callables by url_path.
    at = AppTest.from_function(_prediction_script, args=(str(_APP.parent),), default_timeout=TIMEOUT).run()
    assert not at.exception
    predict = [b for b in at.button if b.label == "Predict performance"]
    assert predict, "Predict button not found"
    predict[0].click()
    at = at.run()
    assert not at.exception
    assert len(history.read_history()) == 1
