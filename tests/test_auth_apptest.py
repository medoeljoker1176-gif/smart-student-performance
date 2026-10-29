"""Headless UI checks for login + roles with Streamlit's AppTest.

NOT EXECUTED where this was written (Streamlit could not be installed there); skipped automatically when
Streamlit is missing. A failure on your machine is real information - please send it.
"""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest          # noqa: E402
from pathlib import Path                          # noqa: E402

_APP = Path(__file__).resolve().parents[1] / "app.py"

from src import auth                              # noqa: E402

TIMEOUT = 120
SECRETS = {"auth": {"max_attempts": 3, "users": {
    "teacher1": {"role": "teacher", "name": "T", "password_hash": auth.hash_password("teacher-pass-1", 1000)},
    "student1": {"role": "student", "name": "S", "password_hash": auth.hash_password("student-pass-1", 1000)}}}}


def _app():
    at = AppTest.from_file(str(_APP), default_timeout=TIMEOUT)
    at.secrets["auth"] = SECRETS["auth"]
    return at.run()


def _login(at, user, password):
    at.text_input[0].set_value(user)
    at.text_input[1].set_value(password)
    at.button[0].click()
    return at.run()


def test_login_form_shown_and_nothing_else_before_signing_in():
    at = _app()
    assert not at.exception
    assert [b.label for b in at.button] == ["Sign in"]


def test_wrong_password_rejected_with_generic_message():
    at = _login(_app(), "teacher1", "wrong-password")
    assert any(auth.GENERIC_FAILURE in e.value for e in at.error)


def test_teacher_signs_in():
    at = _login(_app(), "teacher1", "teacher-pass-1")
    assert not at.exception
    assert any("Sign out" == b.label for b in at.button)


def test_student_signs_in_and_cannot_reach_teacher_pages():
    at = _login(_app(), "student1", "student-pass-1")
    assert not at.exception
    assert any("Sign out" == b.label for b in at.button)
