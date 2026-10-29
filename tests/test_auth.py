"""Login logic: hashing, user loading, roles, lockout, session expiry (no Streamlit needed)."""
import pytest

from src import auth

FAST = 1000                                            # PBKDF2 iterations: fast for tests only
PAGES = ["dashboard", "prediction", "explanation", "analytics", "recommendations", "compare",
         "history", "batch-analysis", "model-performance", "about"]


def _users(**extra):
    cfg = {"users": {
        "Teacher1": {"role": "teacher", "name": "Dr T", "password_hash": auth.hash_password("teacher-pass-1", FAST)},
        "student1": {"role": "student", "password_hash": auth.hash_password("student-pass-1", FAST)},
        **extra}}
    return auth.load_users(cfg)


def test_hash_is_salted_and_verifies():
    a, b = auth.hash_password("correct horse", FAST), auth.hash_password("correct horse", FAST)
    assert a != b and a.startswith("pbkdf2_sha256$1000$")
    assert auth.verify_password("correct horse", a) and not auth.verify_password("wrong horse", a)
    assert "correct horse" not in a


@pytest.mark.parametrize("bad", ["", "short", None, "x" * 300])
def test_weak_or_huge_passwords_rejected(bad):
    with pytest.raises(ValueError):
        auth.hash_password(bad, FAST)


@pytest.mark.parametrize("stored", ["", None, "plain-text", "pbkdf2_sha256$x$y$z", "md5$1$00$00",
                                    "pbkdf2_sha256$0$00$00", "pbkdf2_sha256$1000$zz$zz", "a$b$c$d$e"])
def test_malformed_hashes_never_verify_or_raise(stored):
    assert auth.verify_password("anything-long", stored) is False


def test_load_users_normalises_and_skips_bad_entries():
    users, problems = _users(admin={"role": "admin", "password_hash": auth.hash_password("whatever-1", FAST)},
                             nohash={"role": "teacher"},
                             plain={"role": "teacher", "password_hash": "mypassword"})
    assert set(users) == {"teacher1", "student1"}            # username lower-cased, bad ones skipped
    assert len(problems) == 3 and users["student1"].name == "student1"
    assert auth.load_users(None) == ({}, []) and auth.load_users({}) == ({}, [])


def test_role_pages():
    teacher = auth.pages_for("teacher", PAGES)
    student = auth.pages_for("student", PAGES)
    assert teacher == PAGES
    assert student == ["prediction", "explanation", "recommendations", "about"]
    for private in ("history", "batch-analysis", "dashboard", "compare", "model-performance", "analytics"):
        assert private not in student
    assert auth.pages_for("hacker", PAGES) == [] and auth.pages_for(None, PAGES) == []
    assert auth.ROLE_DEFAULT_PAGE["student"] in student


def test_login_success_and_generic_failures():
    users, _ = _users()
    a = auth.Authenticator(users, max_attempts=5)
    ok = a.login("TEACHER1 ", "teacher-pass-1", 0)
    assert ok.ok and ok.user.role == "teacher"
    wrong, unknown = a.login("teacher1", "nope", 1), a.login("ghost", "nope", 1)
    assert not wrong.ok and not unknown.ok and wrong.message == unknown.message == auth.GENERIC_FAILURE
    assert not a.login(None, None, 2).ok and not a.login("", "", 2).ok


def test_lockout_after_max_attempts_then_expires():
    users, _ = _users()
    a = auth.Authenticator(users, max_attempts=3, lockout_minutes=5)
    for i in range(3):
        assert not a.login("student1", "bad", i).ok
    locked = a.login("student1", "student-pass-1", 10)           # right password, but locked
    assert not locked.ok and locked.locked
    assert a.login("teacher1", "teacher-pass-1", 10).ok          # other accounts unaffected
    assert a.login("student1", "student-pass-1", 10 + 5 * 60 + 1).ok   # lock expired


def test_success_resets_failure_counter_and_unknown_users_lock_too():
    users, _ = _users()
    a = auth.Authenticator(users, max_attempts=3)
    a.login("student1", "bad", 0); a.login("student1", "bad", 1)
    assert a.login("student1", "student-pass-1", 2).ok
    a.login("student1", "bad", 3); a.login("student1", "bad", 4)
    assert not a.login("student1", "bad", 5).ok               # third failure after the reset...
    assert a.login("student1", "student-pass-1", 6).locked    # ...locks the account
    for i in range(3):
        a.login("ghost", "bad", i)
    assert a.login("ghost", "bad", 5).locked                      # no username enumeration via lockout


def test_session_expiry():
    assert auth.session_expired(None, 100)
    assert not auth.session_expired(100, 100 + 29 * 60)
    assert auth.session_expired(100, 100 + 31 * 60)
    assert auth.session_expired(100, 100 + 61, minutes=1)


def test_old_failures_expire():
    from src.auth import Authenticator, hash_password
    users = {"a": __import__("src.auth", fromlist=["User"]).User("a", "student", "A", hash_password("correct-horse"))}
    au = Authenticator(users, max_attempts=3, lockout_minutes=1)
    for t in (0, 1):
        au.login("a", "wrong-pass1", t)
    assert not au.login("a", "wrong-pass1", 1000).locked      # 3rd failure, but long after the first two
    assert au.login("a", "correct-horse", 1001).ok
