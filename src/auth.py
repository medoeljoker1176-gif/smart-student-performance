"""
auth.py
-------
Login logic WITHOUT any Streamlit code, so it can be unit-tested and reused.

  * Passwords are never stored. Streamlit secrets hold a salted PBKDF2-HMAC-SHA256 hash
    ("pbkdf2_sha256$<iterations>$<salt hex>$<hash hex>"); create one with  python tools/make_password_hash.py
  * Two roles: "teacher" (every page) and "student" (prediction, explanation, recommendations, about).
  * Brute-force protection: after MAX_ATTEMPTS wrong passwords for a username, that username is locked
    for LOCKOUT_MINUTES (counted per username, so opening new browser tabs does not reset it).
  * Unknown usernames and wrong passwords give the SAME message and take about the same time,
    so nobody can discover which usernames exist.
  * Sessions expire after SESSION_MINUTES without activity.

Limits (be honest in the report): the lockout counter lives in server memory and resets when the app
restarts; there is no password reset or MFA. For a real deployment use an identity provider (OIDC).
"""
import hashlib
import hmac
import secrets as _secrets
import threading
from dataclasses import dataclass

ALGORITHM = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 600_000          # OWASP 2023 guidance for PBKDF2-HMAC-SHA256
MIN_PASSWORD_LENGTH = 8
MAX_USERNAME_LENGTH = 64
MAX_PASSWORD_LENGTH = 256              # stops absurdly long inputs from being hashed
MAX_ATTEMPTS = 5
LOCKOUT_MINUTES = 5
SESSION_MINUTES = 30
ROLES = ("teacher", "student")
GENERIC_FAILURE = "Wrong username or password."

# Pages each role may open (keys = url_path used in app.py). Anything else is never registered.
ROLE_PAGES = {
    "teacher": None,                                                  # None = every page
    "student": ("prediction", "explanation", "recommendations", "about"),
}
ROLE_DEFAULT_PAGE = {"teacher": "dashboard", "student": "prediction"}


# ------------------------------------------------------------------ passwords
def hash_password(password: str, iterations: int = DEFAULT_ITERATIONS, salt: bytes | None = None) -> str:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters.")
    salt = salt if salt is not None else _secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{ALGORITHM}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check. Any malformed input simply returns False (never raises)."""
    try:
        algo, iterations, salt_hex, hash_hex = str(stored).split("$")
        iterations = int(iterations)
        if algo != ALGORITHM or iterations < 1 or iterations > 10_000_000:
            return False
        if not isinstance(password, str) or len(password) > MAX_PASSWORD_LENGTH:
            return False
        expected = bytes.fromhex(hash_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), iterations)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, AttributeError):
        return False


# ------------------------------------------------------------------ users
@dataclass(frozen=True)
class User:
    username: str
    role: str
    name: str
    password_hash: str


def load_users(config) -> tuple:
    """Parse the [auth] section of the secrets. Returns (users dict, list of problems).

    Invalid entries are skipped and reported, they never crash the app and never become accounts."""
    users, problems = {}, []
    table = (config or {}).get("users") if hasattr(config, "get") else None
    if not table:
        return users, problems
    for username, entry in dict(table).items():
        key = str(username).strip().lower()
        if not key or len(key) > MAX_USERNAME_LENGTH:
            problems.append(f"Skipped a user with an invalid username ({str(username)[:20]!r}).")
            continue
        entry = dict(entry) if hasattr(entry, "keys") else {}
        role, stored = str(entry.get("role", "")).lower(), str(entry.get("password_hash", ""))
        if role not in ROLES:
            problems.append(f"Skipped '{key}': role must be one of {', '.join(ROLES)}.")
        elif stored.split("$")[0] != ALGORITHM or stored.count("$") != 3:
            problems.append(f"Skipped '{key}': password_hash is missing or not in the expected format.")
        else:
            users[key] = User(key, role, str(entry.get("name") or key), stored)
    return users, problems


def pages_for(role: str, all_keys: list) -> list:
    """Page keys this role may open, in the original order. Unknown roles get nothing."""
    if role not in ROLE_PAGES:
        return []
    allowed = ROLE_PAGES[role]
    return list(all_keys) if allowed is None else [k for k in all_keys if k in allowed]


def session_expired(last_seen: float | None, now: float, minutes: float = SESSION_MINUTES) -> bool:
    return last_seen is None or (now - last_seen) > minutes * 60


# ------------------------------------------------------------------ authenticator
@dataclass
class LoginResult:
    ok: bool
    user: User | None = None
    message: str = ""
    locked: bool = False


class Authenticator:
    """Checks logins and remembers failures (thread-safe; one shared instance per server)."""

    def __init__(self, users: dict, max_attempts: int = MAX_ATTEMPTS, lockout_minutes: float = LOCKOUT_MINUTES):
        self.users = dict(users)
        self.max_attempts = max_attempts
        self.lockout_seconds = lockout_minutes * 60
        self._failures: dict = {}                    # username -> (count, first_failure_time)
        self._locked_until: dict = {}
        self._lock = threading.Lock()
        self._dummy_hash = hash_password("not-a-real-password", iterations=1000)

    def _locked_for(self, key: str, now: float) -> float:
        until = self._locked_until.get(key, 0.0)
        if until and now >= until:                   # lock expired: forget it
            self._locked_until.pop(key, None)
            self._failures.pop(key, None)
            return 0.0
        return max(0.0, until - now)

    def login(self, username: str, password: str, now: float) -> LoginResult:
        key = str(username or "").strip().lower()[:MAX_USERNAME_LENGTH]
        password = password if isinstance(password, str) else ""
        with self._lock:
            remaining = self._locked_for(key, now)
        if remaining > 0:
            minutes = int(remaining // 60) + 1
            return LoginResult(False, None, f"Too many failed attempts. Try again in about {minutes} minute(s).", True)

        user = self.users.get(key)
        # Always do one hash check, even for unknown users, so timing does not reveal valid usernames.
        good = verify_password(password, user.password_hash if user else self._dummy_hash)
        if user and good:
            with self._lock:
                self._failures.pop(key, None)
            return LoginResult(True, user)

        with self._lock:
            count, first = self._failures.get(key, (0, now))
            if now - first > self.lockout_seconds:       # old failures expire: count only a recent window
                count, first = 0, now
            count += 1
            self._failures[key] = (count, first)
            if count >= self.max_attempts:
                self._locked_until[key] = now + self.lockout_seconds
        return LoginResult(False, None, GENERIC_FAILURE)
