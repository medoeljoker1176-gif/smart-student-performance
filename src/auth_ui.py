"""
auth_ui.py
----------
Thin Streamlit layer over src/auth.py: the login form, session handling and logout button.
All decisions (hashing, lockout, roles) live in src/auth.py, which has its own unit tests.

Login is ON only when the secrets contain an [auth] section with at least one valid user.
Otherwise the app runs in open demo mode exactly like before (so local runs and tests keep working).
"""
import time

import streamlit as st

from src import auth

SESSION_KEYS = ("auth_user", "auth_role", "auth_name", "auth_last_seen")


def _auth_config() -> dict:
    """The [auth] secrets as a plain dict ({} when there is no secrets file)."""
    try:
        section = st.secrets.get("auth")
        return _plain(section) if section else {}
    except Exception:                                          # noqa: BLE001  (no secrets file, bad TOML...)
        return {}


def _plain(value):
    return {k: _plain(v) for k, v in value.items()} if hasattr(value, "items") else value


@st.cache_resource(show_spinner=False)
def _authenticator(users_signature: str, max_attempts: int, lockout_minutes: float):
    """One shared Authenticator per server, so failed attempts are counted across browser tabs."""
    cfg = _auth_config()
    users, _ = auth.load_users(cfg)
    return auth.Authenticator(users, max_attempts, lockout_minutes)


def _setup():
    cfg = _auth_config()
    users, problems = auth.load_users(cfg)
    if not users:
        return None, cfg, problems
    signature = "|".join(f"{u.username}:{u.role}:{u.password_hash[-12:]}" for u in users.values())
    try:
        max_attempts = int(cfg.get("max_attempts", auth.MAX_ATTEMPTS))
        lockout = float(cfg.get("lockout_minutes", auth.LOCKOUT_MINUTES))
    except (TypeError, ValueError):
        max_attempts, lockout = auth.MAX_ATTEMPTS, auth.LOCKOUT_MINUTES
    return _authenticator(signature, max(1, max_attempts), max(0.1, lockout)), cfg, problems


def _clear_session():
    for key in SESSION_KEYS:
        st.session_state.pop(key, None)


def current_user() -> dict | None:
    """The signed-in user (or the open-mode guest), without drawing anything."""
    authenticator, cfg, _ = _setup()
    if authenticator is None:
        return {"username": "guest", "role": "teacher", "name": "Guest (login disabled)", "open": True}
    if st.session_state.get("auth_user") is None:
        return None
    try:
        minutes = float(cfg.get("session_minutes", auth.SESSION_MINUTES))
    except (TypeError, ValueError):
        minutes = auth.SESSION_MINUTES
    if auth.session_expired(st.session_state.get("auth_last_seen"), time.time(), minutes):
        _clear_session()
        st.session_state["auth_notice"] = "Your session expired. Please sign in again."
        return None
    st.session_state["auth_last_seen"] = time.time()
    return {"username": st.session_state["auth_user"], "role": st.session_state["auth_role"],
            "name": st.session_state.get("auth_name", ""), "open": False}


def require_login() -> dict:
    """Return the signed-in user, or show the login form and stop the script."""
    user = current_user()
    if user is not None:
        return user
    authenticator, _, problems = _setup()
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("## Sign in")
        st.caption("Smart Student Performance & Early Risk Detection")
        notice = st.session_state.pop("auth_notice", None)
        if notice:
            st.info(notice)
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", max_chars=auth.MAX_USERNAME_LENGTH)
            password = st.text_input("Password", type="password", max_chars=auth.MAX_PASSWORD_LENGTH)
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            result = authenticator.login(username, password, time.time())
            if result.ok:
                st.session_state.update(auth_user=result.user.username, auth_role=result.user.role,
                                        auth_name=result.user.name, auth_last_seen=time.time())
                st.rerun()
            st.error(result.message)
        if problems:
            st.caption("Some configured accounts were ignored (check the secrets format).")
    st.stop()


def sidebar_account(user: dict):
    """Who is signed in + logout (nothing to show in open demo mode except a short note)."""
    if user.get("open"):
        st.caption("Login is off (no users in secrets). Demo mode.")
        return
    st.markdown(f"**{user['name']}**  \n{user['role'].title()}")
    if st.button("Sign out", use_container_width=True):
        _clear_session()
        st.rerun()
