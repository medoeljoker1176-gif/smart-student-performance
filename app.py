"""
app.py - Streamlit entry point.

Run (Windows, from the project folder with the virtual environment active):
    python -m streamlit run app.py
First create the model if models/student_performance_model.pkl is missing:
    python train_model.py
"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Smart Student Performance & Early Risk Detection",
                   page_icon=":material/school:", layout="wide", initial_sidebar_state="expanded")

from src import auth, auth_ui, ui                                      # noqa: E402  (after set_page_config)
from views import (about, analytics, batch_analysis, compare, dashboard, explanation,  # noqa: E402
                   history, model_performance, prediction, recommendations)

ui.inject_css()
user = auth_ui.require_login()          # shows the login form and stops here when login is on and nobody is signed in
_LOGO = Path(__file__).parent / "assets" / "logo.svg"
if _LOGO.exists() and hasattr(st, "logo"):
    st.logo(str(_LOGO), size="large")

_PAGE_SPECS = [
    (dashboard.render, "Dashboard", ":material/home:", "dashboard", True),
    (prediction.render, "Prediction", ":material/person_search:", "prediction", False),
    (explanation.render, "AI Explanation", ":material/psychology:", "explanation", False),
    (analytics.render, "Analytics", ":material/bar_chart:", "analytics", False),
    (recommendations.render, "Recommendations", ":material/lightbulb:", "recommendations", False),
    (compare.render, "Compare Students", ":material/compare_arrows:", "compare", False),
    (history.render, "History", ":material/history:", "history", False),
    (batch_analysis.render, "Batch CSV Analysis", ":material/upload_file:", "batch-analysis", False),
    (model_performance.render, "Model Performance", ":material/analytics:", "model-performance", False),
    (about.render, "About Project", ":material/info:", "about", False),
]
# Only the pages this role may open are registered at all, so a student cannot reach the others by URL.
allowed = auth.pages_for(user["role"], [spec[3] for spec in _PAGE_SPECS])
default_key = auth.ROLE_DEFAULT_PAGE.get(user["role"], "dashboard")
if default_key not in allowed and allowed:
    default_key = allowed[0]
specs = [spec for spec in _PAGE_SPECS if spec[3] in allowed]
if not specs:
    st.error("Your account has no pages assigned. Ask the administrator.")
    st.stop()
pages = [st.Page(fn, title=title, icon=icon, url_path=path, default=(path == default_key))
         for fn, title, icon, path, _ in specs]
ui.PAGES.clear()
ui.PAGES.update({spec[3]: page for spec, page in zip(specs, pages)})
current = st.navigation(pages)

with st.sidebar:
    st.markdown("### Student Performance Analytics")
    auth_ui.sidebar_account(user)
    ui.dataset_selector()
    bundle, error = ui.load_bundle_safe()
    if bundle:
        st.success(f"Model ready: {bundle['model_name']}", icon=":material/check_circle:")
    else:
        st.error("Model not available. Run `python train_model.py`.", icon=":material/error:")
    ui.disclaimer(compact=True)

current.run()
