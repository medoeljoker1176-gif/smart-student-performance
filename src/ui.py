"""
ui.py
-----
Streamlit helpers shared by all pages: styling, cached loaders, the input form and
the reusable "result cards". Business logic stays in the other src/ modules.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import charts, ui_html
from src.config import (CATEGORY_VALUES, DATASETS, DEFAULT_DATASET, DISCLAIMER, SAMPLE_PROFILES,
                        dataset_paths)
from src.prediction import ModelLoadError, format_probability, load_bundle
from src.preprocessing import load_dataset
from src.validation import DataError

# ------------------------------------------------------------------ styling
CSS = ui_html.CSS          # stylesheet lives in src/ui_html.py (pure HTML/CSS, previewable without Streamlit)


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# Page objects registered by app.py so pages can link to each other (st.page_link needs the object).
PAGES: dict = {}


def nav_link(key: str, label: str, icon: str | None = None):
    """Link to another page; silently skipped if the page registry is not available."""
    page = PAGES.get(key)
    if page is not None:
        st.page_link(page, label=label, icon=icon)


def hero(title: str, subtitle: str, icon_name: str = "trend"):
    st.markdown(ui_html.hero_html(title, subtitle, icon_name), unsafe_allow_html=True)


def is_real_data() -> bool:
    return active_dataset() != "synthetic"


def data_kind() -> str:
    """Short word used in warnings: 'synthetic' or 'real (UCI, one Portuguese school)'."""
    return "real (UCI, one Portuguese school)" if is_real_data() else "synthetic"


def disclaimer(compact: bool = False):
    if compact:
        st.caption(f"Educational prototype on {data_kind()} data - not an official academic judgment.")
    else:
        st.markdown(ui_html.note_html(DISCLAIMER), unsafe_allow_html=True)


def risk_badge(risk: dict) -> str:
    return f'<span class="badge" style="background:{risk["color"]}">{risk["icon"]} {risk["level"]}</span>'


def kpi_cards(items: list):
    """Row of coloured KPI cards. items: (icon, label, value, tone[, tooltip])."""
    st.markdown(ui_html.kpi_cards_html(items), unsafe_allow_html=True)


def section_title(text: str, icon_name: str = "network"):
    st.markdown(ui_html.section_title_html(text, icon_name), unsafe_allow_html=True)


PIPELINE_STEPS = ui_html.PIPELINE_STEPS


def pipeline_diagram():
    st.markdown(ui_html.pipeline_html(PIPELINE_STEPS), unsafe_allow_html=True)


# ------------------------------------------------------------------ cached loaders
def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource(show_spinner=False)
def _cached_bundle(path: str, mtime: float):
    return load_bundle(Path(path))


@st.cache_data(show_spinner=False)
def _cached_dataset(path: str, mtime: float) -> pd.DataFrame:
    return load_dataset(Path(path))


@st.cache_data(show_spinner=False)
def _cached_json(path: str, mtime: float):
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner=False)
def _cached_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


# ------------------------------------------------------------------ active dataset
def available_datasets() -> list:
    """Dataset keys whose CSV and trained model both exist (the default one is always offered)."""
    out = []
    for key in DATASETS:
        paths = dataset_paths(key)
        if key == DEFAULT_DATASET or (paths["data"].exists() and paths["model"].exists()):
            out.append(key)
    return out


def active_dataset() -> str:
    """The dataset chosen in the sidebar (falls back to the default if it is not available)."""
    key = st.session_state.get("dataset", DEFAULT_DATASET)
    return key if key in available_datasets() else DEFAULT_DATASET


def active_paths() -> dict:
    return dataset_paths(active_dataset())


def dataset_selector():
    """Sidebar switch between the datasets that are ready. Hidden when only one exists."""
    keys = available_datasets()
    if len(keys) < 2:
        return
    st.session_state.setdefault("dataset", DEFAULT_DATASET)
    st.selectbox("Dataset", keys, key="dataset", format_func=lambda k: DATASETS[k]["label"],
                 help="Each dataset has its own trained model and reports.")
    st.caption(DATASETS[active_dataset()]["note"])


@st.cache_resource(show_spinner="Model file does not match this environment - retraining it once (about 30-60 seconds)...")
def _auto_retrain(dataset: str = DEFAULT_DATASET) -> str:
    """Retrain the model once per server run. Returns "" on success or an error text."""
    try:
        from src.model_training import main as train_main
        train_main(quiet=True, dataset=dataset)
        return ""
    except Exception as exc:                                   # noqa: BLE001
        return f"Automatic retraining failed ({type(exc).__name__}: {exc})"


def load_bundle_safe():
    """(bundle, None) or (None, friendly error message). Never raises.

    If the saved model cannot be loaded (e.g. different pandas/scikit-learn version),
    it is retrained automatically once, so nobody has to run train_model.py by hand."""
    dataset = active_dataset()
    model_path = dataset_paths(dataset)["model"]
    try:
        return _cached_bundle(str(model_path), _mtime(model_path)), None
    except ModelLoadError as first_exc:
        failure = _auto_retrain(dataset)
        if failure:
            return None, f"{first_exc} {failure}"
        try:
            return _cached_bundle(str(model_path), _mtime(model_path)), None
        except Exception as exc:                               # noqa: BLE001
            return None, str(exc)
    except Exception as exc:                                   # noqa: BLE001
        return None, f"Unexpected error while loading the model: {exc}"


def load_dataset_safe():
    data_path = active_paths()["data"]
    try:
        return _cached_dataset(str(data_path), _mtime(data_path)), None
    except DataError as exc:
        return None, str(exc)
    except Exception as exc:                                   # noqa: BLE001
        return None, f"Unexpected error while loading the dataset: {exc}"


def load_metrics():
    """metrics.json as dict, or None if missing/unreadable."""
    path = active_paths()["metrics"]
    try:
        return _cached_json(str(path), _mtime(path)) if path.exists() else None
    except Exception:                                          # noqa: BLE001
        return None


def load_oof():
    path = active_paths()["oof"]
    try:
        return _cached_csv(str(path), _mtime(path)) if path.exists() else None
    except Exception:                                          # noqa: BLE001
        return None


def require_bundle():
    """Return the model bundle or show a helpful message and stop the page."""
    bundle, error = load_bundle_safe()
    if bundle is None:
        st.error(error or "The trained model is unavailable.", icon=":material/error:")
        st.caption("Create the model file by running this command in the project folder, then reload the page:")
        st.code("python train_model.py", language="bash")
        st.stop()
    for w in bundle.get("warnings", []):
        st.warning(w)
    return bundle


def require_dataset():
    df, error = load_dataset_safe()
    if df is None:
        st.error(error or "The dataset is unavailable.", icon=":material/error:")
        st.caption("Recreate the dataset by running this command in the project folder, then reload the page:")
        st.code("python generate_dataset.py", language="bash")
        st.stop()
    return df


def require_last_analysis():
    """Latest analysed student from the Prediction page, or a hint to create one."""
    last = st.session_state.get("last_analysis")
    if last is None:
        st.info("No student analysed yet. Open **Prediction** in the sidebar and press *Predict*, or analyse an example below.")
        cols = st.columns(len(SAMPLE_PROFILES))
        bundle = require_bundle()
        from src.prediction import analyze_student
        for col, (name, profile) in zip(cols, SAMPLE_PROFILES.items()):
            if col.button(f"Analyse: {name}", key=f"quick_{name}"):
                st.session_state["last_analysis"] = analyze_student(bundle, profile)
                st.rerun()
        st.stop()
    return last


# ------------------------------------------------------------------ input form
def apply_profile(prefix: str, profile: dict):
    for feature, value in profile.items():
        st.session_state[prefix + feature] = value


def profile_buttons(prefix: str = ""):
    """Buttons that fill the form with an example student."""
    cols = st.columns(len(SAMPLE_PROFILES))
    for col, (name, profile) in zip(cols, SAMPLE_PROFILES.items()):
        col.button(f"Load: {name}", key=f"{prefix}btn_{name}", on_click=apply_profile, args=(prefix, profile))


def student_form(prefix: str = "", stacked: bool = False) -> dict:
    """Draw the input widgets and return the raw values (NOT yet validated)."""
    default = SAMPLE_PROFILES["Average student"]
    # Streamlit deletes a widget's value when its page is left, so the values are also kept
    # in a separate dict and restored when the page is opened again.
    store_key = "_saved_form_" + prefix
    saved = st.session_state.get(store_key, {})
    for feature, value in default.items():
        if prefix + feature not in st.session_state:
            st.session_state[prefix + feature] = saved.get(feature, value)
    k = lambda f: prefix + f

    layout = [st.container(border=True) for _ in range(3)] if stacked else [c for c in st.columns(3)]
    with layout[0]:
        st.markdown("##### Student information")
        # UI limits are wider than the valid range on purpose: the validator explains the problem.
        age = st.number_input("Age", 10, 60, step=1, key=k("Age"), help="Whole years. The model accepts 15-40; ages outside the 17-25 seen in training get a reliability warning.")
        gender = st.selectbox("Gender", CATEGORY_VALUES["Gender"], key=k("Gender"))
    with layout[1]:
        st.markdown("##### Academic performance")
        prev = st.slider("Previous exam score", 0, 100, key=k("Previous_Score"), help="Score on the most recent exam, 0-100.")
        assign = st.slider("Assignment score", 0, 100, key=k("Assignment_Score"))
        attendance = st.slider("Attendance (%)", 0, 100, key=k("Attendance"), help="Share of classes attended, 0-100%.")
        participation = st.radio("Class participation", CATEGORY_VALUES["Participation"], horizontal=True,
                                 key=k("Participation"))
    with layout[2]:
        st.markdown("##### Lifestyle and environment")
        study = st.slider("Study hours per day", 0.0, 16.0, step=0.5, key=k("Study_Hours"), help="Average hours of study per day. Study + sleep cannot exceed 24 h.")
        sleep = st.slider("Sleep hours per day", 0.0, 14.0, step=0.5, key=k("Sleep_Hours"), help="Average hours of sleep per day.")
        internet = st.radio("Internet access", CATEGORY_VALUES["Internet_Access"], horizontal=True,
                            key=k("Internet_Access"))
        extra = st.radio("Extracurricular activities", CATEGORY_VALUES["Extracurricular"], horizontal=True,
                         key=k("Extracurricular"))
    values = {"Age": age, "Gender": gender, "Study_Hours": study, "Attendance": attendance,
              "Previous_Score": prev, "Assignment_Score": assign, "Sleep_Hours": sleep,
              "Participation": participation, "Internet_Access": internet, "Extracurricular": extra}
    st.session_state[store_key] = dict(values)
    return values


def show_validation(validation) -> bool:
    """Display friendly messages; True when the input may go to the model."""
    for msg in validation.errors:
        st.error(msg, icon=":material/error:")
    for msg in validation.warnings:
        st.warning(msg, icon=":material/warning:")
    return validation.ok


# ------------------------------------------------------------------ result cards
def render_verdict(res, key: str):
    is_pass = res.label == "Pass"
    color = charts.GREEN if is_pass else charts.RED
    st.markdown(f'<div class="verdict" style="color:{color};border-color:{color}">'
                f'Predicted result: {res.label}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Probability of passing", format_probability(res.p_pass),
               help="Estimated by the model from the inputs. It is an estimate, not a guarantee.")
    c2.metric("Risk level", f'{res.risk["icon"]} {res.risk["level"]}')
    c3.metric("Performance score", f"{res.score:.0f} / 100",
               help="Application indicator (weighted inputs), not an official grade and not produced by the ML model.")
    g1, g2 = st.columns(2)
    g1.plotly_chart(charts.gauge(res.p_pass * 100, "Probability of passing", "%", res.risk["color"]),
                    key=f"{key}_gauge_p")
    g2.plotly_chart(charts.gauge(res.score, "Performance score (app indicator)", "", charts.BLUE),
                    key=f"{key}_gauge_s")
    st.markdown(f'{risk_badge(res.risk)} &nbsp; {res.risk["message"]}', unsafe_allow_html=True)


def render_top_factors(explanation: dict, n: int = 3):
    table = explanation["table"]
    helping = table[table["Contribution"] >= 0.5].sort_values("Contribution", ascending=False).head(n)
    hurting = table[table["Contribution"] <= -0.5].sort_values("Contribution").head(n)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Helping this student**")
        for s in helping["Sentence"]:
            st.success(s)
        if helping.empty:
            st.caption("No factor clearly increased the chance of passing.")
    with c2:
        st.markdown("**Holding this student back**")
        for s in hurting["Sentence"]:
            st.error(s)
        if hurting.empty:
            st.caption("No factor clearly decreased the chance of passing.")


def render_recommendations(recs: list, risk: dict):
    if not recs:
        st.success("No weak area was found by the rules. Keep the routine and keep revising regularly.")
        return
    if risk["level"] == "High Risk":
        st.warning("This student is at high risk. Start with Priority 1 and Priority 2 this week.")
    for i, rec in enumerate(recs[:3], start=1):
        with st.container(border=True):
            st.markdown(f"#### Priority {i} - {rec.title}")
            st.write(rec.action)
            st.caption(f"**Why this was recommended:** {rec.reason}")
            if rec.what_if:
                st.caption(f"**Model what-if:** {rec.what_if}")
    if len(recs) > 3:
        with st.expander(f"Additional improvements ({len(recs) - 3})"):
            for rec in recs[3:]:
                st.markdown(f"**{rec.title}** - {rec.action}")
                st.caption(f"Why: {rec.reason}")
    st.caption("Ranking: estimated gain in P(pass) from the model's what-if answer, then how large the gap is. "
               f"What-if results show patterns in {data_kind()} data, not guaranteed outcomes.")


def uncertainty_note(metrics: dict | None):
    """Plain-language uncertainty box. Numbers come from reports/metrics.json only if present."""
    with st.expander("How certain is this prediction?"):
        lines = [f"- The probability is an **estimate** from a model trained on **{data_kind()}** data. "
                 "It is not a guarantee and says nothing certain about any individual student.",
                 "- Probabilities near 0% or 100% are shown as *<1%* or *>99%* because the model is not that certain.",
                 "- Risk levels use thresholds chosen for this prototype (not validated cut-offs).",
                 "- A prediction should start a conversation about support, never decide an outcome on its own."]
        try:
            final = metrics["final_model"]
            test = metrics["test_metrics"][final]
            lines.append(f"- On the held-out **{data_kind()}** test set ({metrics['n_test']} students) the final model "
                         f"({final}) reached accuracy {test['accuracy']:.1%} and ROC-AUC {test['roc_auc']:.3f}: "
                         "roughly 1 in "
                         f"{max(1, round(1 / max(1e-9, 1 - test['accuracy'])))} test predictions was wrong. "
                         "This does not establish real-world performance.")
        except (TypeError, KeyError):
            lines.append("- Test-set metrics were not found (reports/metrics.json), so no accuracy figure is shown.")
        st.markdown("\n".join(lines))
