"""Batch analysis: validate an uploaded CSV, preview it, and run the saved model on valid rows."""
import io

import pandas as pd
import plotly.express as px
import streamlit as st

from src import charts, ui
from src.batch import (MAX_ROWS, MAX_UPLOAD_BYTES, BatchInputError, normalize_columns,
                       process_batch_dataframe, summarize_batch)
from src.config import CATEGORY_VALUES, FEATURES, SAMPLE_PROFILES
from src.exports import dataframe_to_safe_csv_bytes
from src.prediction import ModelLoadError

STATE_KEY = "batch_result"


def _template() -> bytes:
    rows = [{**profile} for profile in SAMPLE_PROFILES.values()]
    return pd.DataFrame(rows, columns=FEATURES).to_csv(index=False).encode("utf-8")


def _read_csv(uploaded) -> pd.DataFrame:
    """Read an uploaded CSV as DATA only (never executed). Raises BatchInputError with a friendly message."""
    if uploaded.size > MAX_UPLOAD_BYTES:
        raise BatchInputError(f"The file is {uploaded.size / 1024 / 1024:.1f} MB; the limit is "
                              f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB. Split it into smaller files.")
    raw = uploaded.getvalue()
    if not raw.strip():
        raise BatchInputError("The uploaded file is empty.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BatchInputError("The file is not UTF-8 text. In Excel use 'Save as > CSV UTF-8'.") from exc
    try:
        return pd.read_csv(io.StringIO(text), skipinitialspace=True)
    except pd.errors.EmptyDataError as exc:
        raise BatchInputError("The file has no columns. The first line must be the header row.") from exc
    except pd.errors.ParserError as exc:
        raise BatchInputError(f"The file could not be parsed as CSV ({exc}).") from exc


def _schema_report(frame: pd.DataFrame) -> list:
    """Quick pre-check shown in the preview, before any prediction is made."""
    notes = []
    cols = list(normalize_columns(frame).columns)
    missing = [c for c in FEATURES if c not in cols]
    if missing:
        notes.append(("error", "Missing required column(s): " + ", ".join(missing)))
    extra = [c for c in cols if c not in FEATURES]
    if extra:
        notes.append(("info", "Extra column(s) will be ignored: " + ", ".join(map(str, extra))))
    if not missing:
        empty_cells = int(normalize_columns(frame)[FEATURES].isna().sum().sum())
        if empty_cells:
            notes.append(("warning", f"{empty_cells} empty cell(s) found. Rows with empty required values will be rejected, "
                                     "not filled in."))
    return notes


def _summary(predictions: pd.DataFrame):
    s = summarize_batch(predictions)
    c = st.columns(4)
    c[0].metric("Students scored", f"{s['n']:,}")
    c[1].metric("Predicted to pass", f"{s['pass_share']:.0%}")
    c[2].metric("Average P(pass)", f"{s['mean_p_pass']:.1%}")
    c[3].metric("High-risk students", f"{s['risk_counts'].get('High Risk', 0):,}")

    left, right = st.columns(2)
    risk = pd.DataFrame({"Risk category": list(s["risk_counts"]), "Students": list(s["risk_counts"].values())})
    fig = px.bar(risk, x="Risk category", y="Students", color="Risk category", text="Students",
                 color_discrete_map={"High Risk": charts.RED, "Moderate Risk": charts.AMBER, "Low Risk": charts.GREEN},
                 category_orders={"Risk category": ["High Risk", "Moderate Risk", "Low Risk"]})
    fig.update_layout(showlegend=False)
    left.plotly_chart(charts._style(fig, 320, "Students per risk category"), key="batch_risk")
    hist = px.histogram(predictions, x="P(pass)", nbins=20, color_discrete_sequence=[charts.BLUE],
                        labels={"P(pass)": "Estimated probability of passing"})
    hist.update_yaxes(title="Students")
    hist.update_xaxes(range=[0, 1])
    right.plotly_chart(charts._style(hist, 320, "Distribution of estimated P(pass)"), key="batch_hist")


def _show_result(result: dict):
    predictions, rejected = result["predictions"], result["rejected"]
    (st.success if rejected.empty else st.warning)(
        f"Analysis of **{result['name']}** finished: {len(predictions):,} row(s) accepted, {len(rejected):,} rejected.")
    if not predictions.empty:
        _summary(predictions)
        st.subheader("Predictions")
        st.dataframe(predictions, hide_index=True)
        st.download_button("Download prediction results (CSV)", dataframe_to_safe_csv_bytes(predictions),
                           "student_predictions.csv", "text/csv", key="dl_batch_pred")
        st.caption("Predicted results are model estimates, not confirmed academic outcomes. "
                   "Text cells that start with = + - @ are prefixed with an apostrophe in the CSV to block spreadsheet formulas.")
    else:
        st.error("No row passed validation, so nothing was sent to the model. See the table below.")
    if not rejected.empty:
        st.subheader("Rows requiring correction")
        st.dataframe(rejected, hide_index=True)
        st.download_button("Download validation errors (CSV)", dataframe_to_safe_csv_bytes(rejected),
                           "csv_validation_errors.csv", "text/csv", key="dl_batch_err")


def render():
    ui.hero("Batch CSV Analysis", "Score many students at once with the saved model. The model is never retrained.")
    st.info(f"The active dataset is {ui.data_kind()}. Predictions are estimates for demonstration, not official academic decisions.",
            icon=":material/info:")

    with st.expander("Expected CSV format", expanded=False):
        spec = pd.DataFrame([{"Column": f, "Allowed values": (", ".join(CATEGORY_VALUES[f]) if f in CATEGORY_VALUES
                              else "number")} for f in FEATURES])
        st.dataframe(spec, hide_index=True)
        st.caption(f"One row per student, header in the first line, UTF-8, at most {MAX_ROWS:,} rows and "
                   f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB. Extra columns are ignored. Rows with invalid values are rejected, "
                   "never silently corrected.")
    st.download_button("Download CSV template", _template(), "student_template.csv", "text/csv", key="dl_template")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], help="The file is read as data only; it is never executed.")
    if uploaded is None:
        st.session_state.pop(STATE_KEY, None)
        st.caption("No file selected yet. Download the template above, fill it in and upload it here.")
        return

    try:
        frame = _read_csv(uploaded)
    except BatchInputError as exc:
        st.session_state.pop(STATE_KEY, None)
        st.error(str(exc), icon=":material/error:")
        return

    st.subheader("Preview")
    st.write(f"**{uploaded.name}** - {len(frame):,} row(s), {frame.shape[1]} column(s). First 20 rows:")
    st.dataframe(frame.head(20))
    blocking = False
    for level, text in _schema_report(frame):
        getattr(st, level)(text)
        blocking = blocking or level == "error"
    if len(frame) > MAX_ROWS:
        st.error(f"This demo limits each upload to {MAX_ROWS:,} rows (this file has {len(frame):,}).")
        blocking = True
    if frame.empty:
        st.error("The file has a header but no data rows.")
        blocking = True

    file_id = (uploaded.name, uploaded.size)
    if not blocking and st.button("Analyse CSV", type="primary", key="batch_run"):
        bundle, error = ui.load_bundle_safe()
        if bundle is None:
            st.error(error or "The trained model is unavailable. Run `python train_model.py`.", icon=":material/error:")
            return
        try:
            with st.spinner("Validating rows and running the model..."):
                predictions, rejected = process_batch_dataframe(frame, bundle)
            st.session_state[STATE_KEY] = {"id": file_id, "name": uploaded.name,
                                           "predictions": predictions, "rejected": rejected}
        except (BatchInputError, ModelLoadError) as exc:
            st.error(str(exc), icon=":material/error:")
            return
        except Exception as exc:                                   # noqa: BLE001 - keep other pages alive
            st.error(f"Batch analysis failed safely: {type(exc).__name__}: {exc}")
            return

    # Results live in session_state so a download click (which reruns the script) does not erase them.
    result = st.session_state.get(STATE_KEY)
    if result and result["id"] == file_id:
        st.divider()
        _show_result(result)
