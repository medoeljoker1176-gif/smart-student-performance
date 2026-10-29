"""Prediction - validated input form and the full result."""
import streamlit as st

from src import charts, history, ui
from src.prediction import analyze_student
from src.validation import validate_student_input


def render():
    ui.hero("Student Prediction", "Enter the student's data. The result shows the prediction, probability, "
            "risk level, main factors and recommendations.")
    bundle = ui.require_bundle()

    st.caption("Quick demo: load an example student, then press Predict.")
    ui.profile_buttons("")
    if "student_id" not in st.session_state:                  # restore after visiting another page
        st.session_state["student_id"] = st.session_state.get("_saved_student_id", "")
    student_id = st.text_input("Student ID (optional)", key="student_id", max_chars=40,
                               help="Used only to follow progress in History. Prefer an anonymous ID instead of a real name; "
                                    "it is stored in the local history file.")
    st.session_state["_saved_student_id"] = student_id
    raw = ui.student_form("")

    if st.button("Predict performance", type="primary"):
        validation = validate_student_input(raw, bundle["training_ranges"])
        if ui.show_validation(validation):                     # invalid input never reaches the model
            try:
                with st.spinner("Analysing (exact explanation takes a moment)..."):
                    analysis = analyze_student(bundle, validation.clean)
            except Exception as exc:                                  # noqa: BLE001 - one failure must not break other pages
                st.error(f"The prediction could not be completed: {type(exc).__name__}: {exc}", icon=":material/error:")
                st.stop()
            st.session_state["last_analysis"] = analysis
            try:
                history.add_record(analysis["student"], analysis["result"], student_id)
            except OSError as exc:
                st.warning(f"The prediction could not be saved to history: {exc}")
        else:
            st.session_state.pop("last_analysis", None)

    last = st.session_state.get("last_analysis")
    if not last:
        ui.disclaimer(compact=True)
        return

    st.divider()
    st.subheader("Result")
    ui.render_verdict(last["result"], key="pred")
    try:
        from src.reporting import build_student_report
        report_analysis = {**last, "model_name": bundle.get("model_name", "Saved model")}
        pdf_bytes = build_student_report(report_analysis, st.session_state.get("student_id", ""))
        st.download_button("Download student report (PDF)", pdf_bytes,
                           file_name="student_prediction_report.pdf", mime="application/pdf", key="dl_pdf")
    except Exception as exc:
        st.warning(f"The PDF report could not be created: {type(exc).__name__}: {exc}")

    try:
        from src.exports import dataframe_to_safe_csv_bytes, student_result_to_frame
        csv_bytes = dataframe_to_safe_csv_bytes(
            student_result_to_frame(last["student"], last["result"], st.session_state.get("student_id", "")))
        st.download_button("Download result (CSV)", csv_bytes, file_name="student_prediction.csv",
                           mime="text/csv", key="dl_csv")
    except Exception as exc:                                             # noqa: BLE001
        st.warning(f"The CSV could not be created: {type(exc).__name__}: {exc}")

    ui.uncertainty_note(ui.load_metrics())
    t1, t2, t3 = st.tabs(["Main contributing factors", "Recommendations", "Performance score"])
    with t1:
        ui.render_top_factors(last["explanation"])
        st.plotly_chart(charts.contributions_bar(last["explanation"]["table"]), key="pred_contrib")
        st.caption("Open **AI Explanation** in the sidebar for the full explanation.")
    with t2:
        ui.render_recommendations(last["recs"], last["result"].risk)
    with t3:
        st.markdown("The **Performance Score** is an application-level indicator from 0 to 100. It is a weighted "
                    "average of the student's inputs, **not** an official grade and **not** produced by the ML model.")
        st.dataframe(last["result"].score_table, hide_index=True)
        st.caption("Formula: sum of (weight x component score). Study hours reach 100 at 6 h/day; sleep is best at 7.5 h; "
                   "participation Low/Medium/High = 0/60/100.")
    st.caption("The result above belongs to the inputs at the moment you pressed Predict.")
    ui.disclaimer(compact=True)
