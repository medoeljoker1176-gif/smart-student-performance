"""Compare Students - Student A vs Student B."""
import pandas as pd
import streamlit as st

from src.prediction import format_probability
from src import charts, ui
from src.config import FEATURES, FEATURE_LABELS, FEATURE_UNITS, NUMERIC_FEATURES, SAMPLE_PROFILES
from src.prediction import analyze_student
from src.validation import validate_student_input


def _fmt(feature, value):
    return f"{value:g}{FEATURE_UNITS.get(feature, '')}" if feature in NUMERIC_FEATURES else str(value)


def render():
    ui.hero("Compare Students", "Put two student profiles side by side and see what makes their predictions different.")
    bundle = ui.require_bundle()

    if "b_initialised" not in st.session_state:                       # Student B starts as the at-risk example
        ui.apply_profile("b_", SAMPLE_PROFILES["At-risk student"])
        st.session_state["b_initialised"] = True

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Student A")
        ui.profile_buttons("a_")
        raw_a = ui.student_form("a_", stacked=True)
    with col_b:
        st.subheader("Student B")
        ui.profile_buttons("b_")
        raw_b = ui.student_form("b_", stacked=True)

    va = validate_student_input(raw_a, bundle["training_ranges"])
    vb = validate_student_input(raw_b, bundle["training_ranges"])
    ok_a, ok_b = st.columns(2)
    with ok_a:
        a_ok = ui.show_validation(va)
    with ok_b:
        b_ok = ui.show_validation(vb)
    if not (a_ok and b_ok):
        st.info("Fix the highlighted input(s) to see the comparison.")
        return

    A, B = analyze_student(bundle, va.clean), analyze_student(bundle, vb.clean)
    ra, rb = A["result"], B["result"]

    st.divider()
    st.subheader("Summary")
    c1, c2 = st.columns(2)
    for col, name, r in ((c1, "Student A", ra), (c2, "Student B", rb)):
        with col, st.container(border=True):
            st.markdown(f"**{name}** - Predicted **{r.label}**")
            m1, m2 = st.columns(2)
            m1.metric("Probability of passing", format_probability(r.p_pass))
            m2.metric("Performance score", f"{r.score:.0f}/100")
            st.markdown(ui.risk_badge(r.risk), unsafe_allow_html=True)
    diff = (ra.p_pass - rb.p_pass) * 100
    st.info(f"Student A's probability of passing is **{abs(diff):.1f} points {'higher' if diff >= 0 else 'lower'}** than Student B's.")

    st.subheader("Side-by-side table")
    rows = []
    for f in FEATURES:
        va_, vb_ = A["student"][f], B["student"][f]
        d = f"{va_ - vb_:+g}" if f in NUMERIC_FEATURES else ("same" if va_ == vb_ else "different")
        rows.append({"Item": FEATURE_LABELS[f], "Student A": _fmt(f, va_), "Student B": _fmt(f, vb_), "Difference (A - B)": d})
    rows += [{"Item": "Probability of passing", "Student A": format_probability(ra.p_pass), "Student B": format_probability(rb.p_pass), "Difference (A - B)": f"{diff:+.1f} pts"},
             {"Item": "Risk level", "Student A": ra.risk["level"], "Student B": rb.risk["level"], "Difference (A - B)": ""},
             {"Item": "Performance score", "Student A": f"{ra.score:.0f}", "Student B": f"{rb.score:.0f}", "Difference (A - B)": f"{ra.score - rb.score:+.0f}"}]
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.subheader("Charts")
    sa, sb = {**A["student"], "Performance score": ra.score}, {**B["student"], "Performance score": rb.score}
    g1, g2 = st.columns(2)
    g1.plotly_chart(charts.compare_bars(sa, sb, ["Attendance", "Previous_Score", "Assignment_Score", "Performance score"], "Scores and attendance (0-100)"), key="cmp_scores")
    g2.plotly_chart(charts.compare_bars(sa, sb, ["Study_Hours", "Sleep_Hours"], "Daily hours"), key="cmp_hours")
    st.plotly_chart(charts.contribution_compare(A["explanation"]["table"], B["explanation"]["table"]), key="cmp_contrib")
    ui.disclaimer(compact=True)
