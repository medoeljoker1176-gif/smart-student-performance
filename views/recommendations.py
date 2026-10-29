"""Recommendations - prioritised improvement plan for the analysed student."""
import pandas as pd
import streamlit as st

from src.prediction import format_probability
from src import ui
from src.recommendations import NUMERIC_RULES


def _scorecard(s: dict) -> pd.DataFrame:
    """Habit-by-habit status against the rule-of-thumb targets used by the recommender."""
    rows = []
    unit = {"Study_Hours": "h/day", "Attendance": "%", "Previous_Score": "/100", "Assignment_Score": "/100"}
    name = {"Study_Hours": "Study hours", "Attendance": "Attendance", "Previous_Score": "Previous score",
            "Assignment_Score": "Assignment score"}
    for feature, (critical, target) in NUMERIC_RULES.items():
        v = s[feature]
        status = "On target" if v >= target else ("Below target" if v >= critical else "Far below target")
        rows.append({"Habit": name[feature], "Current": f"{v:g} {unit[feature]}", "Target": f">= {target:g} {unit[feature]}", "Status": status})
    sleep = s["Sleep_Hours"]
    rows.append({"Habit": "Sleep hours", "Current": f"{sleep:g} h/day", "Target": "7 - 9 h/day",
                 "Status": "On target" if 7 <= sleep <= 9 else ("Outside range" if 6 <= sleep <= 10 else "Far outside range")})
    rows.append({"Habit": "Class participation", "Current": s["Participation"], "Target": "Medium or High",
                 "Status": "On target" if s["Participation"] != "Low" else "Low"})
    return pd.DataFrame(rows)


def render():
    ui.hero("Personalised Recommendations", "A prioritised plan based on this student's own data and the model's what-if answers.")
    last = ui.require_last_analysis()
    res = last["result"]

    st.markdown(f"{ui.risk_badge(res.risk)} &nbsp; P(pass) = **{format_probability(res.p_pass)}** &nbsp;|&nbsp; "
                f"Performance score = **{res.score:.0f}/100**", unsafe_allow_html=True)
    st.write("")
    ui.render_recommendations(last["recs"], res.risk)

    st.subheader("Habit scorecard")
    st.dataframe(_scorecard(last["student"]), hide_index=True)
    st.caption("Targets are rule-of-thumb values chosen for this prototype, not scientific standards.")
    with st.expander("How the recommendation engine works"):
        st.markdown(
            "1. **Rules** compare the student's inputs with targets and create candidate recommendations.\n"
            "2. Each gets a **severity** (how far below the target).\n"
            "3. The trained model is asked a **what-if**: *what would P(pass) be if this habit reached the target?*\n"
            "4. Recommendations are **ranked** by that estimated gain (severity breaks ties) -> Priority 1, 2, 3.")
    ui.disclaimer(compact=True)
