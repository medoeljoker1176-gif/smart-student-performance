"""About Project."""
import pandas as pd
import streamlit as st

from src import ui
from src.config import HIGH_RISK_BELOW, LOW_RISK_FROM
from src.prediction import SCORE_WEIGHTS


def render():
    ui.hero("About the Project", "Smart Student Performance & Early Risk Detection AI System")
    ui.disclaimer()

    st.subheader("Problem and goal")
    st.write("Teachers and students often notice poor performance too late. This prototype shows how machine learning "
             "can flag students at risk early, explain the reasons, and suggest what to improve.")

    st.subheader("System pipeline")
    ui.pipeline_diagram()

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("#### Risk level")
        st.markdown(f"Based on the model's probability of passing, P(pass):\n"
                    f"- 🔴 **High risk:** P(pass) < {HIGH_RISK_BELOW:.0%}\n"
                    f"- 🟠 **Moderate risk:** {HIGH_RISK_BELOW:.0%} to {LOW_RISK_FROM:.0%}\n"
                    f"- 🟢 **Low risk:** P(pass) >= {LOW_RISK_FROM:.0%}\n\n"
                    "50% is the model's own decision boundary; 75% is a margin chosen for this prototype. "
                    "The thresholds are **not scientifically validated**.")
    with c2, st.container(border=True):
        st.markdown("#### Performance score")
        st.markdown("Application-level indicator (0-100), **not an official grade**: weighted average of the inputs.")
        st.dataframe(pd.DataFrame([{"Component": k.replace("_", " "), "Weight": f"{w:.0%}"} for k, w in SCORE_WEIGHTS.items()]),
                     hide_index=True)

    st.subheader("Technologies")
    st.dataframe(pd.DataFrame([
        {"Technology": "Python", "Role": "Main language"},
        {"Technology": "pandas / NumPy", "Role": "Data loading, cleaning, calculations"},
        {"Technology": "scikit-learn", "Role": "Pipelines, models, cross-validation, metrics"},
        {"Technology": "Shapley values (own implementation)", "Role": "Explain each prediction (same idea as SHAP)"},
        {"Technology": "Plotly", "Role": "Interactive charts in the app"},
        {"Technology": "Matplotlib / Seaborn", "Role": "Static report images created during training"},
        {"Technology": "Streamlit", "Role": "Web dashboard"},
        {"Technology": "joblib", "Role": "Saving and loading the trained model"},
    ]), hide_index=True)

    st.subheader("Limitations and ethics")
    st.markdown(
        f"- The active dataset is **{ui.data_kind()}**; accuracy figures show the method, not performance in other schools.\n"
        "- The model finds patterns, not causes. What-if answers describe the model, not guaranteed outcomes.\n"
        "- Only 10 features; real performance depends on many more factors.\n"
        "- Fairness checks are basic and cannot prove the absence of bias.\n"
        "- Use it as a conversation starter for support, never to punish or make final decisions about a student.\n"
        "- History is stored locally in a CSV file; use anonymous IDs, not real names, for any real deployment.")
