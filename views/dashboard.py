"""Home dashboard - key numbers, history summary, pipeline overview and navigation."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import charts, history, ui
from src.preprocessing import clean_data


def _history_section():
    df = history.read_history()
    stats = history.summarize_history(df)
    st.subheader("Your analyses")
    if stats["n"] == 0:
        st.info("No analyses saved yet. Run a prediction to see your own summary here.", icon=":material/insights:")
        return
    c = st.columns(4)
    c[0].metric("Students analysed", f"{stats['n']:,}", help="Number of saved predictions (one student can be analysed more than once).")
    c[1].metric("Predicted to pass", f"{stats['predicted_pass_share']:.0%}")
    c[2].metric("Average P(pass)", f"{stats['mean_p_pass']:.1%}")
    c[3].metric("High-risk analyses", f"{stats['risk_counts'].get('High Risk', 0):,}")

    left, right = st.columns(2)
    risk = pd.DataFrame({"Risk level": list(stats["risk_counts"]), "Analyses": list(stats["risk_counts"].values())})
    fig = px.pie(risk, names="Risk level", values="Analyses", hole=0.5, color="Risk level",
                 color_discrete_map={"High Risk": charts.RED, "Moderate Risk": charts.AMBER, "Low Risk": charts.GREEN})
    left.plotly_chart(charts._style(fig, 300, "Risk-level distribution"), key="dash_risk")
    with right:
        st.markdown("**Recent analyses**")
        recent = stats["recent"][["Timestamp", "Student_ID", "Prediction", "P_Pass", "Risk"]].copy()
        recent["P_Pass"] = (recent["P_Pass"] * 100).round(1).astype(str) + "%"
        st.dataframe(recent, hide_index=True)


def render():
    ui.hero("Smart Student Performance & Early Risk Detection",
            "An educational AI system that estimates the chance of passing, flags risk early, explains why, "
            "and suggests what to improve.")
    ui.disclaimer()

    df, data_error = ui.load_dataset_safe()
    metrics = ui.load_metrics()
    bundle, model_error = ui.load_bundle_safe()

    if model_error:
        st.error(model_error, icon=":material/error:")
        st.code("python train_model.py", language="bash")
    if data_error:
        st.error(data_error, icon=":material/error:")
        st.code("python generate_dataset.py", language="bash")

    clean = None
    kpis = []
    if df is not None:
        clean = clean_data(df)
        kpis.append(("users", "Students in dataset", f"{len(clean):,}", "blue",
                     "Synthetic students used to train and test the model."))
        kpis.append(("percent", "Pass rate", f"{(clean['Result'] == 'Pass').mean():.0%}", "green",
                     "Share of students who passed in the reference dataset."))
    if metrics:
        final = metrics["final_model"]
        test = metrics["test_metrics"][final]
        kpis.append(("network", "Final model", final, "indigo", f"Final model: {final}"))
        kpis.append(("target", "Test accuracy", f"{test['accuracy']:.1%}", "violet",
                     f"On the held-out {ui.data_kind()} test set only."))
        kpis.append(("bars", "Test ROC AUC", f"{test['roc_auc']:.3f}", "cyan",
                     f"Ranking quality on the held-out {ui.data_kind()} test set."))
    if kpis:
        ui.kpi_cards(kpis)
    else:
        st.info("Dataset and training report are not available yet, so no reference numbers can be shown.")

    ui.section_title("How the system works")
    ui.pipeline_diagram()

    _history_section()

    st.subheader("Go to")
    n = st.columns(4)
    with n[0]:
        ui.nav_link("prediction", "Predict one student", ":material/person:")
    with n[1]:
        ui.nav_link("batch-analysis", "Analyse a CSV", ":material/upload_file:")
    with n[2]:
        ui.nav_link("model-performance", "Model performance", ":material/analytics:")
    with n[3]:
        ui.nav_link("history", "Prediction history", ":material/history:")

    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.markdown("#### What questions does it answer?")
            st.markdown(
                "- **What is the predicted performance?** Pass or Fail\n"
                "- **How likely is passing?** probability from the ML model\n"
                "- **How risky is the situation?** Low / Moderate / High risk\n"
                "- **Why?** the factors that pushed the prediction up or down\n"
                "- **What should improve?** prioritised, personalised recommendations\n"
                "- **How is progress over time?** prediction history per student")
    with right:
        last = st.session_state.get("last_analysis")
        with st.container(border=True):
            st.markdown("#### Latest analysed student")
            if last:
                res = last["result"]
                st.plotly_chart(charts.gauge(res.p_pass * 100, "Probability of passing", "%", res.risk["color"], 200),
                                key="dash_gauge")
                st.markdown(ui.risk_badge(res.risk), unsafe_allow_html=True)
            else:
                st.info("No prediction yet in this session.")

    if metrics and clean is not None:
        st.subheader("At a glance")
        g1, g2 = st.columns(2)
        g1.plotly_chart(charts.class_distribution(clean), key="dash_pie")
        g2.plotly_chart(charts.global_importance_bar(metrics["global_importance"]), key="dash_importance")
