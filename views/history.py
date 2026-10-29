"""Prediction history - saved predictions and progress over time (storage: local CSV, unchanged)."""
import streamlit as st

from src import charts, history, ui
from src.exports import dataframe_to_safe_csv_bytes

RISK_ORDER = ["High Risk", "Moderate Risk", "Low Risk"]


def render():
    ui.hero("Prediction History", "Predictions are saved locally. Use the same Student ID to follow progress over time.")
    df = history.read_history()
    if df.empty:
        st.info("No predictions saved yet. Make a prediction on the **Prediction** page and it will appear here.",
                icon=":material/history:")
        ui.nav_link("prediction", "Go to Prediction", ":material/arrow_forward:")
        st.caption("Unreadable or hand-edited rows in the history storage are ignored rather than crashing the page.")
        return

    stats = history.summarize_history(df)
    c = st.columns(4)
    c[0].metric("Saved predictions", f"{stats['n']:,}")
    c[1].metric("High-risk share", f"{(df['Risk'] == 'High Risk').mean():.0%}")
    c[2].metric("Average P(pass)", f"{stats['mean_p_pass']:.1%}")
    c[3].metric("Distinct Student IDs", int(df.loc[df["Student_ID"] != "", "Student_ID"].nunique()))

    f1, f2 = st.columns(2)
    ids = sorted(i for i in df["Student_ID"].unique() if i)
    choice = f1.selectbox("Student", ["All predictions"] + ids)
    risks = f2.multiselect("Risk level", RISK_ORDER, default=RISK_ORDER)

    shown = df if choice == "All predictions" else df[df["Student_ID"] == choice]
    shown = shown[shown["Risk"].isin(risks)].sort_values("Timestamp", ascending=False)
    st.caption(f"Showing {len(shown):,} of {len(df):,} saved predictions.")
    if shown.empty:
        st.warning("No saved prediction matches these filters.")
    else:
        st.dataframe(shown, hide_index=True)

        counts = shown["Risk"].value_counts().reindex(RISK_ORDER).dropna()
        if len(counts):
            import plotly.express as px
            fig = px.bar(x=counts.index, y=counts.values, color=counts.index, text=counts.values,
                         color_discrete_map={"High Risk": charts.RED, "Moderate Risk": charts.AMBER, "Low Risk": charts.GREEN},
                         labels={"x": "Risk level", "y": "Predictions"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(charts._style(fig, 300, "Saved predictions by risk level"), key="hist_risk")

        if choice != "All predictions":
            if len(shown) >= 2:
                st.plotly_chart(charts.history_trend(shown.sort_values("Timestamp")), key="hist_trend")
            else:
                st.caption("Make at least two predictions for this Student ID to see the trend.")
        elif ids:
            st.caption("Select a Student ID above to see that student's progress over time.")

        st.download_button("Download shown rows as CSV", dataframe_to_safe_csv_bytes(shown),
                           file_name="prediction_history.csv", mime="text/csv", key="dl_history")
    with st.expander("Clear history"):
        confirm = st.checkbox("I understand that all saved predictions will be deleted.")
        if st.button("Delete all history", disabled=not confirm):
            try:
                history.clear_history()
            except Exception as exc:
                st.error(f"History could not be deleted: {exc}")
            else:
                st.rerun()
    where = ("a Cloud Firestore collection shared by all service instances" if history.backend() == "firestore"
             else "a local CSV file that is reset when the cloud instance restarts or scales")
    st.caption(f"Privacy: prefer anonymous Student IDs. History is stored in {where}.")
