"""Analytics - data quality, distributions, relationships and correlations."""
import pandas as pd
import streamlit as st

from src import charts, ui
from src.config import CATEGORICAL_FEATURES, CATEGORY_VALUES, FEATURE_LABELS, NUMERIC_FEATURES
from src.preprocessing import clean_data, data_quality_report

KEY_NUMERIC = ["Study_Hours", "Attendance", "Previous_Score", "Assignment_Score", "Sleep_Hours"]


def _data_quality_tab(raw: pd.DataFrame):
    q = data_quality_report(raw)
    c = st.columns(4)
    c[0].metric("Records", f"{q['n_records']:,}")
    c[1].metric("Features", q["n_features"])
    c[2].metric("Missing values", f"{q['missing_total']}", help=f"{q['missing_pct']:.2f}% of all feature cells")
    c[3].metric("Duplicate records", q["duplicates"])

    left, right = st.columns(2)
    left.plotly_chart(charts.class_distribution(raw.dropna(subset=["Result"])), key="dq_pie")
    if q["missing_by_column"]:
        right.plotly_chart(charts.missing_values_bar(q["missing_by_column"]), key="dq_missing")
    else:
        right.success("No missing values found.")
    counts = q["class_counts"]
    total = sum(counts.values())
    st.write("**Class distribution:** " + ", ".join(f"{k}: {v} ({v / total:.0%})" for k, v in counts.items()))

    a, b = st.columns(2)
    a.markdown("**Numeric features:** " + ", ".join(FEATURE_LABELS[f] for f in q["numeric_features"]))
    b.markdown("**Categorical features:** " + ", ".join(FEATURE_LABELS[f] for f in q["categorical_features"]))
    st.markdown("**Summary statistics of numeric features**")
    st.dataframe(raw[NUMERIC_FEATURES].describe().T.round(2))

    with st.expander("How are missing values handled?", expanded=True):
        st.markdown(
            f"{q['rows_with_missing']} rows have at least one missing value. We do **not** delete them and we do **not** "
            "fill them before splitting the data. Inside the training pipeline: numeric gaps are filled with the "
            "**median** and categorical gaps with the **most frequent** value, both learned from the **training rows only**. "
            "This avoids data leakage.")
    with st.expander("Why does preprocessing matter?"):
        st.markdown("Models cannot use empty cells, and features on different scales (attendance 0-100 vs study hours 0-12) "
                    "or text categories need converting. Doing it inside a pipeline keeps training and prediction identical "
                    "and prevents information from the test set leaking into the model.")


def render():
    ui.hero("Analytics", "Explore the dataset: quality, distributions and how each factor relates to the result.")
    raw = ui.require_dataset()
    df = clean_data(raw)

    # ---- filters (used by the chart tabs)
    with st.expander("Filters for the charts", expanded=False):
        f1, f2, f3 = st.columns(3)
        genders = f1.multiselect("Gender", CATEGORY_VALUES["Gender"], default=CATEGORY_VALUES["Gender"])
        parts = f2.multiselect("Participation", CATEGORY_VALUES["Participation"], default=CATEGORY_VALUES["Participation"])
        lo, hi = int(df["Age"].min()), int(df["Age"].max())
        ages = f3.slider("Age range", lo, hi, (lo, hi)) if lo < hi else (lo, hi)
    view = df[df["Gender"].isin(genders) & df["Participation"].isin(parts) & df["Age"].between(*ages)]
    # rows with a missing Participation value are excluded when that filter is used; say so
    st.caption(f"Charts show {len(view):,} of {len(df):,} students (the Data Quality tab always uses the full raw file).")

    tabs = st.tabs(["Data quality", "Distributions", "Features vs Result", "Correlations"])
    with tabs[0]:
        _data_quality_tab(raw)

    if view.empty:
        for t in tabs[1:]:
            with t:
                st.warning("No students match the filters.")
        return

    with tabs[1]:
        col = st.selectbox("Feature", NUMERIC_FEATURES + CATEGORICAL_FEATURES, format_func=lambda f: FEATURE_LABELS[f], key="an_dist")
        l, r = st.columns([2, 1])
        l.plotly_chart(charts.distribution(view.dropna(subset=[col]), col), key="an_dist_chart")
        r.plotly_chart(charts.class_distribution(view), key="an_pie")

    with tabs[2]:
        st.markdown("Each box plot compares Pass and Fail students. Higher boxes for Pass suggest a helpful factor.")
        cols = st.columns(2)
        for i, feature in enumerate(KEY_NUMERIC):
            cols[i % 2].plotly_chart(charts.box_vs_result(view.dropna(subset=[feature]), feature), key=f"an_box_{feature}")
        cols[1].plotly_chart(charts.pass_rate_by_category(view.dropna(subset=["Participation"]), "Participation",
                                                          CATEGORY_VALUES["Participation"]), key="an_part")
        other = st.selectbox("Pass rate by another category", ["Gender", "Internet_Access", "Extracurricular"],
                             format_func=lambda f: FEATURE_LABELS[f], key="an_cat")
        st.plotly_chart(charts.pass_rate_by_category(view, other), key="an_cat_chart")

    with tabs[3]:
        st.plotly_chart(charts.correlation_heatmap(view), key="an_corr")
        st.caption("Correlation shows that two things move together, not that one causes the other.")
        x, y = st.columns(2)
        xf = x.selectbox("X axis", NUMERIC_FEATURES, index=NUMERIC_FEATURES.index("Study_Hours"), format_func=lambda f: FEATURE_LABELS[f], key="an_x")
        yf = y.selectbox("Y axis", NUMERIC_FEATURES, index=NUMERIC_FEATURES.index("Previous_Score"), format_func=lambda f: FEATURE_LABELS[f], key="an_y")
        st.plotly_chart(charts.scatter(view.dropna(subset=[xf, yf]), xf, yf), key="an_scatter")
