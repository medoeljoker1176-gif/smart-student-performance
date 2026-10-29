"""
charts.py
---------
All Plotly figures in one place. Every function takes real data (dataset, metrics.json,
explanations) and returns a figure; nothing is hard-coded.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import FEATURE_LABELS, NUMERIC_FEATURES, TARGET

GREEN, RED, AMBER, BLUE, GREY = "#1FA774", "#E5534B", "#F5A524", "#0A9BC4", "#8FA6BC"
COLORWAY = ["#0A9BC4", "#1FA774", "#5AD0F5", "#F5A524", "#4A47C9", "#E5534B"]
RESULT_COLORS = {"Pass": GREEN, "Fail": RED}
TEMPLATE = "plotly_white"


def _style(fig, height=340, title=None):
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      title=dict(text=title, x=0, font=dict(color="#0B2C5E")) if title else None, legend_title_text="",
                      colorway=COLORWAY, font=dict(color="#16324F"), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ------------------------------------------------------------ gauges
def gauge(value: float, title: str, suffix: str = "", color: str = BLUE, height: int = 230):
    """Semi-circle gauge for a 0-100 value."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": suffix, "font": {"size": 38}},
        title={"text": title, "font": {"size": 15}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": 0.35},
               "steps": [{"range": [0, 50], "color": "#FDECEA"}, {"range": [50, 75], "color": "#FFF4E0"},
                         {"range": [75, 100], "color": "#DFF5EA"}]}))
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ------------------------------------------------------------ analytics
def class_distribution(df: pd.DataFrame):
    counts = df[TARGET].value_counts().reset_index()
    counts.columns = [TARGET, "Students"]
    fig = px.pie(counts, names=TARGET, values="Students", hole=0.5, color=TARGET, color_discrete_map=RESULT_COLORS)
    return _style(fig, 300, "Pass vs Fail")


def distribution(df: pd.DataFrame, col: str):
    """Histogram (numeric) or grouped bars (categorical), split by Result."""
    if col in NUMERIC_FEATURES:
        fig = px.histogram(df, x=col, color=TARGET, barmode="overlay", opacity=0.65, nbins=25,
                           color_discrete_map=RESULT_COLORS, labels={col: FEATURE_LABELS[col]})
    else:
        counts = df.groupby([col, TARGET]).size().reset_index(name="Students")
        fig = px.bar(counts, x=col, y="Students", color=TARGET, barmode="group",
                     color_discrete_map=RESULT_COLORS, labels={col: FEATURE_LABELS[col]})
    return _style(fig, 340, f"Distribution of {FEATURE_LABELS[col]}")


def box_vs_result(df: pd.DataFrame, col: str):
    fig = px.box(df, x=TARGET, y=col, color=TARGET, color_discrete_map=RESULT_COLORS,
                 category_orders={TARGET: ["Fail", "Pass"]}, points="outliers", labels={col: FEATURE_LABELS[col]})
    fig.update_layout(showlegend=False)
    return _style(fig, 320, f"{FEATURE_LABELS[col]} vs Result")


def pass_rate_by_category(df: pd.DataFrame, col: str, order=None):
    rate = (df.assign(Passed=(df[TARGET] == "Pass")).groupby(col)["Passed"].agg(["mean", "size"]).reset_index())
    rate["Pass rate (%)"] = rate["mean"] * 100
    fig = px.bar(rate, x=col, y="Pass rate (%)", text=rate["Pass rate (%)"].round(1), hover_data={"size": True},
                 category_orders={col: order} if order else None, color_discrete_sequence=[BLUE],
                 labels={col: FEATURE_LABELS[col]})
    fig.update_yaxes(range=[0, 100])
    return _style(fig, 320, f"Pass rate by {FEATURE_LABELS[col]}")


def scatter(df: pd.DataFrame, x: str, y: str):
    fig = px.scatter(df, x=x, y=y, color=TARGET, color_discrete_map=RESULT_COLORS, opacity=0.7,
                     labels={x: FEATURE_LABELS[x], y: FEATURE_LABELS[y]})
    return _style(fig, 400, f"{FEATURE_LABELS[y]} vs {FEATURE_LABELS[x]}")


def correlation_heatmap(df: pd.DataFrame):
    corr_df = df[NUMERIC_FEATURES].copy()
    corr_df["Passed (1/0)"] = (df[TARGET] == "Pass").astype(int)
    fig = px.imshow(corr_df.corr().round(2), text_auto=True, zmin=-1, zmax=1, aspect="auto",
                    color_continuous_scale="RdBu_r")
    return _style(fig, 460, "Correlation heat-map")


def missing_values_bar(missing: dict):
    data = pd.DataFrame({"Column": list(missing), "Missing values": list(missing.values())})
    fig = px.bar(data, x="Column", y="Missing values", color_discrete_sequence=[AMBER], text="Missing values")
    return _style(fig, 300, "Missing values per column")


# ------------------------------------------------------------ explanations
def contributions_bar(table: pd.DataFrame, title="How each factor moved the probability of passing"):
    """Horizontal bars: green = increases P(pass), red = decreases (in percentage points)."""
    data = table.sort_values("Contribution")
    fig = go.Figure(go.Bar(
        x=data["Contribution"], y=[f"{l} = {v}" for l, v in zip(data["Label"], data["Value"])], orientation="h",
        marker_color=[GREEN if c >= 0 else RED for c in data["Contribution"]],
        text=[f"{c:+.1f}" for c in data["Contribution"]], textposition="outside"))
    fig.update_xaxes(title="Change in P(pass), percentage points", zeroline=True, zerolinecolor="#444")
    return _style(fig, 420, title)


def global_importance_bar(importance: dict):
    data = pd.Series(importance).sort_values()
    fig = go.Figure(go.Bar(x=data.values, y=[FEATURE_LABELS[k] for k in data.index], orientation="h",
                           marker_color=BLUE, text=[f"{v:.1f}" for v in data.values], textposition="outside"))
    fig.update_xaxes(title="Average absolute impact on P(pass), percentage points")
    return _style(fig, 420, "Global feature importance (mean |Shapley value|)")


# ------------------------------------------------------------ model performance
def metric_comparison(rows: pd.DataFrame, metrics: list, title: str):
    """rows: index = model names, columns include the metrics (0-1)."""
    long = rows[metrics].reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", text=long["Score"].round(3))
    fig.update_yaxes(range=[0, 1.05])
    fig.update_traces(textposition="outside", textfont_size=9)
    return _style(fig, 420, title)


def roc_curves(test_metrics: dict):
    fig = go.Figure()
    for name, m in test_metrics.items():
        fig.add_trace(go.Scatter(x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"], mode="lines",
                                 name=f"{name} (AUC {m['roc_auc']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=GREY),
                             name="Random guessing"))
    fig.update_xaxes(title="False positive rate"); fig.update_yaxes(title="True positive rate (recall)")
    return _style(fig, 420, "ROC curves (test set)")


def confusion_matrix_fig(cm: list, title: str):
    """cm = [[TN, FP], [FN, TP]] with Fail = 0 and Pass = 1."""
    fig = px.imshow(cm, text_auto=True, x=["Predicted Fail", "Predicted Pass"], y=["Actual Fail", "Actual Pass"],
                    color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(coloraxis_showscale=False)
    return _style(fig, 320, title)


def risk_tier_chart(tiers: list):
    data = pd.DataFrame(tiers).dropna()
    data["Observed pass rate (%)"] = data["observed_pass_rate"] * 100
    fig = px.bar(data, x="tier", y="Observed pass rate (%)", color="tier", text=data["Observed pass rate (%)"].round(1),
                 color_discrete_map={"High Risk": RED, "Moderate Risk": AMBER, "Low Risk": GREEN},
                 category_orders={"tier": ["High Risk", "Moderate Risk", "Low Risk"]}, hover_data={"students": True})
    fig.update_yaxes(range=[0, 100]); fig.update_layout(showlegend=False)
    return _style(fig, 320, "Share of students who really passed, per risk tier")


def calibration_curve(table: pd.DataFrame):
    """Reliability diagram: points on the diagonal mean the probabilities can be taken at face value."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=GREY), name="Perfect calibration"))
    fig.add_trace(go.Scatter(x=table["Mean predicted"], y=table["Observed pass rate"], mode="lines+markers",
                             name="Model", text=[f"{n} students" for n in table["Students"]],
                             hovertemplate="Predicted %{x:.2f}<br>Observed %{y:.2f}<br>%{text}<extra></extra>"))
    fig.update_xaxes(title="Mean predicted P(pass)", range=[0, 1]); fig.update_yaxes(title="Observed pass rate", range=[0, 1])
    return _style(fig, 360, "Calibration curve (out-of-fold predictions)")


def fairness_bars(table: pd.DataFrame, group_col: str):
    metrics = ["Predicted pass rate", "Recall (TPR)", "False positive rate", "False negative rate"]
    long = table.melt(id_vars=group_col, value_vars=metrics, var_name="Metric", value_name="Rate")
    fig = px.bar(long, x="Metric", y="Rate", color=group_col, barmode="group", text=long["Rate"].round(2))
    fig.update_yaxes(range=[0, 1.05])
    return _style(fig, 380, f"Model behaviour by {group_col}")


# ------------------------------------------------------------ comparison / history
def compare_bars(a: dict, b: dict, keys: list, title: str, height=340):
    fig = go.Figure()
    for name, data, color in (("Student A", a, BLUE), ("Student B", b, AMBER)):
        fig.add_trace(go.Bar(name=name, x=[FEATURE_LABELS.get(k, k) for k in keys], y=[data[k] for k in keys],
                             marker_color=color, text=[f"{data[k]:g}" for k in keys], textposition="outside"))
    fig.update_layout(barmode="group")
    return _style(fig, height, title)


def contribution_compare(table_a: pd.DataFrame, table_b: pd.DataFrame):
    a, b = table_a.set_index("Feature")["Contribution"], table_b.set_index("Feature")["Contribution"]
    order = (a.abs() + b.abs()).sort_values().index
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Student A", y=[FEATURE_LABELS[f] for f in order], x=a[order], orientation="h", marker_color=BLUE))
    fig.add_trace(go.Bar(name="Student B", y=[FEATURE_LABELS[f] for f in order], x=b[order], orientation="h", marker_color=AMBER))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Change in P(pass), percentage points")
    return _style(fig, 480, "What drives each student's prediction")


def history_trend(df: pd.DataFrame):
    """P(pass) over time for one student, with the risk thresholds drawn."""
    data = df.copy()
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data["P(pass) %"] = data["P_Pass"] * 100
    fig = px.line(data, x="Timestamp", y="P(pass) %", markers=True, color_discrete_sequence=[BLUE],
                  hover_data=["Risk", "Performance_Score"])
    fig.add_hline(y=75, line_dash="dot", line_color=GREEN, annotation_text="Low risk from 75%")
    fig.add_hline(y=50, line_dash="dot", line_color=RED, annotation_text="High risk below 50%")
    fig.update_yaxes(range=[0, 100])
    return _style(fig, 340, "Probability of passing over time")
