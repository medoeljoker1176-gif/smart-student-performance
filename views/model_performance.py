"""Model Performance - comparison, selection, risk tiers and fairness."""
import pandas as pd
import streamlit as st

from src import charts, ui
from src.calibration import calibration_table, expected_calibration_error
from src.config import HIGH_RISK_BELOW, LOW_RISK_FROM
from src.fairness import fairness_summary

METRIC_NAMES = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"}


def _test_table(metrics: dict) -> pd.DataFrame:
    rows = [{"Model": n, **{METRIC_NAMES[k]: round(m[k], 3) for k in METRIC_NAMES}} for n, m in metrics["test_metrics"].items()]
    return pd.DataFrame(rows).set_index("Model")


def _cv_table(metrics: dict) -> pd.DataFrame:
    rows = [{"Model": n, **{METRIC_NAMES[k]: round(m[k]["mean"], 3) for k in METRIC_NAMES},
             "ROC-AUC std": round(m["roc_auc"]["std"], 3)} for n, m in metrics["cv_metrics"].items()]
    return pd.DataFrame(rows).set_index("Model")


def _comparison_tab(metrics: dict):
    final = metrics["final_model"]
    cv, test = _cv_table(metrics), _test_table(metrics)
    st.success(f"Final model: **{final}**")
    with st.container(border=True):
        st.markdown("##### Why was this model selected?")
        st.markdown(
            f"- **Metric:** cross-validated **ROC-AUC** on the training data ({metrics['cv']}). Risk levels are built from "
            "probabilities, so the model's ability to *rank* students from likely-fail to likely-pass matters most.\n"
            f"- **Best CV score:** {metrics['best_cv_model']} ({cv.loc[metrics['best_cv_model'], 'ROC-AUC']:.3f}).\n"
            f"- **Simplicity rule:** models within one standard error of the best are treated as tied "
            f"({', '.join(metrics['tied_models'])}); the simplest of them is chosen because it is easier to explain and less likely to over-fit.\n"
            "- **The test set was not used for the choice** - it is only the final exam.")

    st.markdown("##### Cross-validation on training data (used for selection)")
    st.dataframe(cv)
    st.plotly_chart(charts.metric_comparison(cv, list(METRIC_NAMES.values()), "Cross-validated metrics"), key="mp_cv")
    st.markdown("##### Held-out test set (final check)")
    st.dataframe(test)
    st.plotly_chart(charts.metric_comparison(test, list(METRIC_NAMES.values()), "Test-set metrics"), key="mp_test")
    st.plotly_chart(charts.roc_curves(metrics["test_metrics"]), key="mp_roc")
    st.caption(f"Test set = {metrics['n_test']} students, training set = {metrics['n_train']} students. "
               "With this few test students, small differences between models are not meaningful.")


def _selected_tab(metrics: dict):
    final = metrics["final_model"]
    m = metrics["test_metrics"][final]
    st.markdown(f"##### {final}")
    st.write("Best hyper-parameters found by grid search:", metrics["best_params"][final])
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.confusion_matrix_fig(m["confusion_matrix"], "Confusion matrix (test set)"), key="mp_cm")
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    with c2:
        st.markdown(f"- **Precision {m['precision']:.2f}**: of the {tp + fp} students predicted Pass, {tp} really passed.\n"
                    f"- **Recall {m['recall']:.2f}**: of the {tp + fn} students who really passed, {tp} were found.\n"
                    f"- **F1 {m['f1']:.2f}**: balance between precision and recall.\n"
                    f"- **Brier score {m['brier']:.3f}**: quality of the probabilities (lower is better).\n"
                    f"- {fp} failing students were predicted to pass (missed warnings) and {fn} passing students were flagged as fail.")
    st.markdown("##### How is the risk level calculated?")
    st.markdown(f"The model gives P(pass). **High risk** if P(pass) < {HIGH_RISK_BELOW:.0%} (the model predicts Fail); "
                f"**Moderate risk** if {HIGH_RISK_BELOW:.0%} <= P(pass) < {LOW_RISK_FROM:.0%}; **Low risk** if P(pass) >= {LOW_RISK_FROM:.0%}. "
                "These thresholds are design choices for this prototype and are **not** scientifically validated.")
    tiers = metrics["risk_tiers"]
    st.plotly_chart(charts.risk_tier_chart(tiers), key="mp_tiers")
    st.caption("Sanity check on out-of-fold predictions for all students: the higher the risk tier, the lower the observed pass rate. "
               + " | ".join(f"{t['tier']}: {t['students']} students" for t in tiers))
    _calibration_section()


def _calibration_section():
    st.markdown("##### Can the probabilities be trusted? (calibration)")
    oof = ui.load_oof()
    if oof is None or not {"y_true", "p_pass"} <= set(oof.columns):
        st.info("Calibration needs the out-of-fold predictions created during training. Run `python train_model.py`.")
        return
    table = calibration_table(oof["y_true"], oof["p_pass"])
    if table.empty:
        st.info("Not enough predictions to draw a calibration curve.")
        return
    ece = expected_calibration_error(table)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(charts.calibration_curve(table), key="mp_calib")
    with c2:
        st.dataframe(table.round(3), hide_index=True)
        st.metric("Expected calibration error", f"{ece:.3f}", help="Average gap between predicted and observed pass rate. Lower is better.")
    st.caption("Students are grouped by predicted P(pass). If the model says about 70%, roughly 70% of those students should really "
               "pass. Points on the dashed line mean well-calibrated probabilities. Bins with few students are noisy.")


def _fairness_tab():
    oof = ui.load_oof()
    if oof is None:
        st.warning("Fairness needs the out-of-fold predictions created during training. Run `python train_model.py`.")
        return
    out = fairness_summary(oof, "Gender")
    st.markdown("Compares how the model behaves for **Male** and **Female** students, using out-of-fold predictions "
                "(each student was predicted by a model that never saw them). Positive outcome = predicted Pass.")
    table = out["table"]
    st.dataframe(table.round(3), hide_index=True)
    st.plotly_chart(charts.fairness_bars(table, "Gender"), key="mp_fair")
    if "gaps" in out:
        st.markdown("**Gaps between groups:** " + " | ".join(f"{k}: {v * 100:.1f} pts" for k, v in out["gaps"].items()))
        p = out["p_value_pass_rate"]
        st.caption(f"Two-proportion test on the predicted pass rate: p = {p:.2f} "
                   f"({'no statistically clear difference' if p >= 0.05 else 'a statistically detectable difference'}).")
    if out["small_groups"]:
        st.warning("At least one group has fewer than 30 students; the numbers are unreliable.")
    st.warning(f"**Important:** the dataset is {ui.data_kind()}. Similar numbers for two groups do **not** prove the model is unbiased, "
               "and these results cannot be generalised to other schools or students. Fairness also needs checks on other groups "
               "and on how the data was collected.")
    st.caption("FPR = failing students wrongly predicted Pass. FNR = passing students wrongly predicted Fail.")


def _training_tab(metrics: dict):
    st.markdown(f"- **Trained at:** {metrics['trained_at']}\n- **Split:** {metrics['n_train']} train / {metrics['n_test']} test, stratified, fixed random seed\n"
                f"- **Cross-validation:** {metrics['cv']} on the training set\n"
                "- **Tuning:** small `GridSearchCV` per model, refit on ROC-AUC\n"
                "- **Data-leakage prevention:** split first; imputation, scaling and encoding live inside a scikit-learn "
                "`Pipeline` that is fitted on training folds only; the test set is used once for the final report.")
    st.markdown("**Best hyper-parameters per model**")
    st.dataframe(pd.DataFrame([{"Model": n, "Parameters": str(p)} for n, p in metrics["best_params"].items()]), hide_index=True)
    if ui.is_real_data():
        st.info("On this real dataset the earlier-period grades (G1, G2) are very strong predictors, so simple models "
                "often do about as well as complex ones. The result describes one school, not students in general.")
    else:
        st.info("Logistic Regression can win here because the synthetic labels were generated from a mostly linear rule. "
                "On real data, tree-based models may do better.")


def _glossary():
    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **Accuracy**: share of all students classified correctly (Pass or Fail).\n"
            "- **Precision**: of the students predicted *Pass*, the share who really passed.\n"
            "- **Recall**: of the students who really passed, the share the model found.\n"
            "- **F1-score**: balance of precision and recall (their harmonic mean).\n"
            "- **ROC-AUC**: how well the model ranks a random passing student above a random failing one "
            "(0.5 = guessing, 1.0 = perfect).\n"
            "- **Brier score**: quality of the probabilities themselves (lower is better).\n\n"
            "**Which data each table uses:** *cross-validation* (CV) metrics are averages over validation folds of the "
            "training set and were used to choose the model; *test* metrics come from a held-out set used once at the end. "
            "No separate fixed validation split exists in this project.")


def render():
    ui.hero("Model Performance", "How the models were trained, compared and why the final one was selected.")
    metrics = ui.load_metrics()
    if metrics is None:
        st.error("No training report found (reports/metrics.json). Metrics are only shown when they have been computed.",
                 icon=":material/error:")
        st.code("python train_model.py", language="bash")
        return
    st.warning(f"All figures below come from a **{ui.data_kind()}** dataset (metrics.json). They show that the method works "
               "on this data and do **not** establish how the model would perform on other students.", icon=":material/warning:")
    _glossary()
    t1, t2, t3, t4 = st.tabs(["Model comparison", "Selected model", "Fairness (Gender)", "Training details"])
    with t1:
        _comparison_tab(metrics)
    with t2:
        _selected_tab(metrics)
    with t3:
        _fairness_tab()
    with t4:
        _training_tab(metrics)
    ui.disclaimer(compact=True)
