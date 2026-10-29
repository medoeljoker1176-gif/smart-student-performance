"""
model_training.py
-----------------
Full, reproducible training workflow.

  1. load + validate + clean data        6. pick the final model (rule below)
  2. EDA plots                           7. out-of-fold predictions (fairness, risk tiers)
  3. stratified train/test split         8. global Shapley importance
  4. tune 4 models with cross-validation 9. save model bundle + reports
  5. evaluate on the untouched test set

Run:  python train_model.py
"""
import json
from pathlib import Path
from datetime import datetime
import joblib
import matplotlib
matplotlib.use("Agg")                         # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import (CATEGORICAL_FEATURES, FEATURES, HIGH_RISK_BELOW, LOW_RISK_FROM,
                        METRICS_PATH, MODEL_PATH, N_BACKGROUND, NUMERIC_FEATURES, OOF_PATH,
                        RANDOM_STATE, REPORT_DIR, TARGET, DEFAULT_DATASET, dataset_paths)
from src.explainability import global_importance
from src.preprocessing import build_preprocessor, clean_data, load_dataset, split_xy

SELECTION_METRIC = "roc_auc"
SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc"}


def get_candidates() -> dict:
    """name -> (estimator, small hyper-parameter grid). Listed from simplest to most complex.

    Logistic Regression and Decision Tree are the original baselines. Random Forest and
    Gradient Boosting are added because they can capture non-linear effects and
    interactions that the baselines cannot. KNN was left out: it needs no extra insight
    here and is slow and hard to explain. Grids are kept small so training takes seconds.
    """
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                                {"model__C": [0.1, 1.0]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                          {"model__max_depth": [3, 5], "model__min_samples_leaf": [10]}),
        "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
                          {"model__max_depth": [4, 8], "model__min_samples_leaf": [3, 8]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=RANDOM_STATE),
                              {"model__n_estimators": [50], "model__learning_rate": [0.05, 0.1],
                               "model__max_depth": [2]}),
    }


def make_pipeline(model) -> Pipeline:
    """Preprocessing + model in ONE object -> everything is fitted on training data only."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


# ------------------------------------------------------------------ EDA plots
def empty_features(df: pd.DataFrame) -> list:
    """Features with no data at all (e.g. sleep hours in the UCI dataset)."""
    return [c for c in FEATURES if df[c].isna().all()]


def complete_rows(X: pd.DataFrame, empty: list) -> pd.DataFrame:
    """Rows with no gaps. Features that do not exist in a dataset are filled with a neutral value
    (the model ignores them) so they do not wipe out every row."""
    neutral = {"Sleep_Hours": 7.0, "Participation": "Medium"}
    X = X.copy()
    for col in empty:
        X[col] = neutral.get(col, 0)
    return X.dropna()


def save_eda_plots(df: pd.DataFrame, report_dir=REPORT_DIR) -> None:
    REPORT_DIR = Path(report_dir)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    num_cols = [c for c in NUMERIC_FEATURES if df[c].notna().any()]
    cat_cols = [c for c in CATEGORICAL_FEATURES if df[c].notna().any()]
    sns.set_theme(style="whitegrid")
    palette = {"Pass": "#2E8B57", "Fail": "#D9534F"}

    plt.figure(figsize=(4, 4))
    sns.countplot(data=df, x=TARGET, hue=TARGET, palette=palette, legend=False)
    plt.title("Class distribution")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "class_distribution.png", dpi=120); plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax in axes.ravel()[len(num_cols):]:
        ax.axis("off")
    for ax, col in zip(axes.ravel(), num_cols):
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, palette=palette, legend=False, ax=ax)
        ax.set_title(f"{col} vs Result")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "numeric_vs_result.png", dpi=120); plt.close()

    fig, axes = plt.subplots(1, len(cat_cols), figsize=(15, 4))
    for ax, col in zip(np.atleast_1d(axes), cat_cols):
        rate = df.groupby(col)[TARGET].apply(lambda s: (s == "Pass").mean() * 100)
        rate.plot(kind="bar", ax=ax, color="#4A90D9")
        ax.set_title(f"Pass rate (%) by {col}"); ax.set_ylabel("Pass %"); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
    plt.tight_layout(); plt.savefig(REPORT_DIR / "categorical_vs_result.png", dpi=120); plt.close()

    corr_df = df[num_cols].copy()
    corr_df["Passed"] = (df[TARGET] == "Pass").astype(int)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heat-map")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=120); plt.close()


# ------------------------------------------------------------------ evaluation
def evaluate(pipeline, X_test, y_test) -> dict:
    """Test-set metrics with Pass (1) as the positive class."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba)
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(80, len(fpr))).astype(int))
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "brier": brier_score_loss(y_test, proba),          # lower = better calibrated probabilities
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),   # [[TN, FP], [FN, TP]]
        "roc_curve": {"fpr": np.round(fpr[keep], 4).tolist(), "tpr": np.round(tpr[keep], 4).tolist()},
    }


def select_final_model(cv_summary: dict, complexity_order: list) -> tuple:
    """Selection rule (simple and explainable):

    1. Metric = cross-validated ROC-AUC on the TRAINING data. We use ROC-AUC because
       risk levels are built from probabilities, so what matters is how well the model
       RANKS students from likely-fail to likely-pass (it does not depend on one threshold).
    2. "One-standard-error rule": every model whose mean AUC is within one standard error
       of the best is considered equally good; among those we choose the SIMPLEST one.
       Simpler models are easier to explain and less likely to over-fit.
    The test set is NOT used for selection.
    """
    best_name = max(cv_summary, key=lambda n: cv_summary[n]["roc_auc"]["mean"])
    best = cv_summary[best_name]["roc_auc"]
    limit = best["mean"] - best["se"]
    tied = [n for n in complexity_order if cv_summary[n]["roc_auc"]["mean"] >= limit]
    return tied[0], best_name, tied


def risk_tier_validation(y_true, p_pass) -> list:
    """Observed pass rate inside each risk tier (out-of-fold data) - checks the tiers make sense."""
    tiers = [("High Risk", p_pass < HIGH_RISK_BELOW),
             ("Moderate Risk", (p_pass >= HIGH_RISK_BELOW) & (p_pass < LOW_RISK_FROM)),
             ("Low Risk", p_pass >= LOW_RISK_FROM)]
    out = []
    for name, mask in tiers:
        n = int(mask.sum())
        out.append({"tier": name, "students": n,
                    "observed_pass_rate": float(y_true[mask].mean()) if n else None})
    return out


# ------------------------------------------------------------------ main
def main(quiet: bool = False, dataset: str = DEFAULT_DATASET) -> dict:
    log = (lambda *a, **k: None) if quiet else print
    paths = dataset_paths(dataset)
    REPORT_DIR = paths["reports"]                       # local names shadow the default-dataset constants
    MODEL_PATH, METRICS_PATH, OOF_PATH = paths["model"], paths["metrics"], paths["oof"]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. data
    raw = load_dataset(paths["data"])
    df = clean_data(raw)
    log(f"Dataset '{dataset}': loaded {len(raw)} rows -> {len(df)} after cleaning")
    save_eda_plots(df, REPORT_DIR)
    missing_features = empty_features(df)
    if missing_features:
        log(f"Features with no data in this dataset (ignored by the model): {missing_features}")
    X, y = split_xy(df)

    # 2. split BEFORE anything is fitted; stratify keeps the Pass/Fail ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    log(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # 3. tune every candidate with stratified 5-fold CV on the TRAINING data only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    n_folds = cv.get_n_splits()
    models, cv_summary, test_results, best_params = {}, {}, {}, {}
    candidates = get_candidates()
    for name, (estimator, grid) in candidates.items():
        search = GridSearchCV(make_pipeline(estimator), grid, scoring=SCORING,
                              refit=SELECTION_METRIC, cv=cv, n_jobs=1)
        search.fit(X_train, y_train)
        i = search.best_index_
        cv_summary[name] = {m: {"mean": float(search.cv_results_[f"mean_test_{m}"][i]),
                                "std": float(search.cv_results_[f"std_test_{m}"][i]),
                                "se": float(search.cv_results_[f"std_test_{m}"][i] / np.sqrt(n_folds))}
                            for m in SCORING}
        best_params[name] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        models[name] = search.best_estimator_                  # refitted on the full training set
        test_results[name] = evaluate(models[name], X_test, y_test)
        log(f"{name:20s} CV AUC {cv_summary[name]['roc_auc']['mean']:.3f} | "
            f"test AUC {test_results[name]['roc_auc']:.3f} F1 {test_results[name]['f1']:.3f} | {best_params[name]}")

    # 4. choose the final model
    order = list(candidates)                                    # simplest -> most complex
    final_name, top_name, tied = select_final_model(cv_summary, order)
    final = models[final_name]
    log(f">>> Final model: {final_name} (best CV AUC: {top_name}; within 1 SE: {tied})")

    # 5. confusion-matrix images
    for name, res in test_results.items():
        ConfusionMatrixDisplay(np.array(res["confusion_matrix"]), display_labels=["Fail", "Pass"]).plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(REPORT_DIR / f"confusion_matrix_{name.replace(' ', '_').lower()}.png", dpi=120)
        plt.close()

    # 6. out-of-fold predictions on ALL students (used for fairness and risk-tier checks)
    oof_p = cross_val_predict(clone(final), X, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                              method="predict_proba")[:, 1]
    oof = X.copy()
    oof["y_true"], oof["p_pass"], oof["y_pred"] = y.to_numpy(), oof_p.round(4), (oof_p >= 0.5).astype(int)
    oof.to_csv(OOF_PATH, index=False)
    tiers = risk_tier_validation(y.to_numpy(), oof_p)

    # 7. explanations: reference students + global Shapley importance
    complete = complete_rows(X_train, missing_features)
    background = complete.sample(min(N_BACKGROUND, len(complete)), random_state=RANDOM_STATE).reset_index(drop=True)
    predict_fn = lambda d: final.predict_proba(d[FEATURES])[:, 1]
    # Global explanations are expensive: 2^10 feature subsets per student.
    # Use a reproducible, representative sample and a smaller background for this
    # one-time summary; per-student explanations still use the full background.
    global_background = background.sample(min(20, len(background)), random_state=RANDOM_STATE).reset_index(drop=True)
    test_complete = complete_rows(X_test, missing_features)
    sample = test_complete.sample(min(12, len(test_complete)), random_state=RANDOM_STATE)
    importance = global_importance(predict_fn, sample, global_background)

    # 8. save the bundle and the metrics
    ranges = {c: (float(X_train[c].min()), float(X_train[c].max()))
              for c in NUMERIC_FEATURES if c not in missing_features}
    bundle = {"pipeline": final, "model_name": final_name, "features": FEATURES,
              "background": background, "training_ranges": ranges,
              "dataset": dataset, "unavailable_features": missing_features,
              "risk_thresholds": {"high_below": HIGH_RISK_BELOW, "low_from": LOW_RISK_FROM},
              "sklearn_version": sklearn.__version__, "random_state": RANDOM_STATE,
              "trained_at": datetime.now().isoformat(timespec="seconds")}
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "final_model": final_name, "best_cv_model": top_name, "tied_models": tied,
        "selection_metric": "Cross-validated ROC-AUC (training data) with the one-standard-error rule",
        "n_train": len(X_train), "n_test": len(X_test), "cv": f"{n_folds}-fold stratified cross-validation (shuffled, fixed random seed)",
        "best_params": best_params, "cv_metrics": cv_summary, "test_metrics": test_results,
        "global_importance": importance, "risk_tiers": tiers,
        "trained_at": bundle["trained_at"], "dataset": dataset,
        "n_records": int(len(df)), "unavailable_features": missing_features,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log(f"Saved {MODEL_PATH.name}, {METRICS_PATH.name}, {OOF_PATH.name}")
    log("Global importance (points of P(pass)):", {k: round(v, 1) for k, v in importance.items()})
    log("Risk tiers (out-of-fold):", tiers)
    return metrics


if __name__ == "__main__":
    main()
