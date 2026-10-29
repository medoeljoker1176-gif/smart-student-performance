# Demo Guide — Project Discussion

## 1. Set up

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
```

If PowerShell blocks activation, run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and use `.venv\Scripts\python.exe` for commands.

## 2. Start the application

```powershell
streamlit run app.py
```

If the model is missing or was created with an incompatible scikit-learn version, run:

```powershell
python train_model.py
```

Training writes the model bundle, metrics, out-of-fold predictions, and plots into `models/` and `reports/`.

## 3. Suggested live demo (5–7 minutes)

1. **Dashboard:** explain that all current records and metrics come from synthetic data.
2. **Prediction:** load the built-in “At-risk student” example and run a prediction.
3. **Explanation:** show local Shapley contributions; explain that they describe model behavior, not causation.
4. **What-if simulator:** change one input and compare the model probability; stress that this is hypothetical.
5. **Recommendations:** explain the rule-based suggestions and model-based what-if estimate.
6. **Batch CSV Analysis:** download the template, upload a small edited copy, analyze, and download the results/errors.
7. **Model Performance:** explain train/test separation, cross-validation, confusion matrix, ROC-AUC, and the synthetic-data limitation.
8. **History:** show that repeated predictions with the same anonymous ID can create a local trend; clear the history after the demo.
9. **PDF report:** download and open a student report.

Do not use real student information in the demo. Use anonymous labels.

## 4. Final checklist

- [ ] `python -m pytest -q` passes.
- [ ] `streamlit run app.py` starts without errors.
- [ ] Prediction, explanation, what-if, recommendations, CSV export, PDF export, model-performance tabs, and history have been manually checked.
- [ ] PDF opens and CSV downloads have the expected columns.
- [ ] `reports/metrics.json` matches the current model training run.
- [ ] No personal data or credentials are included in the release ZIP.
- [ ] Screenshots in `docs/screenshots/` are captured from the actual running application; do not use mockups as evidence of functionality.

## 5. Likely discussion questions

**Why use a pipeline?** It keeps imputation, scaling, encoding, and the estimator together so preprocessing is fitted only on training data and applied consistently at inference.

**What prevents data leakage?** The split is made first; learned preprocessing and hyperparameter selection happen only within training data/cross-validation. The held-out test split is reserved for final evaluation.

**Why report ROC-AUC?** It measures ranking across thresholds. Precision, recall, F1, confusion matrix, and Brier score add complementary information.

**Does 82% accuracy mean it works on real students?** No. The included labels are synthetic and generated from a known artificial model. Real-world validation is still required.

**Do Shapley values show causes?** No. They attribute the model's output across features; they do not establish real-world causation.

**Can the system make academic decisions?** No. It is a prototype for demonstration and supportive human review only.
