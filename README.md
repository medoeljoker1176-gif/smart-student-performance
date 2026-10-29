# 🎓 Smart Student Performance & Early Risk Detection AI System

An educational machine-learning application that goes beyond *Pass / Fail*. For a student profile it predicts the **probability of passing**, assigns a **risk level**, **explains why** with Shapley values, and gives **prioritised, personalised recommendations**. Predictions can be saved to track progress over time.

> ⚠️ **This system is an educational AI prototype based on synthetic data. Predictions are intended for demonstration and early-warning purposes and should not be treated as an official academic judgment.**

> 🌐 **Live demo:** _add your Streamlit Cloud link here after deployment_ (`https://<your-app>.streamlit.app`) · Deployment guide: [DEPLOYMENT.md](DEPLOYMENT.md)

> ☁️ **Cloud version:** the project can be deployed as an auto-scaling container on Google Cloud Run with a Cloud Firestore history store. See [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md) and the measurement worksheet in [docs/CLOUD_EVALUATION.md](docs/CLOUD_EVALUATION.md).

## Project overview
A scikit-learn pipeline (preprocessing + tuned classifier) is trained on 1,000 synthetic students. A multi-page Streamlit dashboard loads the saved model and shows the prediction, explanation, recommendations, analytics, model comparison and fairness checks.

## Problem statement
Poor academic performance is often noticed too late. Teachers and students need an early-warning tool that says not only *whether* a student is at risk, but *why* and *what to change*.

## Objectives
- Predict Pass/Fail and the probability of passing
- Turn the probability into an understandable risk level
- Explain each prediction and the model as a whole
- Recommend prioritised actions based on the student's own data
- Compare several models fairly, without data leakage
- Provide data-quality, analytics, fairness and history views

## Features
| Area | What it does |
|---|---|
| Prediction | Validated form, Pass/Fail, P(pass), risk level, performance score, example students |
| AI Explanation | Per-student Shapley contributions, global importance, what-if simulator |
| Recommendations | Priority 1/2/3 with a "why" and a model what-if gain, habit scorecard |
| Analytics | Data quality, distributions, features vs result, correlation heat-map, filters (Plotly) |
| Compare Students | Student A vs B: table, charts, what drives each prediction |
| Batch CSV Analysis | Upload a class CSV, per-row validation, predictions + rejected rows as separate downloads (max 5,000 rows) |
| PDF report | Download a student report (inputs, risk, explanation, recommendations, limitations) from the Prediction page |
| History | CSV history, per-student progress chart, download, clear |
| Model Performance | Model table + charts, ROC, confusion matrix, risk-tier check, fairness by gender |
| Reliability | Input validation, missing/corrupt model and dataset handled with friendly messages |

## Technologies
Python · pandas · NumPy · scikit-learn · Plotly · Matplotlib/Seaborn · Streamlit · joblib.
Explanations use our own **exact Shapley-value** implementation (no extra dependency).

## Dataset
`data/student_performance.csv` - 1,000 **synthetic** students created by `generate_dataset.py` (seed 42): 10 features, target `Result` (Pass/Fail, about 70% / 30%), about 3% of some values deliberately missing.

The generator builds students from hidden traits (ability, discipline, socio-economic background) so features are correlated like real classes: disciplined students study and attend more, able students have higher previous/assignment scores, heavy study trims sleep, study hours have diminishing returns and sleep has a 7-8 h sweet spot. `Result` is Pass when a hidden exam score reaches 50; that exam score is **not saved** (it would leak the label), and missing values never depend on `Result`. Same seed gives the same file; `generate_dataset.py` also checks that every value stays inside the app's input limits.

| Feature | Type | Meaning |
|---|---|---|
| Age | number | 18-25 in the data |
| Gender | category | Male / Female |
| Study_Hours | number | hours per day |
| Attendance | number | percent |
| Previous_Score | number | 0-100 |
| Assignment_Score | number | 0-100 |
| Sleep_Hours | number | hours per day |
| Participation | category | Low / Medium / High |
| Internet_Access | category | Yes / No |
| Extracurricular | category | Yes / No |
| **Result** | target | Pass / Fail |

Because the data is synthetic, results demonstrate the **method**, not real-world accuracy.

### Real data (UCI Student Performance) - optional second dataset
The app can also run on the real UCI *Student Performance* data (Cortez & Silva, 2008; CC BY 4.0), so results are not only on synthetic students. The original files (`student-mat.csv`, `student-por.csv`) are included in `data/raw/`, together with the converted CSV, the trained UCI model and `reports/uci/`, so the **Dataset** switch works out of the box. `data/raw/` is git-ignored; if it is missing, re-download the data and rebuild:
1. Get it from <https://archive.ics.uci.edu/dataset/320/student+performance>, unzip, and put `student-mat.csv` and/or `student-por.csv` in `data/raw/`.
2. `python prepare_uci.py` converts it to this project's schema (`data/uci_student_performance.csv`).
3. `python train_model.py --dataset uci` trains and saves a separate model and `reports/uci/`.
4. Commit `data/uci_student_performance.csv`, `models/uci_student_performance_model.pkl` and `reports/uci/`. The **Dataset** switch in the sidebar appears when these exist; the synthetic dataset stays the default.

Mapping (approximations, also in `docs/DATA_DICTIONARY.md`): `studytime` band -> hours/day, `absences` -> Attendance (assuming ~90 lessons), `G1` -> Previous score, `G2` -> Assignment score, `G3 >= 10` -> Pass. **Sleep hours and class participation do not exist in UCI**, so they stay empty and the model ignores them. `G3` is never a feature. Students found in both subject files are kept once. G1/G2 are very strong predictors, so expect high accuracy that says little about other schools.

## Login and roles
Login is switched on by adding an `[auth]` section to Streamlit secrets (see `.streamlit/secrets.toml.example`); without it the app runs open, as before. Passwords are stored only as salted PBKDF2-HMAC-SHA256 hashes (`python tools/make_password_hash.py <user> <teacher|student>`).

| Role | Pages |
|---|---|
| Teacher | all pages, including Dashboard, Analytics, Compare, History, Batch CSV, Model Performance |
| Student | Prediction, AI Explanation, Recommendations, About |

Pages a role may not open are never registered, so they cannot be reached by URL. Protection: wrong password limit (5 tries, then 5 min lock per username), identical error for unknown user and wrong password, 30-minute idle logout, input length limits. Limits: the lock counter is kept in server memory (resets on restart), no password reset or MFA, and a student account is a role, not a per-student identity. For a real deployment use an identity provider (OIDC).

## Machine learning workflow
`load + validate → clean → EDA → stratified 80/20 split → tune 4 models with stratified 5-fold CV → choose final model → evaluate once on the test set → out-of-fold predictions → global Shapley importance → save model bundle + reports`

## Preprocessing
Inside one scikit-learn `Pipeline`: numeric → median imputation → standard scaling; categorical → most-frequent imputation → one-hot encoding. Only row-wise fixes (duplicates, impossible values) happen before the split. **Everything that learns from data is fitted on training data only**, so there is no data leakage (a unit test checks this).

## Models
| Model | Role |
|---|---|
| Logistic Regression | baseline, simple and explainable |
| Decision Tree | baseline, easy to visualise |
| Random Forest | added: captures non-linear effects and interactions |
| Gradient Boosting | added: sequential trees, often strong on tabular data |

Each model is tuned with a small `GridSearchCV`. KNN was left out on purpose (adds nothing here and is hard to explain). HistGradientBoosting was not added because it would duplicate Gradient Boosting on 800 rows.

## Evaluation
Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix (plus Brier score for probability quality). Pass is the positive class.

**Selection rule:** highest cross-validated **ROC-AUC** on the training data (risk levels come from probabilities, so ranking quality matters), then the *one-standard-error rule*: among models that are statistically tied with the best, pick the simplest. The test set is not used for selection.

Results from the included run (test set = 200 students; regenerate with `python train_model.py`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** ✅ | 0.820 | 0.837 | 0.921 | 0.877 | 0.913 | 0.901 |
| Decision Tree | 0.810 | 0.818 | 0.935 | 0.872 | 0.846 | 0.861 |
| Random Forest | 0.825 | 0.829 | 0.942 | 0.882 | 0.903 | 0.894 |
| Gradient Boosting | 0.820 | 0.846 | 0.906 | 0.875 | 0.903 | 0.891 |

Logistic Regression wins partly because the synthetic labels come from a mostly additive rule; real data could favour tree-based models. With 200 test students, small differences between the top models are not meaningful. All numbers shown in the app are read from `reports/metrics.json`, never typed by hand.

## Explainable AI
For each student the app shows how many percentage points every factor added to or removed from the probability of passing, compared with an average reference student (Shapley values, computed **exactly** over all 1,024 factor subsets; the contributions always add up to the prediction). A global bar chart shows mean absolute contributions over a reproducible sample of 12 held-out students with a 20-student background, keeping model training practical. Per-student explanations still calculate exact Shapley values across all 1,024 feature subsets using the saved background set. It works for any of the four models.

## Risk detection
Based on the model's P(pass):

| P(pass) | Risk level |
|---|---|
| < 50% | 🔴 High risk (the model predicts Fail) |
| 50% - 75% | 🟠 Moderate risk |
| ≥ 75% | 🟢 Low risk |

The thresholds are **design choices, not scientifically validated cut-offs**. As a sanity check, the Model Performance page shows the observed pass rate in each tier on out-of-fold predictions (about 20% / 67% / 93% in the included run).

## Performance score
An application-level indicator from 0 to 100 (not a grade, not from the ML model): Previous score 25% + Attendance 25% + Assignment score 20% + Study hours 15% (100 at 6 h/day) + Participation 10% (0/60/100) + Sleep 5% (best at 7.5 h).

## Recommendation system
1. Rules compare inputs with rule-of-thumb targets (not scientific standards).
2. Each triggered rule gets a severity.
3. The trained model answers a what-if: "P(pass) if this habit reached the target?" → estimated gain.
4. Ranking by gain (severity breaks ties) → Priority 1, 2, 3 + additional, each with a "why".

## Deployment
The app can be deployed on **Streamlit Community Cloud** (free). Steps, Docker alternative and the screenshot checklist are in [DEPLOYMENT.md](DEPLOYMENT.md). A `Dockerfile` is included for Render / Hugging Face Spaces / Cloud Run. A GitHub Actions workflow (`.github/workflows/tests.yml`) runs the tests on every push.

## Application screenshots
_Screenshots are not included yet: run the app and add them in `docs/screenshots/` (names listed there)._

| Dashboard | Prediction | Explanation |
|---|---|---|
| ![](docs/screenshots/dashboard.png) | ![](docs/screenshots/prediction.png) | ![](docs/screenshots/explanation.png) |

## Project structure
```
smart-student-performance/
├── app.py                      Streamlit entry point + sidebar navigation
├── train_model.py              run this to train (calls src/model_training.py)
├── generate_dataset.py         recreate the synthetic dataset
├── prepare_uci.py              convert real UCI data to this schema (optional)
├── tools/make_password_hash.py create login password hashes
├── views/                      one file per dashboard page
│   ├── dashboard.py  prediction.py  explanation.py  analytics.py  batch_analysis.py
│   ├── recommendations.py  compare.py  history.py  model_performance.py  about.py
├── src/
│   ├── config.py               paths, features, limits, risk thresholds
│   ├── validation.py           input + dataset checks
│   ├── preprocessing.py        load, clean, data-quality report, preprocessor
│   ├── model_training.py       tuning, CV, evaluation, selection, saving
│   ├── prediction.py           model loading, prediction, risk, performance score
│   ├── explainability.py       exact Shapley values
│   ├── recommendations.py      prioritised recommendations
│   ├── batch.py                CSV batch validation + inference (no Streamlit)
│   ├── calibration.py          probability calibration table
│   ├── reporting.py  exports.py  PDF report, safe CSV export
│   ├── auth.py  auth_ui.py     optional login and roles
│   ├── fairness.py  history.py  charts.py  ui.py  ui_html.py
├── tests/                      core, batch, validation, auth, calibration, UCI, reporting, history,
│                               dataset-generation tests; *_apptest files need Streamlit (skipped otherwise)
├── .streamlit/config.toml      blue/white theme, 5 MB upload limit
├── pytest.ini  requirements.txt  requirements-dev.txt  requirements-lock-tested.txt
├── Dockerfile  .dockerignore  DEPLOYMENT.md  LICENSE  .github/workflows/tests.yml
├── data/  models/  reports/    dataset, saved model, metrics + plots
├── notebooks/model_training.ipynb   original step-by-step notebook
├── docs/                       STREAMLIT_SETUP.md, TEST_REPORT.md, model card, Q&A, screenshots
├── PRESENTATION_SLIDES.md      18-slide outline with speaker notes
└── requirements.txt
```
(`views/` is used instead of `pages/` because Streamlit auto-builds its own menu from a `pages/` folder, which would conflict with the custom navigation.)

## Supported Python and installation (Windows)
Python **3.11 or newer** (scikit-learn 1.8 requires it). Full guide: [docs/STREAMLIT_SETUP.md](docs/STREAMLIT_SETUP.md).
```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
For the tests also run `pip install -r requirements-dev.txt`.

## How to run (VS Code or any terminal)
```powershell
python -m streamlit run app.py
```
Open http://localhost:8501. Try **Prediction -> Load: At-risk student -> Predict**, then **AI Explanation** and **Recommendations**.
If the saved model cannot be loaded (for example a different pandas / scikit-learn version), the app retrains it automatically once (about 30-60 seconds). You can also rebuild it manually: `python train_model.py`.

## How to run the tests
```powershell
python -m pytest -q
python -m compileall -q app.py src views tests
```
`tests/test_ui_apptest.py` needs Streamlit and is skipped without it. See [docs/TEST_REPORT.md](docs/TEST_REPORT.md)
for what has and has not been verified.

## Batch CSV format
One row per student, header in the first line, UTF-8 (Excel "CSV UTF-8" is fine), at most 5,000 rows and 5 MB.

| Column | Allowed values |
|---|---|
| Age | whole number 15-40 |
| Gender | Male, Female |
| Study_Hours | 0-16 (hours per day) |
| Attendance | 0-100 (percent) |
| Previous_Score | 0-100 |
| Assignment_Score | 0-100 |
| Sleep_Hours | 0-14 (hours per day); Study_Hours + Sleep_Hours must not exceed 24 |
| Participation | Low, Medium, High |
| Internet_Access | Yes, No |
| Extracurricular | Yes, No |

Extra columns are ignored. Rows with missing or invalid values are **rejected with a reason**, never silently
corrected, and are downloadable as a separate CSV. Exported CSV cells that begin with `=`, `+`, `-` or `@` are
prefixed with an apostrophe to prevent spreadsheet formula injection. A template can be downloaded on the Batch page.

The CSV batch logic is implemented in `src/batch.py`, separately from the Streamlit interface, so input handling can be tested without a browser session. Missing required columns, empty files, and batches above 5,000 rows are rejected with a clear message. Each invalid row is excluded from inference and included in a separate validation-error download. Non-fatal unusual values are shown in the prediction output as warnings.

## Limitations
- Synthetic data: metrics show the method, not real performance.
- Correlation, not causation; what-if answers describe the model only.
- Ten features only; small test set (200 students).
- Fairness analysis is basic and cannot prove absence of bias.
- Shapley values explain the model, not the real world.
- History is a local CSV file (reset on some cloud hosts) and may contain whatever Student ID you type: use anonymous IDs.
- The PDF report uses a built-in Latin font: Arabic text typed as a Student ID is replaced by a note in the PDF (the web pages display Arabic normally).
- The Streamlit user interface could not be launched or browser-tested in the environment where these changes were made; see docs/TEST_REPORT.md.

## Future improvements
Real anonymised data with consent · calibration of probabilities · regression on actual scores · more fairness metrics and groups · user accounts and a proper database · a persistent database for History on the cloud · monitoring for data drift.

## Ethical considerations
Use predictions to offer support, never to punish or label students. Keep humans in the decision loop. Protect privacy: use anonymous IDs. Real deployment would need consent, bias audits on real data and clear communication of uncertainty.

## Disclaimer
This system is an educational AI prototype based on synthetic data. Predictions are intended for demonstration and early-warning purposes and should not be treated as an official academic judgment.


## Additional tools

### PDF student report
After making a prediction on the Prediction page, use **Download student report (PDF)** to export the inputs, model estimate, risk category, explanation summary, recommendations, and limitations.

### Synthetic-data limitation
The bundled dataset is synthetic/simulated. Metrics describe performance on held-out samples from this dataset and should not be interpreted as evidence of real-world performance. Predictions are educational estimates, not official academic outcomes. Feature contributions describe model behavior and do not establish causation.

### Rebuild the model in your environment
Serialized scikit-learn models can be incompatible across library versions. If the bundled model fails to load, activate the project virtual environment, install the dependencies, then run:

```bash
python train_model.py
python -m pytest -q
streamlit run app.py
```

Retraining regenerates the model artifact and evaluation reports using the installed library versions. Do not load untrusted `.pkl` or joblib files.


## Verification snapshot
The included model artifact and `reports/metrics.json` were generated together; the selected model is Logistic Regression
(test accuracy 0.820, ROC-AUC about 0.913 on the 200-row **synthetic** held-out set). These numbers do not describe real-world
performance. Exact, dated test results are in [docs/TEST_REPORT.md](docs/TEST_REPORT.md).
