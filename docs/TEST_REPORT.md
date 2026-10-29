# Test report

Prepared on 2026-10-21. This report separates **what was executed and observed** from **what was not**.

## 1. Environment limits (read first)
The environment used for this work (Linux, Python 3.12.3) had **no network access**. `pip install streamlit plotly pytest`
failed with "No matching distribution found". Consequently:

| Item | Status |
|---|---|
| `python -m streamlit run app.py` (server start) | **NOT run** |
| Browser / UI testing of any page | **NOT performed** |
| `python -m pytest -q` | **NOT run** (pytest not installable) |
| Streamlit- and Plotly-specific API behaviour | **Unverified** |

No screenshots exist and none are claimed. Please run the checklist in section 5 on your Windows machine.

## 2. Automated results that WERE obtained
| Check | Command / method | Result |
|---|---|---|
| Compilation | `python -m compileall -q app.py src views tests train_model.py generate_dataset.py` | OK, exit code 0 |
| Original core tests (before my changes) | `python -m tests.test_core` | 25 / 25 passed |
| Core tests (after changes) | `python -m tests.test_core` | 25 / 25 passed |
| Full suite, 8 files | A small **pytest-compatible shim** written for the sandbox (supports fixture, raises, parametrize, tmp_path, monkeypatch, importorskip) | **75 passed, 0 failed, 1 skipped** (the skipped file is `test_ui_apptest.py`, which needs Streamlit) |

The shim is **not pytest**. Treat the 75 figure as strong evidence for the logic, not as a substitute for running
`python -m pytest -q` yourself; expect the same tests to be collected (counts may differ slightly because pytest
reports parametrised cases individually).

Non-vacuity check: the new `tests/test_validation_edge.py` was run against the **original** code and produced
3 failures (a `pd.NA` crash, and a boolean accepted as a number), then passed on the fixed code.

Dependency versions present when these ran: see `requirements-lock-tested.txt` (pandas 3.0.2, numpy 2.4.4,
scikit-learn 1.8.0, reportlab 4.4.10). The bundled model was saved with scikit-learn 1.8.0 and loaded with no version warning.

## 3. Stub-based smoke execution (NOT a UI test)
`streamlit` and `plotly` were replaced by lightweight fakes and every page's `render()` plus `app.py` were **executed** against
the real model, data and validation code. This catches NameErrors, KeyErrors and data-handling crashes; it cannot show layout,
widgets, charts or browser behaviour. 30 scenarios ran with 0 exceptions: all pages with fresh state; Predict with valid input;
invalid Attendance (140) and Study+Sleep > 24 (nothing stored, no prediction); dashboard and history with and without saved
history; batch with a valid file, invalid rows, missing column, header-only, empty, binary, 6 MB, formula text and Excel-BOM files;
missing model file. (This harness is not shipped; it lived outside the project.)

## 4. Defects found by running the code, and fixes
| # | Defect (reproduced) | Fix |
|---|---|---|
| 1 | PDF report **always** printed "Detailed explanation was unavailable": it parsed the `Sentence` column as a number. The old test only checked the `%PDF` header. | `reporting.explanation_rows()` uses the real `Label/Value/Contribution` columns; test now asserts real rows. |
| 2 | `pd.NA` in an uploaded cell raised `TypeError`, aborting the whole batch. | `validation._is_missing()`; per-row rejection; test added. |
| 3 | 5,000-row batch took **32.5 s** (one model call per row). | One vectorised call: **0.11 s**, identical probabilities (tested). |
| 4 | Duplicate column names produced a cryptic pandas error. | Clear `BatchInputError`. |
| 5 | Batch results vanished after clicking a download button (the rerun reset the Analyze button). | Results kept in `st.session_state`. *(Streamlit behaviour; fix not browser-verified.)* |
| 6 | Probabilities displayed as "0.0%" / "100.0%" overstated certainty. | `format_probability` shows `<1%` / `>99%`; uncertainty note on the Prediction page quotes the real test accuracy from `metrics.json`. |
| 7 | A hand-edited history file with non-numeric `P_Pass` would crash History. | Unusable rows are ignored (tested). |
| 8 | `requirements.txt` unpinned: a fresh install could pull a scikit-learn that cannot load the bundled pickle; scikit-learn 1.8 also needs Python 3.11+. | Pinned `scikit-learn>=1.8,<1.9`, documented Python 3.11+. |
| 9 | PDF would print boxes for Arabic Student IDs (Latin-only font). | Replaced by a note in the PDF; documented limitation. |
| 10 | README claimed "29 tests passed" (actual count differed). | Replaced with a pointer to this report. |

UI work done (unverified visually): blue/white theme in `.streamlit/config.toml` and CSS, Material icons in navigation,
emoji reduced, tooltips on inputs, spinner and error handling on prediction, dashboard KPIs from saved history with navigation links,
batch page with format guide, preview, schema notes, summary charts and safe downloads, history filters, metric glossary and
synthetic-data warning on Model Performance, correlation / attribution / causation explainer. `use_container_width` removed
because recent Streamlit versions deprecate it.

## 5. Manual UI checklist for you (none of these have been executed)
Run `python -m streamlit run app.py`, then tick:
- [ ] App starts; sidebar shows "Model ready: Logistic Regression".
- [ ] Every sidebar page opens with no red error box; the Dashboard links navigate.
- [ ] Prediction: Load *At-risk student* -> Predict -> verdict, gauges, tabs, PDF download opens and lists contributions.
- [ ] Prediction: set Attendance 100 and sleep 14 / study 16 -> red validation message, no result.
- [ ] Batch: download template -> upload it -> Analyse -> table, charts; click both download buttons and confirm results stay visible.
- [ ] Batch: upload a file missing a column, an empty file, and one with `Attendance = 150` -> clear messages.
- [ ] History: empty state before the first prediction; filters and download after it.
- [ ] Rename `models/student_performance_model.pkl` temporarily -> friendly message with the retrain command; restore it.
- [ ] Arabic text typed in Student ID displays correctly in the page (PDF will show a note instead).
- [ ] `python -m pytest -q` (all tests, including `test_ui_apptest.py`).

## 6. Unverified assumptions and remaining issues
* Streamlit API use (`st.Page` Material icons, `st.page_link` with page objects, `Box`-style containers, `st.metric` CSS selectors) follows the
  documentation for `streamlit>=1.42` but was never executed against real Streamlit. CSS selectors such as `[data-testid="stMetric"]` can change between releases.
* Plotly 6 with pandas 3 was not exercised; chart code paths were only run against a fake Plotly (data preparation ran, rendering did not).
* `tests/test_ui_apptest.py` was written without being run; its page-switching step is best effort. Adjust or drop it if AppTest differs in your version.
* The model's metrics are synthetic-data results only. Probabilities are not calibrated beyond what the Brier score on that data indicates.
* History stores the typed Student ID in plain text in a local CSV (unchanged architecture).
* Existing pages not deeply reworked: Analytics, Compare, Recommendations, About (only text/emoji/API cleanups).

## 7. Files changed
Changed: README.md, app.py, requirements.txt, src/{batch,history,prediction,reporting,ui,validation}.py,
views/{about,analytics,batch_analysis,compare,dashboard,explanation,history,model_performance,prediction,recommendations}.py.
Added: .streamlit/config.toml, pytest.ini, requirements-lock-tested.txt, docs/STREAMLIT_SETUP.md, docs/TEST_REPORT.md,
tests/{test_batch_edge,test_validation_edge,test_exports_history,test_prediction_reporting,test_ui_apptest}.py.

## 6. Update: improved synthetic data generator
`generate_dataset.py` was rewritten (same 10 features + `Result`, same CSV layout), then `train_model.py` was re-run so the
model, `reports/metrics.json` and plots match the new data. Results in the sandbox with real pytest
(`python -m pytest -q`): **93 passed, 0 failed, 1 skipped** (`test_ui_apptest.py` is skipped because Streamlit is not
installed there). New file `tests/test_generate_dataset.py` covers seed reproducibility, schema/limits, class balance,
missingness, feature relationships, leakage and invalid sizes. One old assertion in `test_prediction_reporting.py`
hard-coded which feature is the top driver; it now checks the driver is a real labelled feature that lowers P(pass).

## 7. Update: real-dataset support and login (this session)
Real pytest, sandbox: **122 passed, 0 failed, 2 skipped** (the two Streamlit AppTest files, because Streamlit cannot be installed there).
New tests: `tests/test_auth.py` (hashing, lockout, roles, expiry), `tests/test_uci_dataset.py` (adapter, de-duplication, bad input, training and prediction on the second dataset). **Not verified:** the real UCI files were not available (the domain is blocked in the sandbox), so the adapter was tested on a made-up file with the real column layout, and the Streamlit login/selector screens have never been run. Run `python -m pytest -q` (including `tests/test_auth_apptest.py`) and `streamlit run app.py` on your machine, then try both roles.


## Note
The real UCI files (`student-mat.csv`, `student-por.csv`) are now bundled in `data/raw/`, so the adapter limitation above no longer applies. The Streamlit AppTest files have still not been executed.
