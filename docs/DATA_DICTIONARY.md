# Data Dictionary

The included file `data/student_performance.csv` is synthetic. It contains 1,000 generated student-like records. It is not collected from actual students.

| Column | Type | Expected values / range | Meaning and cautions |
|---|---|---|---|
| `Age` | Numeric integer | 15–40 accepted by input validation; synthetic generator uses 18–25 | Illustrative age value. Not expected to be a meaningful predictor across all real populations. |
| `Gender` | Category | `Male`, `Female` | Synthetic category used for a limited demonstration of group metrics. Do not infer identity or generalize fairness results. |
| `Study_Hours` | Numeric | 0–16 hours/day | Illustrative daily study time; high values may be unusual. |
| `Attendance` | Numeric | 0–100 | Attendance percentage. |
| `Previous_Score` | Numeric | 0–100 | Previous assessment score. |
| `Assignment_Score` | Numeric | 0–100 | Assignment score. |
| `Sleep_Hours` | Numeric | 0–14 hours/day | Illustrative sleep duration. |
| `Participation` | Category | `Low`, `Medium`, `High` | Synthetic class-participation category. |
| `Internet_Access` | Category | `Yes`, `No` | Synthetic access category. |
| `Extracurricular` | Category | `Yes`, `No` | Synthetic participation category. |
| `Result` | Target category | `Pass`, `Fail` | Pass when a hidden exam score (0–100, never saved) reaches 50. The score comes from previous/assignment scores, attendance, study hours (diminishing returns), sleep (7–8 h sweet spot), participation, ability and heteroscedastic noise; not an observed real academic result. |

## Optional real dataset: UCI Student Performance
`prepare_uci.py` converts `student-mat.csv` / `student-por.csv` into the same 10 columns. Mapping and caveats:

| App column | UCI source | Note |
|---|---|---|
| `Age`, `Gender` | `age`, `sex` | direct |
| `Study_Hours` | `studytime` (1-4) | band midpoints 1 / 3.5 / 7.5 / 12 h per week, divided by 7 (hours per day, so values are small) |
| `Attendance` | `absences` | `100 * (1 - absences / 90)`, clipped to 0-100; the 90 lessons are an assumption |
| `Previous_Score` | `G1` x 5 | period-1 grade |
| `Assignment_Score` | `G2` x 5 | period-2 grade used as the recent-work signal (proxy, not real assignments) |
| `Internet_Access`, `Extracurricular` | `internet`, `activities` | direct |
| `Sleep_Hours`, `Participation` | none | left empty; the model ignores them |
| `Result` | `G3` | Pass when `G3 >= 10` of 20; `G3` is never a feature |

Students present in both subject files are kept once so one person cannot be in both training and test sets. Rows with out-of-range values are dropped, never guessed.

## Missing values
About 3% of values in Study_Hours, Attendance, Sleep_Hours, Assignment_Score and Participation are set to missing by `generate_dataset.py` (slightly more often for students without internet, never depending on `Result`, at most 2 gaps per row). Imputation is performed inside the model pipeline, fitted only on training folds/splits.

## Cleaning policy
Duplicate rows and rows without a target are removed. Rows with numeric values outside configured hard input ranges, or with study and sleep hours totaling more than 24 hours, are excluded rather than silently clipped. Missing feature values are preserved for the pipeline imputer. Unexpected categories and non-numeric/non-finite numeric values are validation errors.

## CSV upload schema
Batch prediction CSVs must include all ten input feature columns above. `Result` is not required for inference. Additional columns are ignored. Invalid rows are reported separately and are not sent to the model.
