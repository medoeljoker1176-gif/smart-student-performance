# Presentation Outline (18 slides) - Smart Student Performance & Early Risk Detection

Each slide: **On the slide** (what to show) and **Say** (short speaker notes you can memorise). Numbers come from the included run; re-check them in the app's *Model Performance* page before presenting.

---
### 1. Title
**On the slide:** project name, your name, Huawei programme, instructor, date.
**Say:** "My project is an AI system that predicts whether a student is likely to pass, shows how risky the situation is, explains why, and suggests what to improve."

### 2. Problem Statement
**On the slide:** "Students at risk are noticed too late."
**Say:** "Usually we discover a problem after the exam. I wanted an early-warning tool that works from simple data like study hours, attendance and scores."

### 3. Motivation
**On the slide:** why early warning; why explainability.
**Say:** "A plain Pass/Fail is not helpful. A student needs to know how risky it is, why, and what to change. That is why I added risk levels, explanations and recommendations."

### 4. Proposed Solution
**On the slide:** the 6 questions the system answers.
**Say:** "The system answers: what is the prediction, how likely is passing, how risky is it, which factors matter, what to improve, and how progress changes over time."

### 5. System Architecture
**On the slide:** pipeline diagram: Input → Validation → Preprocessing → Model → Probability → Risk → Explanation → Recommendations → Dashboard.
**Say:** "Input is validated first, so impossible values never reach the model. The model outputs a probability. From it I compute risk, an explanation and recommendations, and the dashboard shows everything."

### 6. Dataset
**On the slide:** 1,000 students, 10 features, about 70% Pass / 30% Fail, synthetic.
**Say:** "The dataset is synthetic and generated with a fixed seed, so results are reproducible. I say clearly that it shows the method, not real-world accuracy."

### 7. Data Preprocessing
**On the slide:** missing values → median / most frequent; scaling; one-hot; Pipeline; split first.
**Say:** "About 3% of some values are missing. I fill them inside a pipeline that learns only from training data. This prevents data leakage, which means information from the test set sneaking into training."

### 8. Exploratory Data Analysis
**On the slide:** box plots and correlation heat-map from the Analytics page.
**Say:** "Students who pass study more, attend more and have higher previous and assignment scores. Correlation shows a relationship, not a cause."

### 9. Machine Learning Models
**On the slide:** Logistic Regression, Decision Tree (baselines); Random Forest, Gradient Boosting (added).
**Say:** "I kept the two baselines and added two stronger models that can capture non-linear patterns. Each is tuned with a small grid search and 5-fold cross-validation repeated twice."

### 10. Model Evaluation
**On the slide:** comparison table + bar chart + confusion matrix.
**Say:** "I use accuracy, precision, recall, F1 and ROC-AUC. I chose the model by cross-validated ROC-AUC on training data, because risk levels depend on probabilities. Ties go to the simpler model. The test set is used only once, at the end."

### 11. Explainable AI
**On the slide:** contribution bar chart for one student + global importance.
**Say:** "For each student I show how many percentage points every factor added or removed from the probability. It is computed with Shapley values, the idea behind SHAP, exactly over all 1,024 factor combinations. Study hours, previous score and assignment score matter most."

### 12. Risk Detection
**On the slide:** threshold table + observed pass rate per tier.
**Say:** "Below 50% probability is High risk, 50 to 75% Moderate, above 75% Low. These thresholds are my design choice, not scientifically validated. As a check, the observed pass rate falls from about 93% in Low risk to about 20% in High risk."

### 13. Recommendation Engine
**On the slide:** Priority 1/2/3 cards with reason.
**Say:** "Rules find weak habits. Then I ask the model a what-if: what would the probability be if this habit reached the target? The biggest estimated gain becomes Priority 1. Every card explains why it was generated."

### 14. Application Demo (2 minutes)
**On the slide:** live app.
**Say:** "First a strong student: Low risk. Then the at-risk example: High risk, and the explanation shows attendance, study hours and scores pulling the probability down. Then I open Recommendations, Compare Students and History." *(Try one invalid input to show validation.)*

### 15. Results
**On the slide:** final model, test accuracy 0.820, ROC-AUC 0.913, F1 0.877, risk-tier chart, fairness table.
**Say:** "Logistic Regression was selected. It is accurate and easy to explain. It wins partly because the synthetic labels follow a mostly linear rule. The fairness check shows similar behaviour for male and female students, but that does not prove the model is unbiased."

### 16. Limitations
**On the slide:** synthetic data, correlation not causation, 10 features, small test set, basic fairness.
**Say:** "It is a prototype. It cannot be used to judge real students. Real data, more features and bias audits would be required."

### 17. Future Work
**On the slide:** real data, calibration, regression on scores, database + accounts, deployment, drift monitoring.
**Say:** "Next steps: use anonymised real data with consent, calibrate probabilities, store history in a database, deploy online, and monitor the model over time."

### 18. Conclusion
**On the slide:** one-line summary + thank you.
**Say:** "The project turns a simple Pass/Fail model into an explainable early-warning prototype: prediction, risk, reasons, and actions. It is honest about its limits. Thank you, I am happy to take questions."

---
**Tips:** keep the demo under 2 minutes; be ready to explain data leakage, why ROC-AUC/F1, what Shapley values mean, and why the data is synthetic. Discussion Q&A (English + Arabic): `docs/PROJECT_DISCUSSION_QA.md`.
