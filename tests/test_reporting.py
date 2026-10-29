from types import SimpleNamespace

import pandas as pd

from src.reporting import build_student_report


def test_pdf_report_generates_valid_pdf_bytes():
    result = SimpleNamespace(
        label="Pass",
        p_pass=0.82,
        risk={"level": "Low Risk"},
        score=76.2,
        score_table=pd.DataFrame(),
    )
    explanation = {"table": pd.DataFrame([
        {"Feature": "Study_Hours", "Value": "5 h/day", "Contribution": 12.4, "Sentence": "Helpful"}
    ])}
    rec = SimpleNamespace(title="Improve study consistency", action="Plan short sessions.", reason="Below target.", gain=0.04)
    analysis = {
        "student": {"Age": 20, "Study_Hours": 5.0},
        "result": result,
        "explanation": explanation,
        "recs": [rec],
        "model_name": "Logistic Regression",
    }
    payload = build_student_report(analysis, "demo-student")
    assert payload.startswith(b"%PDF")
    assert len(payload) > 1000
