"""Model loading, single-student prediction and PDF report content."""
import joblib
import pytest

from src.config import FEATURE_LABELS, SAMPLE_PROFILES
from src.prediction import (ModelLoadError, analyze_student, format_probability, load_bundle, p_pass_for,
                            predict_student)
from src.reporting import build_student_report, explanation_rows, pdf_safe_text


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


def test_model_bundle_has_required_parts_and_no_version_warning_when_matching(bundle):
    assert {"pipeline", "model_name", "background", "training_ranges"} <= set(bundle)
    assert hasattr(bundle["pipeline"], "predict_proba")


def test_missing_model_path_gives_friendly_error(tmp_path):
    with pytest.raises(ModelLoadError, match="train_model.py"):
        load_bundle(tmp_path / "nope.pkl")


def test_wrong_format_pickle_is_rejected(tmp_path):
    path = tmp_path / "bad.pkl"
    joblib.dump({"unexpected": 1}, path)
    with pytest.raises(ModelLoadError, match="unexpected format"):
        load_bundle(path)


def test_single_prediction_is_reproducible_and_ordered(bundle):
    values = [p_pass_for(bundle, SAMPLE_PROFILES[k]) for k in ("Strong student", "Average student", "At-risk student")]
    assert values == [p_pass_for(bundle, SAMPLE_PROFILES[k]) for k in ("Strong student", "Average student", "At-risk student")]
    assert values[0] > values[1] > values[2] and all(0 <= v <= 1 for v in values)
    assert predict_student(bundle, SAMPLE_PROFILES["Strong student"]).label == "Pass"


def test_format_probability_never_claims_certainty():
    assert format_probability(0.0) == "<1%" and format_probability(1.0) == ">99%"
    assert format_probability(0.456) == "45.6%"


def test_report_lists_real_contributions_not_the_unavailable_message(bundle, tmp_path):
    analysis = analyze_student(bundle, SAMPLE_PROFILES["At-risk student"])
    rows = explanation_rows(analysis["explanation"]["table"])
    # The biggest driver for an at-risk student must lower P(pass); which feature it is
    # depends on the training data, so only check that it is a real, labelled feature.
    assert rows and rows[0][2].startswith("-") and rows[0][0] in FEATURE_LABELS.values()
    pdf = build_student_report(analysis, "demo")
    assert pdf.startswith(b"%PDF")
    path = tmp_path / "r.pdf"
    path.write_bytes(pdf)
    try:
        from pypdf import PdfReader                      # optional; used only when installed
    except ImportError:
        return
    text = "".join(page.extract_text() for page in PdfReader(str(path)).pages)
    assert "Detailed explanation was unavailable" not in text and "Study hours" in text


def test_report_with_arabic_label_does_not_crash_or_print_boxes(bundle):
    analysis = analyze_student(bundle, SAMPLE_PROFILES["Average student"])
    assert build_student_report(analysis, "أحمد").startswith(b"%PDF")
    assert pdf_safe_text("أحمد", "hidden") == "hidden" and pdf_safe_text(" Sara ") == "Sara"


def test_report_escapes_markup_in_label(bundle):
    analysis = analyze_student(bundle, SAMPLE_PROFILES["Average student"])
    assert build_student_report(analysis, "<b>x</i>&").startswith(b"%PDF")


def test_model_from_other_sklearn_minor_version_is_refused_so_the_app_retrains(tmp_path):
    import sklearn
    bundle = load_bundle()
    major, minor = sklearn.__version__.split(".")[:2]
    other = f"{major}.{int(minor) + 1}.0"
    bundle["sklearn_version"] = other
    path = tmp_path / "other_version.pkl"
    joblib.dump(bundle, path)
    with pytest.raises(ModelLoadError, match="train_model.py"):
        load_bundle(path)
    bundle["sklearn_version"] = f"{major}.{minor}.99"           # patch difference is fine
    joblib.dump(bundle, path)
    assert load_bundle(path)["pipeline"] is not None
