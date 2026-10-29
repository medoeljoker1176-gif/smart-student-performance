"""PDF report generation for a single validated student prediction."""
from io import BytesIO
from xml.sax.saxutils import escape
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from src.config import FEATURE_LABELS
from src.prediction import format_probability
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def pdf_safe_text(value, fallback: str = "") -> str:
    """Text the built-in PDF font (Helvetica, Latin-1) can draw.

    Arabic and other non-Latin characters would be printed as black boxes, so a
    value that cannot be encoded is replaced by `fallback` instead of garbage.
    """
    text = str(value if value is not None else "").strip()
    try:
        text.encode("latin-1")
    except UnicodeEncodeError:
        return fallback
    return text


def explanation_rows(factors, limit: int = 6) -> list:
    """Rows [Feature, Value, Contribution (pts)] for the PDF, strongest factors first.

    Reads the real columns produced by explainability.explain_student() (Label, Value,
    Contribution) and also tolerates a table that only has Feature + Contribution.
    """
    if factors is None or len(factors) == 0 or "Contribution" not in factors.columns:
        return []
    name_col = "Label" if "Label" in factors.columns else "Feature"
    rows = []
    for _, row in factors.reindex(factors["Contribution"].abs().sort_values(ascending=False).index).head(limit).iterrows():
        rows.append([str(row[name_col]), str(row["Value"]) if "Value" in factors.columns else "",
                     f"{float(row['Contribution']):+.1f}"])
    return rows


def build_student_report(analysis: dict, student_id: str = "") -> bytes:
    """Build a PDF summary without presenting predictions as confirmed outcomes."""
    result = analysis["result"]
    student = analysis["student"]
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=16*mm, bottomMargin=16*mm, title="Smart Student Performance Report")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#555555")))
    story = [
        Paragraph("Smart Student Performance", styles["Title"]),
        Paragraph("Student Prediction Report", styles["Heading2"]),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["BodyText"]),
        Paragraph(f"Student label: {escape(pdf_safe_text(student_id, 'Not shown (non-Latin text cannot be drawn in this PDF)') or 'Anonymous')}", styles["BodyText"]),
        Spacer(1, 8),
        Paragraph("Prediction summary", styles["Heading2"]),
    ]
    summary = [
        ["Predicted class", result.label],
        ["Probability of passing", format_probability(result.p_pass)],
        ["Risk category", result.risk["level"]],
        ["Performance score (application indicator)", f"{result.score:.1f}/100"],
        ["Model", str(analysis.get("model_name", "Saved model"))],
    ]
    table = Table(summary, colWidths=[70*mm, 85*mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#EEF3F8")),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#18212B")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#C9D3DF")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story += [table, Spacer(1, 10), Paragraph("Input profile", styles["Heading2"])]
    input_rows = [["Feature", "Value"]] + [[FEATURE_LABELS.get(k, str(k)), str(v)] for k, v in student.items()]
    inputs_table = Table(input_rows, colWidths=[70*mm, 85*mm], repeatRows=1, hAlign="LEFT")
    inputs_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DCE7F2")),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#C9D3DF")),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story += [inputs_table, Spacer(1, 10), Paragraph("Key contributing factors", styles["Heading2"])]
    explanation = analysis.get("explanation", {})
    rows = explanation_rows(explanation.get("table"))
    if rows:
        story.append(Paragraph("Contributions are in percentage points of P(pass) compared with an average reference "
                               "student. They are model explanation outputs, not proof of causation.", styles["SmallNote"]))
        ft = Table([["Factor", "Student value", "Contribution (pts)"]] + rows,
                   colWidths=[60*mm, 50*mm, 45*mm], repeatRows=1, hAlign="LEFT")
        ft.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE7F2")),
                                ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#C9D3DF")),
                                ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        story.append(ft)
    else:
        story.append(Paragraph("Detailed explanation was unavailable.", styles["BodyText"]))
    story += [Spacer(1, 10), Paragraph("Recommendations", styles["Heading2"])]
    recs = analysis.get("recs", [])
    if recs:
        for rec in recs[:6]:
            if isinstance(rec, dict):
                title = rec.get("title") or rec.get("recommendation") or rec.get("text") or str(rec)
                detail = rec.get("reason", "")
            else:
                title = getattr(rec, "title", str(rec))
                detail = getattr(rec, "action", "") or getattr(rec, "reason", "")
                gain = getattr(rec, "gain", None)
                if gain is not None:
                    detail = (str(detail) + f" Estimated model-based what-if change: {gain * 100:+.1f} percentage points.").strip()
            story.append(Paragraph("&bull; " + escape(str(title)), styles["BodyText"]))
            if detail:
                story.append(Paragraph(escape(str(detail)), styles["SmallNote"]))
    else:
        story.append(Paragraph("No recommendations were available.", styles["BodyText"]))
    story += [Spacer(1, 12), Paragraph(
        "Important limitation: This is an educational AI prototype trained on synthetic data. "
        "Predictions are estimates and must not be treated as official academic results or used alone "
        "for consequential decisions. Model explanations describe model behavior and do not establish causation.",
        styles["SmallNote"])]
    doc.build(story)
    return buffer.getvalue()
