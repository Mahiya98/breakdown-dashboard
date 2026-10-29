"""
build_preprocessing_pdf.py
===========================
Builds data_preprocessing_report.pdf from preprocessing_output/pipeline_results.json,
mirroring the structure of build_audit_pdf.py and build_target_pdf.py.
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable

OUT_DIR = Path("preprocessing_output")
RESULTS_FILE = OUT_DIR / "pipeline_results.json"
PDF_FILE = "data_preprocessing_report.pdf"

with open(RESULTS_FILE, "r") as f:
    results = json.load(f)

doc = SimpleDocTemplate(
    PDF_FILE,
    pagesize=letter,
    rightMargin=72, leftMargin=72,
    topMargin=72, bottomMargin=18
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Justify", alignment=4))
styles.add(ParagraphStyle(name="Indent", leftIndent=25))

story = []

story.append(Paragraph("Data Preprocessing Report", styles["Title"]))
story.append(Spacer(1, 12))

r = results
story.append(Paragraph(
    f"Processed dataset: {r['dataset_shape'][0]:,} rows x {r['dataset_shape'][1]:,} columns",
    styles["Heading2"]))
story.append(Spacer(1, 12))

story.append(Paragraph(
    f"VRM data from {r['date_range'][0]} to {r['date_range'][1]} ({r['n_calendar_days']} days).",
    styles["Normal"]))

story.append(Spacer(1, 12))
story.append(Paragraph("Operating status by machine:", styles["Heading3"]))
for machine, counts in r['per_machine_operating_status'].items():
    line = f"- {machine}:  " + ", ".join(f"{k}: {v}" for k,v in counts.items())
    story.append(Paragraph(line, styles["Indent"]))

story.append(Spacer(1, 12))
story.append(Paragraph("Label confidence by machine:", styles["Heading3"]))
for machine, counts in r['per_machine_label_confidence'].items():
    line = f"- {machine}:  " + ", ".join(f"{k}: {v}" for k,v in counts.items())
    story.append(Paragraph(line, styles["Indent"]))

story.append(Spacer(1, 12))
story.append(Paragraph(
    f"Features: {r['n_feature_columns_lagged']} lagged columns, {r['n_sameday_audit_only_columns']} same-day audit-only.",
    styles["Normal"]))

story.append(Spacer(1, 12))
story.append(Paragraph("Targets:", styles["Heading3"]))
targets_table = [['Target', 'Description']] + [
    [col, "Binary: any eligible breakdown" if col == "A_any_eligible_breakdown" else
          f"Binary: '{col[2:-9]}' family occurred" if col.startswith("B_") else
          f"Count: events in '{col[2:-12]}' family" if col.startswith("C_") else
          "Total downtime (min) across eligible events"
     ] for col in r['target_columns']
]
tbl = Table(targets_table, colWidths=[80, 300])
tbl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.grey),
    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,0), 10),
    ('BOTTOMPADDING', (0,0), (-1,0), 8),
    ('BACKGROUND', (0,1), (-1,-1), colors.beige),
    ('TEXTCOLOR', (0,1), (-1,-1), colors.black),
    ('ALIGN', (0,1), (-1,-1), 'LEFT'),
    ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
    ('FONTSIZE', (0,1), (-1,-1), 8),
    ('TOPPADDING', (0,1), (-1,-1), 3),
    ('BOTTOMPADDING', (0,1), (-1,-1), 3),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
]))
story.append(tbl)
story.append(Spacer(1, 12))

story.append(Paragraph(
    f"{r['n_trainable_rows_(positive_or_confident_negative)']:,} trainable rows "
    f"({r['trainable_positive_rate_pct']}% positive rate) after excluding "
    f"{r['n_excluded_unknown_rows']:,} unknowns.",
    styles["Normal"]))

story.append(Spacer(1, 12))
story.append(Paragraph(
    "No-leakage guarantee: all features are lagged, same-day columns prefixed "
    "SAMEDAY_AUDIT_ONLY__ (never trained on), operating_status and label_confidence "
    "are metadata only (never features). See feature_data_dictionary.md for details.",
    styles["Normal"]
))

doc.build(story)
print(f"Wrote {PDF_FILE}")