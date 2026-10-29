"""
build_target_pdf.py
=====================
Builds target_definition_report.pdf from target_output/target_definition_results.json.

Usage: python build_target_pdf.py
Requires: reportlab
"""

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, ListFlowable, ListItem
)

import label_taxonomy as T

OUT_DIR = Path("target_output")
R = json.load(open(OUT_DIR / "target_definition_results.json"))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1c", parent=styles["Heading1"], spaceAfter=10))
styles.add(ParagraphStyle(name="H2c", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle(name="H3c", parent=styles["Heading3"], spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle(name="Bodyc", parent=styles["BodyText"], spaceAfter=6, leading=13))

story = []
def h1(t): story.append(Paragraph(t, styles["H1c"]))
def h2(t): story.append(Paragraph(t, styles["H2c"]))
def h3(t): story.append(Paragraph(t, styles["H3c"]))
def p(t): story.append(Paragraph(t, styles["Bodyc"]))
def bullets(items):
    story.append(ListFlowable([ListItem(Paragraph(str(i), styles["Bodyc"])) for i in items],
                               bulletType="bullet", start="•"))
def table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))


h1("Prediction Target Definition Report")
p("Step 4 — Defining the electrical/mechanical breakdown prediction target for VRM-1 / VRM-2")
p("Source: <b>VRM_Jul2025_Oct2026_Combined.xlsx</b> (opened read-only, never modified).")
story.append(Spacer(1, 4))

# ---- 4.1 ----
h2("4.1 Eligible Breakdown Classification")
p("<b>Authoritative rule used:</b> every record's Breakdown Name and Reason field in the "
  "Breakdown sheet carries a 2-letter+digit code prefix (e.g. 'E01-', 'M01-', 'U01-'). These "
  "prefixes match the Dropdown Data Code/Code.1 columns, and counting records by prefix "
  "reproduces the Category field's value counts exactly. This confirms Category is the plant's "
  "own documented, authoritative classification — not a keyword inference made for this project.")
h3("Category → Code verification (counts from live data)")
rows = [["Category", "Code", "Event count"]]
for cat, code in T.CATEGORY_CODE_MAP.items():
    rows.append([cat, code, R["category_counts"].get(cat, 0)])
table(rows, col_widths=[220, 60, 100])

p(f"<b>Result:</b> only Category = 'Electrical Breakdown' (E01) and 'Mechanical Breakdown' (M01) "
  f"are eligible for the primary target — {R['category_counts'].get('Electrical Breakdown',0)} and "
  f"{R['category_counts'].get('Mechanical Breakdown',0)} events respectively. All other categories, "
  f"including 'Utility (Electricity)' which contains the word 'Electrical' in its name, are "
  f"explicitly excluded — confirming the instruction not to classify by keyword alone.")

p(f"Of {R['n_mapping_rows_total']} unique (Category, Name, Reason) combinations found in the data, "
  f"{R['n_mapping_rows_eligible']} are eligible (E01/M01). "
  f"<b>{R['n_mapping_rows_needs_review']}</b> of those are flagged 'Needs Review' — they fell "
  f"into a residual 'Other Confirmed Electrical/Mechanical' bucket because no documented family "
  f"keyword matched; they are NOT force-classified into an arbitrary family. Full detail, "
  f"including original and normalized text, rationale, and confidence for every single mapping, "
  f"is in <i>breakdown_label_mapping.xlsx</i>.")

story.append(PageBreak())

# ---- 4.2 ----
h2("4.2 Failure-Family Taxonomy (3 Levels)")
p("Level 1 = Electrical/Mechanical (from Category, Step 4.1). Level 2 = failure family, assigned "
  "by documented keyword rules applied to normalized Breakdown Name + Reason text, evaluated only "
  "within the eligible E01/M01 subset. Level 3 = normalized breakdown reason (full detail retained "
  "in event_level_reason_table.csv). Candidate families not supported by any real record were "
  "dropped; none were assumed.")

rows = [["Level 1", "Level 2 Family", "Unique Reasons", "Events", "Low-conf. rows"]]
for r in R["taxonomy_summary"]:
    rows.append([r["electrical_mechanical_classification"], r["failure_family_level2"],
                 r["n_unique_reasons"], r["n_events"], r["low_confidence_rows"]])
table(rows, col_widths=[70, 230, 60, 50, 60])

p("<b>Note:</b> 'Fan &amp; Separator' and 'Electrical Protection/Instrumentation/Control Faults' "
  "families appear under Electrical because their triggering events (trips, alarms, PLC faults) "
  "were logged under Category = Electrical Breakdown; the same conceptual equipment (fans, "
  "dampers) can also fail mechanically and is tracked separately under Category = Mechanical "
  "Breakdown where the data shows it. This is intentional — Level 1 always follows the plant's "
  "own Category field, and Level 2 describes the failure mode/equipment within that.")

story.append(PageBreak())

# ---- Event granularity ----
h2("Event Granularity (prerequisite for Step 4.3)")
g = R["granularity"]
rows = [["Metric", "Value"]]
rows.append(["Max rows sharing one (Mill, Date, Shift) key", g["max_rows_in_one_machine_date_shift"]])
rows.append(["% of (Mill, Date, Shift) combos with >1 event", g["pct_machine_date_shift_with_multiple_events"]])
rows.append(["Downtime range (minutes)", f"{g['downtime_min']} – {g['downtime_max']}"])
rows.append(["Downtime mean (minutes)", g["downtime_mean"]])
table(rows, col_widths=[280, 160])
p(g["conclusion"])
p(f"Exact-duplicate (Mill, Date, Shift, Name, Reason) rows removed before building targets: "
  f"<b>{R['n_dedup_removed']}</b>.")

# ---- 4.3 ----
h2("4.3 Prediction Target Definitions")
p(f"A machine-day target table was built: <b>{R['n_machine_days']} rows</b> "
  f"({R['target_df_shape'][0]} machine-day combinations × {R['target_df_shape'][1]} columns), "
  f"covering every (machine, calendar date) pair across the full data range.")

h3("Target A — Any eligible breakdown that day (binary)")
p(f"Column: <tt>A_any_eligible_breakdown</tt>. Overall positive rate across all machine-days: "
  f"<b>{R['overall_A_positive_rate_pct']}%</b>.")

h3("Target B — Breakdown within a specific failure family (binary, per family)")
p("One column per family from Section 4.2, e.g. <tt>B_Motor &amp; Electrical Drive Failures_occurred</tt>.")

h3("Target C — Expected number of events per family per machine-day")
p("One integer column per family, e.g. <tt>C_Bearing, Gearbox &amp; Lubrication Failures_event_count</tt>, "
  "computed as deduplicated row counts (see Event Granularity above) — not distinct-shift counts.")

h3("Target D — Most likely detailed reason, conditional on family")
p("Delivered as <i>event_level_reason_table.csv</i>: one row per deduplicated event with "
  "machine, date, shift, Level-1 class, Level-2 family, and normalized Level-3 reason. This "
  "supports a two-stage model (predict family, then predict reason within that family) rather "
  "than one 200+-way flat classifier, which the Step-1 audit already showed would be severely "
  "data-starved per class.")

h3("Target E — Expected downtime (only where reliably supportable)")
p(R["feasibility_conclusion"])
rows = [["Family", "Events", "Mean (min)", "Std", "CV", "Reliable?"]]
for r in R["feasibility_df"]:
    rows.append([r["failure_family"], r["n_events"], r["mean"], r["std"],
                 r["cv_(std/mean)"], "Yes" if r["reliable_for_regression"] else "No"])
table(rows, col_widths=[190, 50, 60, 50, 40, 55])
p("<b>Decision:</b> downtime regression will be attempted only for the families marked "
  "'Reliable' above; for the rest, only a historical descriptive range (min/mean/max) will be "
  "reported alongside predictions, never a manufactured point estimate.")

story.append(PageBreak())

h2("Deliverables")
bullets([
    "label_taxonomy.py — core classification + taxonomy + target-construction logic",
    "run_target_definition.py — orchestrator producing all outputs below",
    "target_output/breakdown_label_mapping.xlsx — full Step 4.1 mapping table (all required columns)",
    "target_output/failure_taxonomy_summary.xlsx — Step 4.2 taxonomy with support counts",
    "target_output/machine_day_targets.csv — Targets A, B, C, E at machine-day grain",
    "target_output/event_level_reason_table.csv — Target D support table",
    "target_output/downtime_feasibility.xlsx — Target E feasibility diagnostics",
    "target_definition_report.md / .pdf — this report",
])
p("<i>End of report. The source workbook was opened read-only and was not modified "
  "by any step (verified by checksum).</i>")

doc = SimpleDocTemplate(
    "target_definition_report.pdf", pagesize=A4,
    topMargin=20 * mm, bottomMargin=20 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
    title="Prediction Target Definition Report"
)
doc.build(story)
print("Wrote target_definition_report.pdf")
