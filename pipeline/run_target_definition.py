"""
run_target_definition.py
==========================
Orchestrates label_taxonomy.py and writes every Step-4 deliverable:

    target_output/breakdown_label_mapping.xlsx      (Step 4.1 mapping table)
    target_output/failure_taxonomy_summary.xlsx      (Step 4.2 taxonomy)
    target_output/machine_day_targets.csv            (Step 4.3 A/B/C/E targets)
    target_output/event_level_reason_table.csv       (Step 4.3 D support table)
    target_output/downtime_feasibility.xlsx          (Step 4.3 E feasibility)
    target_output/target_definition_results.json     (machine-readable, PDF source)
    target_definition_report.md

Source workbook is opened read-only; never modified.

Usage: python run_target_definition.py
"""

import json
from pathlib import Path

import pandas as pd

import label_taxonomy as T

OUT_DIR = Path("target_output")
OUT_DIR.mkdir(exist_ok=True)


def to_jsonable(obj):
    import numpy as np
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main():
    print("Running label taxonomy pipeline on source workbook (read-only)...")
    out = T.run_all()

    mapping_df = out["mapping_df"]
    tax_summary = out["taxonomy_summary"]
    granularity = out["granularity"]
    target_df = out["target_df"]
    families = out["families"]
    n_dedup = out["n_dedup_removed"]
    reason_table = out["reason_table"]
    feas_df = out["feasibility_df"]
    feas_conclusion = out["feasibility_conclusion"]

    # ---- 4.1 mapping table ----
    with pd.ExcelWriter(OUT_DIR / "breakdown_label_mapping.xlsx", engine="openpyxl") as w:
        mapping_df.to_excel(w, sheet_name="Full Mapping Table", index=False)
        mapping_df[mapping_df["eligible_for_target"]].to_excel(
            w, sheet_name="Eligible (E+M) Only", index=False)
        mapping_df[mapping_df["review_status"] == "Needs Review"].to_excel(
            w, sheet_name="Needs Review", index=False)
        mapping_df[~mapping_df["eligible_for_target"]].to_excel(
            w, sheet_name="Excluded Categories", index=False)
    print("Wrote breakdown_label_mapping.xlsx")

    # ---- 4.2 taxonomy summary ----
    with pd.ExcelWriter(OUT_DIR / "failure_taxonomy_summary.xlsx", engine="openpyxl") as w:
        tax_summary.to_excel(w, sheet_name="Family Summary", index=False)
        cat_check = pd.DataFrame([{
            "category": k, "documented_code_prefix": v,
        } for k, v in T.CATEGORY_CODE_MAP.items()])
        cat_check.to_excel(w, sheet_name="Category-Code Verification", index=False)
    print("Wrote failure_taxonomy_summary.xlsx")

    # ---- 4.3 targets ----
    target_df.to_csv(OUT_DIR / "machine_day_targets.csv", index=False)
    reason_table.to_csv(OUT_DIR / "event_level_reason_table.csv", index=False)
    with pd.ExcelWriter(OUT_DIR / "downtime_feasibility.xlsx", engine="openpyxl") as w:
        feas_df.to_excel(w, sheet_name="Downtime Feasibility", index=False)
    print("Wrote machine_day_targets.csv, event_level_reason_table.csv, downtime_feasibility.xlsx")

    # ---- JSON (PDF source of truth) ----
    results = {
        "families": families,
        "n_dedup_removed": int(n_dedup),
        "granularity": granularity,
        "taxonomy_summary": to_jsonable(tax_summary),
        "n_mapping_rows_total": int(len(mapping_df)),
        "n_mapping_rows_eligible": int(mapping_df["eligible_for_target"].sum()),
        "n_mapping_rows_needs_review": int((mapping_df["review_status"] == "Needs Review").sum()),
        "category_counts": mapping_df.groupby("original_category")["event_count"].sum().to_dict(),
        "feasibility_df": to_jsonable(feas_df),
        "feasibility_conclusion": feas_conclusion,
        "target_df_shape": list(target_df.shape),
        "target_df_columns": list(target_df.columns),
        "n_machine_days": int(len(target_df)),
        "overall_A_positive_rate_pct": round(float(target_df["A_any_eligible_breakdown"].mean() * 100), 2),
    }
    with open(OUT_DIR / "target_definition_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Wrote target_definition_results.json")

    write_markdown_report(results, mapping_df, tax_summary, target_df, feas_df)
    print("Wrote target_definition_report.md")

    print(f"\nDone. Outputs in {OUT_DIR.resolve()}")
    return out


def write_markdown_report(R, mapping_df, tax_summary, target_df, feas_df):
    lines = []
    lines.append("# Prediction Target Definition Report\n")
    lines.append("## Step 4.1 — Eligible Breakdown Classification\n")
    lines.append("Authoritative rule: a Breakdown-sheet record is eligible for the primary "
                  "electrical/mechanical failure target **only if** `Category` is exactly "
                  "'Electrical Breakdown' (code E01) or 'Mechanical Breakdown' (code M01). "
                  "This is not a keyword guess — every record's Breakdown Name and Reason "
                  "field carries this code as a literal text prefix, and per-prefix counts "
                  "match the Category field's value counts exactly, confirming Category is "
                  "the plant's own documented classification.\n")
    lines.append("### Category → Code verification\n")
    lines.append("| Category | Code | Event count |")
    lines.append("|---|---|---|")
    for cat, code in T.CATEGORY_CODE_MAP.items():
        n = R["category_counts"].get(cat, 0)
        lines.append(f"| {cat} | {code} | {n} |")
    lines.append(f"\nEligible (E01+M01) unique name/reason combinations: "
                 f"{R['n_mapping_rows_eligible']} of {R['n_mapping_rows_total']} total. "
                 f"{R['n_mapping_rows_needs_review']} are flagged 'Needs Review' "
                 f"(fell into a residual 'Other Confirmed ___' bucket with no matching "
                 f"family keyword — not force-classified).\n")

    lines.append("\n## Step 4.2 — Failure-Family Taxonomy (3 levels)\n")
    lines.append("Level 1: Electrical / Mechanical (from Category). "
                 "Level 2: failure family (below, data-driven). "
                 "Level 3: normalized breakdown reason (see event_level_reason_table.csv).\n")
    lines.append("| Level 1 | Level 2 Family | Unique Reasons | Events | Low-confidence rows |")
    lines.append("|---|---|---|---|---|")
    for r in tax_summary.to_dict(orient="records"):
        lines.append(f"| {r['electrical_mechanical_classification']} | {r['failure_family_level2']} | "
                     f"{r['n_unique_reasons']} | {r['n_events']} | {r['low_confidence_rows']} |")

    lines.append("\n\n## Event Granularity\n")
    lines.append(R["granularity"]["conclusion"] + "\n")
    lines.append(f"Duplicate (Mill, Date, Shift, Name, Reason) rows removed before target "
                 f"construction: {R['n_dedup_removed']}\n")

    lines.append("\n## Step 4.3 — Target Definitions\n")
    lines.append(f"Machine-day target table: {R['n_machine_days']} rows "
                 f"({R['target_df_shape'][0]} rows x {R['target_df_shape'][1]} columns), "
                 f"covering every (machine, calendar date) combination in range.\n")
    lines.append(f"- **Target A** (`A_any_eligible_breakdown`): binary, any E/M breakdown that "
                 f"machine-day. Overall positive rate: {R['overall_A_positive_rate_pct']}%.\n")
    lines.append("- **Target B** (`B_<family>_occurred`): binary per failure family, one column "
                 "per family discovered in 4.2.\n")
    lines.append("- **Target C** (`C_<family>_event_count`): integer count of events per family "
                 "per machine-day (row counts, post-deduplication, per the granularity finding).\n")
    lines.append("- **Target D**: event-level reason table (`event_level_reason_table.csv`) — "
                 "most-likely detailed reason, modeled conditional on the predicted family, "
                 "not as a flat 200+-way classification.\n")
    lines.append("- **Target E** (`E_downtime_minutes_total/max`): see feasibility check below — "
                 "only modeled as a regression where the data supports it.\n")

    lines.append("\n## Target E Feasibility (downtime)\n")
    lines.append(R["feasibility_conclusion"] + "\n")
    lines.append("| Family | Events | Mean (min) | Std | CV | Reliable for regression? |")
    lines.append("|---|---|---|---|---|---|")
    for r in feas_df.to_dict(orient="records"):
        lines.append(f"| {r['failure_family']} | {r['n_events']} | {r['mean']} | {r['std']} | "
                     f"{r['cv_(std/mean)']} | {r['reliable_for_regression']} |")

    with open("target_definition_report.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
