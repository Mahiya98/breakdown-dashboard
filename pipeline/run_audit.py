"""
run_audit.py
=============
Runs the full Task-1-style data audit using audit_lib.py and writes:
    - data_audit_report.md
    - data_dictionary.xlsx
    - data_quality_report.xlsx
    - audit_v2_output/audit_results.json   (raw results, source of truth for the PDF)

The original workbook (VRM_Jul2025_Oct2026_Combined.xlsx) is opened
READ-ONLY via pandas and is never written to or modified.

Usage:
    python run_audit.py
Requires: pandas, numpy, openpyxl
"""

import json
from pathlib import Path

import pandas as pd

import audit_lib as A

OUT_DIR = Path("audit_v2_output")
OUT_DIR.mkdir(exist_ok=True)


def to_jsonable(obj):
    """Recursively convert pandas/numpy objects to plain JSON-safe types."""
    import numpy as np
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if pd.isna(obj) if not isinstance(obj, (list, dict)) else False:
        return None
    return obj


def main():
    print(f"Loading workbook read-only: {A.SOURCE_WORKBOOK}")
    sheets, results = A.run_full_audit()

    # -------------------- data_dictionary.xlsx --------------------
    with pd.ExcelWriter(OUT_DIR / "data_dictionary.xlsx", engine="openpyxl") as writer:
        results["data_dictionary"].to_excel(writer, sheet_name="Data Dictionary", index=False)
        for mname, dfp in results["machine_parameters"].items():
            dfp.to_excel(writer, sheet_name=f"{mname} Parameters", index=False)
    print("Wrote data_dictionary.xlsx")

    # -------------------- data_quality_report.xlsx --------------------
    with pd.ExcelWriter(OUT_DIR / "data_quality_report.xlsx", engine="openpyxl") as writer:
        results["sheet_inventory"].to_excel(writer, sheet_name="Sheet Inventory", index=False)
        results["date_coverage"].to_excel(writer, sheet_name="Date Coverage", index=False)
        results["missingness_summary"].to_excel(writer, sheet_name="Missingness Summary", index=False)

        dup_rows = []
        for sheet, info in results["duplicate_and_timestamp_audit"].items():
            row = {"sheet": sheet}
            row.update(info)
            dup_rows.append(row)
        pd.DataFrame(dup_rows).to_excel(writer, sheet_name="Duplicates", index=False)

        bf = results["breakdown_frequencies"]
        if "category_counts" in bf:
            pd.Series(bf["category_counts"], name="count").rename_axis("category").reset_index()\
                .to_excel(writer, sheet_name="Category Freq", index=False)
        if "breakdown_name_counts" in bf:
            pd.Series(bf["breakdown_name_counts"], name="count").rename_axis("breakdown_name").reset_index()\
                .to_excel(writer, sheet_name="Breakdown Name Freq", index=False)
        if "reason_counts" in bf:
            pd.Series(bf["reason_counts"], name="count").rename_axis("reason").reset_index()\
                .to_excel(writer, sheet_name="Reason Freq", index=False)

        cross = results["crossref_breakdown_dropdown"]
        cross_summary = {k: v for k, v in cross.items() if not isinstance(v, list)}
        pd.DataFrame([cross_summary]).to_excel(writer, sheet_name="Dropdown Crossref Summary", index=False)
        pd.DataFrame({"unmapped_breakdown_names": cross.get("breakdown_names_not_in_dropdown", [])})\
            .to_excel(writer, sheet_name="Unmapped Breakdown Names", index=False)
        pd.DataFrame({"unmapped_reasons": cross.get("reasons_not_in_dropdown", [])})\
            .to_excel(writer, sheet_name="Unmapped Reasons", index=False)

        inc = results["inconsistent_breakdown_names"]
        inc_summary = {k: v for k, v in inc.items() if not isinstance(v, (list, dict))}
        pd.DataFrame([inc_summary]).to_excel(writer, sheet_name="Name Consistency Summary", index=False)

        pd.DataFrame({"limitation": results["limitations"]}).to_excel(writer, sheet_name="Limitations", index=False)
    print("Wrote data_quality_report.xlsx")

    # -------------------- audit_results.json --------------------
    json_safe = {
        "sheet_inventory": to_jsonable(results["sheet_inventory"]),
        "date_coverage": to_jsonable(results["date_coverage"]),
        "missingness_summary": to_jsonable(results["missingness_summary"]),
        "duplicate_and_timestamp_audit": to_jsonable(results["duplicate_and_timestamp_audit"]),
        "breakdown_frequencies": to_jsonable(results["breakdown_frequencies"]),
        "crossref_breakdown_dropdown": to_jsonable(results["crossref_breakdown_dropdown"]),
        "inconsistent_breakdown_names": to_jsonable(results["inconsistent_breakdown_names"]),
        "linkability_assessment": to_jsonable(results["linkability_assessment"]),
        "limitations": results["limitations"],
        "machine_parameter_counts": {m: len(df) for m, df in results["machine_parameters"].items()},
    }
    with open(OUT_DIR / "audit_results.json", "w") as f:
        json.dump(json_safe, f, indent=2, default=str)
    print("Wrote audit_results.json")

    # -------------------- data_audit_report.md --------------------
    write_markdown_report(results)
    print("Wrote data_audit_report.md")

    print(f"\nAll audit outputs in: {OUT_DIR.resolve()}")
    return sheets, results


def write_markdown_report(R):
    lines = []
    lines.append("# Data Audit Report — VRM Predictive Maintenance Workbook\n")
    lines.append(f"Source file: `{A.SOURCE_WORKBOOK}` (opened read-only; never modified)\n")

    lines.append("\n## 1. Workbook & Sheet Inventory\n")
    lines.append(R["sheet_inventory"].to_markdown(index=False))

    lines.append("\n\n## 2. Date Coverage Per Sheet\n")
    lines.append(R["date_coverage"].to_markdown(index=False))

    lines.append("\n\n## 3. Missing-Value Summary Per Sheet\n")
    lines.append(R["missingness_summary"].to_markdown(index=False))
    lines.append("\n\n(Full per-column missingness is in `data_dictionary.xlsx`.)\n")

    lines.append("\n## 4. Duplicate Records & Timestamp Issues\n")
    for sheet, info in R["duplicate_and_timestamp_audit"].items():
        lines.append(f"### {sheet}")
        for k, v in info.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append("\n## 5. Breakdown Category / Name / Reason Frequencies\n")
    bf = R["breakdown_frequencies"]
    if "mill_name_counts" in bf:
        lines.append("### By Machine (Mill Name as recorded)")
        for k, v in bf["mill_name_counts"].items():
            lines.append(f"- {k}: {v}")
    if "category_counts" in bf:
        lines.append("\n### By Category")
        for k, v in sorted(bf["category_counts"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- {k}: {v}")
    if "reason_counts" in bf:
        top_reasons = sorted(bf["reason_counts"].items(), key=lambda kv: -kv[1])[:20]
        lines.append(f"\n### Top 20 Reasons (of {len(bf['reason_counts'])} distinct)")
        for k, v in top_reasons:
            lines.append(f"- {k}: {v}")

    lines.append("\n## 6. Breakdown ↔ Dropdown Data Cross-Reference\n")
    cross = R["crossref_breakdown_dropdown"]
    for k, v in cross.items():
        if not isinstance(v, list):
            lines.append(f"- {k}: {v}")
    if cross.get("breakdown_names_not_in_dropdown"):
        lines.append(f"\n### Breakdown names NOT found in Dropdown Data "
                      f"({len(cross['breakdown_names_not_in_dropdown'])})")
        for n in cross["breakdown_names_not_in_dropdown"][:40]:
            lines.append(f"- {n}")
        if len(cross["breakdown_names_not_in_dropdown"]) > 40:
            lines.append(f"- ... ({len(cross['breakdown_names_not_in_dropdown']) - 40} more; "
                          f"see data_quality_report.xlsx)")

    lines.append("\n## 7. Invalid / Inconsistent Breakdown Name Report\n")
    inc = R["inconsistent_breakdown_names"]
    for k, v in inc.items():
        if not isinstance(v, (list, dict)):
            lines.append(f"- {k}: {v}")
    if inc.get("casing_variant_examples"):
        lines.append("\n### Example casing/whitespace variants of the same label")
        for norm, variants in inc["casing_variant_examples"].items():
            lines.append(f"- `{norm}` → {variants}")
    if inc.get("placeholder_values_found"):
        lines.append(f"\n### Placeholder/empty-like values found: {inc['placeholder_values_found']}")

    lines.append("\n## 8. Linkability Assessment\n")
    link = R["linkability_assessment"]
    for key, info in link.items():
        lines.append(f"### {key}")
        for k, v in info.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append("\n## 9. Data Limitations (derived from the checks above)\n")
    for lim in R["limitations"]:
        lines.append(f"- {lim}")

    lines.append("\n\n---\n*This report was generated entirely by run_audit.py from live computations "
                  "on the source workbook. No figures were manually edited or estimated.*")

    with open("data_audit_report.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
