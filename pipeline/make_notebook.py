"""Builds 01_data_audit.ipynb -- a runnable notebook version of the audit,
using the same audit_lib.py functions so results are guaranteed identical
to run_audit.py's output (single source of truth: audit_lib.py)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# 01 — Data Audit: VRM Predictive Maintenance Workbook

Read-only inspection of `VRM_Jul2025_Oct2026_Combined.xlsx`.
Sheets: `VRM 1`, `VRM 2`, `Production`, `Breakdown`, `Dropdown Data`.

This notebook **never writes to the source workbook**. All audit logic
lives in `audit_lib.py` so this notebook and `run_audit.py` always agree."""
))

cells.append(nbf.v4.new_code_cell(
"""import pandas as pd
import audit_lib as A

pd.set_option('display.max_columns', 80)
pd.set_option('display.width', 160)

sheets, sheet_names = A.load_workbook()
print("Sheets found:", sheet_names)
for name, df in sheets.items():
    print(f"  {name}: {df.shape[0]} rows x {df.shape[1]} cols")"""
))

cells.append(nbf.v4.new_markdown_cell("## 1. Workbook & Sheet Inventory"))
cells.append(nbf.v4.new_code_cell(
"""inventory = A.sheet_inventory(sheets)
inventory"""
))

cells.append(nbf.v4.new_markdown_cell("## 2. Data Dictionary (every column, every sheet)"))
cells.append(nbf.v4.new_code_cell(
"""data_dict = A.data_dictionary(sheets)
print(f"{len(data_dict)} columns documented across all sheets")
data_dict.head(20)"""
))

cells.append(nbf.v4.new_markdown_cell("## 3. Machine Parameters & Units (VRM 1 / VRM 2)"))
cells.append(nbf.v4.new_code_cell(
"""params = A.machine_parameter_list(sheets)
for mname, df in params.items():
    print(f"--- {mname}: {len(df)} parameters ---")
    display(df)"""
))

cells.append(nbf.v4.new_markdown_cell("## 4. Record Counts & Date Coverage Per Sheet"))
cells.append(nbf.v4.new_code_cell(
"""coverage = A.date_coverage(sheets)
coverage"""
))

cells.append(nbf.v4.new_markdown_cell("## 5. Missing-Value Percentages"))
cells.append(nbf.v4.new_code_cell(
"""missingness = A.missingness_summary(sheets)
missingness"""
))

cells.append(nbf.v4.new_markdown_cell("## 6. Duplicate Records & Inconsistent Timestamps"))
cells.append(nbf.v4.new_code_cell(
"""dup_audit = A.duplicate_and_timestamp_audit(sheets)
for sheet, info in dup_audit.items():
    print(sheet, '->', info)"""
))

cells.append(nbf.v4.new_markdown_cell("## 7. Breakdown Category / Reason Frequencies"))
cells.append(nbf.v4.new_code_cell(
"""freqs = A.breakdown_frequencies(sheets)
print("Category counts:")
display(pd.Series(freqs.get('category_counts', {})).sort_values(ascending=False))
print("\\nTop 15 breakdown names:")
display(pd.Series(freqs.get('breakdown_name_counts', {})).sort_values(ascending=False).head(15))
print("\\nTop 15 reasons:")
display(pd.Series(freqs.get('reason_counts', {})).sort_values(ascending=False).head(15))"""
))

cells.append(nbf.v4.new_markdown_cell("## 8. Cross-Reference: Breakdown <-> Dropdown Data"))
cells.append(nbf.v4.new_code_cell(
"""cross = A.crossref_breakdown_dropdown(sheets)
for k, v in cross.items():
    if not isinstance(v, list):
        print(k, ':', v)
print("\\nSample unmapped breakdown names:", cross['breakdown_names_not_in_dropdown'][:15])
print("Sample unmapped reasons:", cross['reasons_not_in_dropdown'][:15])"""
))

cells.append(nbf.v4.new_markdown_cell("## 9. Invalid / Unmapped / Inconsistent Breakdown Names"))
cells.append(nbf.v4.new_code_cell(
"""inconsistent = A.inconsistent_breakdown_names(sheets)
for k, v in inconsistent.items():
    if not isinstance(v, dict):
        print(k, ':', v)"""
))

cells.append(nbf.v4.new_markdown_cell(
    "## 10. Linkability: Can Breakdown records be reliably joined to machine parameters and production?"))
cells.append(nbf.v4.new_code_cell(
"""linkability = A.linkability_assessment(sheets)
for k, v in linkability.items():
    print(k, '->', v)"""
))

cells.append(nbf.v4.new_markdown_cell("## 11. Data Limitations (derived from checks above, not assumed)"))
cells.append(nbf.v4.new_code_cell(
"""results_bundle = {
    'missingness_summary': missingness,
    'duplicate_and_timestamp_audit': dup_audit,
    'crossref_breakdown_dropdown': cross,
    'linkability_assessment': linkability,
    'date_coverage': coverage,
}
limitations = A.derive_limitations(sheets, results_bundle)
for l in limitations:
    print('-', l)"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Summary

This notebook reproduces every number in `data_audit_report.md`,
`data_dictionary.xlsx`, and `data_quality_report.xlsx` — all four
artifacts are generated from the same `audit_lib.py` functions, so
they cannot drift apart. The source workbook was opened read-only
throughout and was never modified."""
))

nb['cells'] = cells
with open("01_data_audit.ipynb", "w") as f:
    nbf.write(nb, f)
print("Wrote 01_data_audit.ipynb")
