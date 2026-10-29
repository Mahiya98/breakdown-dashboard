"""Builds 02_target_definition.ipynb -- runnable notebook wrapping label_taxonomy.py,
so the notebook and run_target_definition.py always agree (single source of truth)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# 02 — Defining the Prediction Target (Electrical + Mechanical Breakdowns)

Builds the Step 4.1/4.2/4.3 deliverables: eligible-breakdown classification,
3-level failure taxonomy, and the machine-day target tables (A, B, C, D, E).

All logic lives in `label_taxonomy.py` so this notebook and
`run_target_definition.py` are guaranteed to produce identical results."""
))

cells.append(nbf.v4.new_code_cell(
"""import pandas as pd
import label_taxonomy as T

pd.set_option('display.max_columns', 40)
pd.set_option('display.max_colwidth', 60)

xl = pd.ExcelFile(T.SOURCE_WORKBOOK)
bdf = xl.parse('Breakdown')
print(f"Breakdown sheet: {len(bdf)} rows")
bdf['Category'].value_counts()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Step 4.1 — Verify Category is the authoritative classification

Every Breakdown Name / Reason carries a code prefix (E01-, M01-, U01-, ...).
We confirm these prefixes match the Dropdown Data Code columns and that
per-prefix counts equal Category value counts exactly."""
))
cells.append(nbf.v4.new_code_cell(
"""import re
prefix_pat = re.compile(r'^([A-Z]\\d{2})-')
name_prefixes = bdf['Breakdwon Name'].dropna().astype(str).apply(
    lambda s: (prefix_pat.match(s.strip()) or [None])[0])
print(name_prefixes.value_counts())
print()
print("Compare to Category counts:")
print(bdf['Category'].value_counts())"""
))

cells.append(nbf.v4.new_markdown_cell("## Build the full mapping table (Step 4.1 deliverable)"))
cells.append(nbf.v4.new_code_cell(
"""mapping_df = T.build_mapping_table(bdf)
print(f"{len(mapping_df)} unique (Category, Name, Reason) combinations")
print(f"Eligible (Electrical/Mechanical): {mapping_df['eligible_for_target'].sum()}")
mapping_df.head(10)"""
))

cells.append(nbf.v4.new_code_cell(
"""needs_review = mapping_df[mapping_df['review_status'] == 'Needs Review']
print(f"{len(needs_review)} combinations flagged Needs Review (not force-classified)")
needs_review[['original_breakdown_name','original_reason','failure_family_level2','event_count']]"""
))

cells.append(nbf.v4.new_markdown_cell("## Step 4.2 — Failure-family taxonomy summary"))
cells.append(nbf.v4.new_code_cell(
"""tax_summary = T.taxonomy_summary(mapping_df)
tax_summary"""
))

cells.append(nbf.v4.new_markdown_cell("## Event granularity assessment (prerequisite for Step 4.3)"))
cells.append(nbf.v4.new_code_cell(
"""granularity = T.event_granularity_report(bdf)
for k, v in granularity.items():
    print(k, '->', v)"""
))

cells.append(nbf.v4.new_markdown_cell("## Step 4.3 — Build machine-day targets (A, B, C, E)"))
cells.append(nbf.v4.new_code_cell(
"""target_df, families, n_removed = T.build_machine_day_targets(bdf, mapping_df)
print(f"Deduplicated rows removed before target construction: {n_removed}")
print(f"Failure families discovered: {families}")
print(f"Target table shape: {target_df.shape}")
target_df.head()"""
))

cells.append(nbf.v4.new_code_cell(
"""print(f"Overall Target A positive rate: {target_df['A_any_eligible_breakdown'].mean()*100:.2f}%")
for fam in families:
    col = f'B_{fam}_occurred'
    print(f"  {fam}: {target_df[col].mean()*100:.2f}% of machine-days")"""
))

cells.append(nbf.v4.new_markdown_cell("## Target D — event-level reason table"))
cells.append(nbf.v4.new_code_cell(
"""reason_table = T.event_level_reason_table(bdf, mapping_df)
print(f"{len(reason_table)} deduplicated eligible events")
reason_table.head()"""
))

cells.append(nbf.v4.new_markdown_cell("## Target E — downtime feasibility check (no model forced)"))
cells.append(nbf.v4.new_code_cell(
"""feas_df, feas_conclusion = T.downtime_feasibility_check(reason_table)
print(feas_conclusion)
feas_df"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Summary

- Eligibility is determined strictly by `Category` (E01/M01 code-verified), never by keyword
  matching on arbitrary text.
- 3-level taxonomy: Electrical/Mechanical → failure family → normalized reason, built only
  from families with real support in the data.
- Breakdown rows are individual events (not shift aggregates); duplicates are removed before
  counting.
- Five distinct targets (A–E) are defined at the appropriate grain, with Target E (downtime)
  restricted to families that pass a sample-size + variance feasibility bar — never forced."""
))

nb['cells'] = cells
with open('02_target_definition.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Wrote 02_target_definition.ipynb')
