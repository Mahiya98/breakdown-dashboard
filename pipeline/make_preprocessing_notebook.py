"""Builds 02_data_preprocessing.ipynb -- runnable notebook wrapping data_pipeline.py,
so the notebook and build_training_dataset.py always agree (single source of truth)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# 02 — Data Preprocessing: Unified, Time-Aware Dataset for VRM-1 / VRM-2

Builds a daily (machine_id, Date) feature + target table with **no future
leakage**. All logic lives in `data_pipeline.py`, so this notebook and
`build_training_dataset.py` are guaranteed to produce identical output.

## Key design decisions (evidence-based, verified below)
1. **Daily grain, not shift-level** — VRM sensor sheets use 3 shift codes
   (A/B/C); Production/Breakdown sheets use 4 (A/B/C/D = Durbar/Durjoy/
   Duronto/Reliever Group). Mismatched shift systems make a reliable
   shift-level join indefensible.
2. **Breakdown-log completeness cross-check** — Production's own
   Mechanical/Electrical downtime columns corroborate the Breakdown-sheet
   event log on 898/900 (99.8%) covered machine-days.
3. **Confident-negative logic** — a machine-day is only a trusted "no
   breakdown" negative when a Production record exists AND shows zero
   E/M downtime AND no Breakdown event is logged. Missing records are
   never silently treated as negatives.
4. **Leakage-free features** — every feature is a rolling mean over the
   prior 1/3/7 days, computed via `shift(1)` before rolling, so day d's
   features never include day d's own readings."""
))

cells.append(nbf.v4.new_code_cell(
"""import pandas as pd
import data_pipeline as P
import label_taxonomy as T

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 160)"""
))

cells.append(nbf.v4.new_markdown_cell("## 1. Verify shift-system mismatch (justifies daily grain)"))
cells.append(nbf.v4.new_code_cell(
"""sheets = P.load_and_normalize()
v1, pdf, bdf = sheets['VRM 1'], sheets['Production'], sheets['Breakdown']

print('VRM 1 sensor sheet shift codes:', sorted(v1['Shift'].dropna().unique().tolist()))
print('Production shift codes:', sorted(pdf['Shift'].dropna().unique().tolist()))
print('Production shift NAMES:', sorted(pdf['Shift Name'].dropna().unique().tolist()))"""
))

cells.append(nbf.v4.new_markdown_cell("## 2. Verify breakdown-log completeness via Production downtime cross-check"))
cells.append(nbf.v4.new_code_cell(
"""prod_daily_bd = pdf.groupby(['machine_id','Date'])[['Mechanical Breakdown','Electrical Breakdown']].sum().reset_index()
prod_daily_bd['prod_total'] = prod_daily_bd['Mechanical Breakdown'] + prod_daily_bd['Electrical Breakdown']

elig = bdf[bdf['Category'].isin(['Electrical Breakdown','Mechanical Breakdown'])]
bdown_daily = elig.groupby(['machine_id','Date'])['Downtime'].sum().reset_index().rename(columns={'Downtime':'bdown_total'})

merged = prod_daily_bd.merge(bdown_daily, on=['machine_id','Date'], how='outer').fillna(0)
merged['diff'] = merged['prod_total'] - merged['bdown_total']
print(f"Covered machine-days: {len(merged)}")
print(f"Exact match: {(merged['diff']==0).sum()} ({(merged['diff']==0).mean()*100:.1f}%)")
print(f"Confident clean negatives (both zero): {((merged['prod_total']==0)&(merged['bdown_total']==0)).sum()}")"""
))

cells.append(nbf.v4.new_markdown_cell("## 3. Build the daily sensor and production aggregates"))
cells.append(nbf.v4.new_code_cell(
"""v2 = sheets['VRM 2']
sensor_daily = pd.concat([P.daily_sensor_aggregates(v1), P.daily_sensor_aggregates(v2)], ignore_index=True)
prod_daily = P.daily_production_aggregates(pdf)
print('Sensor daily shape:', sensor_daily.shape)
print('Production daily shape:', prod_daily.shape)
sensor_daily.head()"""
))

cells.append(nbf.v4.new_markdown_cell("## 4. Full (machine, date) grid + operating-status determination"))
cells.append(nbf.v4.new_code_cell(
"""full_grid = P.build_full_grid(sheets)
status_df = P.determine_operating_status(full_grid, sensor_daily, prod_daily)
print(status_df['operating_status'].value_counts())
status_df.groupby('machine_id')['operating_status'].value_counts().unstack(fill_value=0)"""
))

cells.append(nbf.v4.new_markdown_cell("## 5. Targets (re-using Step 4's authoritative taxonomy) + label confidence"))
cells.append(nbf.v4.new_code_cell(
"""mapping_df = T.build_mapping_table(bdf)
target_df, families, n_dedup = T.build_machine_day_targets(bdf, mapping_df)
target_df = target_df.rename(columns={'machine':'machine_id','date':'Date'})
print(f"Families: {families}")
print(f"Duplicate rows removed before target build: {n_dedup}")

merged_status = status_df.merge(target_df, on=['machine_id','Date'], how='left')
for col in merged_status.columns:
    if col.startswith(('A_','B_','C_')):
        merged_status[col] = merged_status[col].fillna(0).astype(int)
    if col.startswith('E_'):
        merged_status[col] = merged_status[col].fillna(0.0)

conf_df = P.determine_label_confidence(merged_status, prod_daily)
conf_df['label_confidence'].value_counts()"""
))

cells.append(nbf.v4.new_markdown_cell("## 6. Build the full leakage-free dataset (single call, same as build_training_dataset.py)"))
cells.append(nbf.v4.new_code_cell(
"""df, families, mapping_df, n_dedup, sheets = P.build_processed_dataset()
print(f"Final dataset shape: {df.shape}")
df[['machine_id','Date','operating_status','label_confidence','A_any_eligible_breakdown']].head(10)"""
))

cells.append(nbf.v4.new_markdown_cell("## 7. Verify NO leakage: lag-1 feature on day d must equal the raw same-day value on day d-1"))
cells.append(nbf.v4.new_code_cell(
"""v1_df = df[df['machine_id']=='VRM-1'].sort_values('Date').reset_index(drop=True)
same_col = 'SAMEDAY_AUDIT_ONLY__Main Motor Power | KW_mean'
lag1_col = 'Main Motor Power | KW_mean_lag1d_mean'

check_rows = v1_df[[same_col, lag1_col]].dropna()
idx = check_rows.index[1]
prev_sameday = v1_df.loc[idx-1, same_col]
curr_lag1 = v1_df.loc[idx, lag1_col]
print(f"Prior day same-day value: {prev_sameday}")
print(f"Current day lag1d feature: {curr_lag1}")
assert prev_sameday == curr_lag1, "LEAKAGE DETECTED"
print("PASS: lag-1 feature matches prior day exactly -- no same-day leakage.")"""
))

cells.append(nbf.v4.new_markdown_cell("## Summary"))
cells.append(nbf.v4.new_code_cell(
"""print(f"Dataset: {df.shape[0]} machine-days x {df.shape[1]} columns")
print(f"Trainable rows (positive or confident_negative): "
      f"{df['label_confidence'].isin(['positive','confident_negative']).sum()}")
print(f"Excluded as unknown: "
      f"{(~df['label_confidence'].isin(['positive','confident_negative'])).sum()}")"""
))

nb['cells'] = cells
with open('02_data_preprocessing.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Wrote 02_data_preprocessing.ipynb')
