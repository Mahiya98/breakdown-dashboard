"""
build_training_dataset.py
===========================
Orchestrates data_pipeline.py and writes every Step-5 (preprocessing)
deliverable:

    processed_dataset.parquet        -- combined, both machines, machine_id column
    vrm1_processed_dataset.parquet   -- VRM-1 only
    vrm2_processed_dataset.parquet   -- VRM-2 only
    breakdown_taxonomy.xlsx          -- re-exported from Step 4 (single source of truth)
    feature_data_dictionary.md       -- documents every feature/target column
    preprocessing_output/pipeline_results.json  -- machine-readable, PDF source

Usage: python build_training_dataset.py
Requires: pandas, numpy, openpyxl, pyarrow
"""

import json
from pathlib import Path

import pandas as pd

import data_pipeline as P
import label_taxonomy as T

OUT_DIR = Path("preprocessing_output")
OUT_DIR.mkdir(exist_ok=True)


def main():
    print("Building leakage-free daily dataset (read-only against source workbook)...")
    df, families, mapping_df, n_dedup, sheets = P.build_processed_dataset()

    # ---- processed_dataset.parquet (combined) ----
    df.to_parquet("processed_dataset.parquet", index=False)
    print(f"Wrote processed_dataset.parquet  shape={df.shape}")

    # ---- per-machine parquet files ----
    for m in ["VRM-1", "VRM-2"]:
        sub = df[df["machine_id"] == m].reset_index(drop=True)
        fname = f"{m.lower().replace('-', '')}_processed_dataset.parquet"
        sub.to_parquet(fname, index=False)
        print(f"Wrote {fname}  shape={sub.shape}")

    # ---- breakdown_taxonomy.xlsx (single source of truth: label_taxonomy.py) ----
    tax_summary = T.taxonomy_summary(mapping_df)
    with pd.ExcelWriter("breakdown_taxonomy.xlsx", engine="openpyxl") as w:
        mapping_df.to_excel(w, sheet_name="Full Mapping (Levels 1-3)", index=False)
        tax_summary.to_excel(w, sheet_name="Family Summary (Level 2)", index=False)
    print("Wrote breakdown_taxonomy.xlsx")

    # ---- feature data dictionary ----
    write_feature_dictionary(df)
    print("Wrote feature_data_dictionary.md")

    # ---- results json for PDF ----
    results = summarize_results(df, families, n_dedup, sheets)
    with open(OUT_DIR / "pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Wrote preprocessing_output/pipeline_results.json")

    return df, families, mapping_df


def summarize_results(df, families, n_dedup, sheets):
    import numpy as np

    def jsonable(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        return x

    op_counts = df["operating_status"].value_counts().to_dict()
    conf_counts = df["label_confidence"].value_counts().to_dict()
    per_machine_op = df.groupby(["machine_id", "operating_status"]).size().unstack(fill_value=0).to_dict(orient="index")
    per_machine_conf = df.groupby(["machine_id", "label_confidence"]).size().unstack(fill_value=0).to_dict(orient="index")

    feature_cols = [c for c in df.columns if "_lag" in c and c.endswith("_mean")]
    sameday_audit_cols = [c for c in df.columns if c.startswith("SAMEDAY_AUDIT_ONLY__")]
    target_cols = [c for c in df.columns if c.startswith(("A_", "B_", "C_", "E_"))]

    trainable = df[df["label_confidence"].isin(["positive", "confident_negative"])]

    return {
        "dataset_shape": list(df.shape),
        "n_machines": int(df["machine_id"].nunique()),
        "date_range": [str(df["Date"].min()), str(df["Date"].max())],
        "n_calendar_days": int(df["Date"].nunique()),
        "operating_status_counts": {str(k): jsonable(v) for k, v in op_counts.items()},
        "label_confidence_counts": {str(k): jsonable(v) for k, v in conf_counts.items()},
        "per_machine_operating_status": per_machine_op,
        "per_machine_label_confidence": per_machine_conf,
        "n_feature_columns_lagged": len(feature_cols),
        "n_sameday_audit_only_columns": len(sameday_audit_cols),
        "n_target_columns": len(target_cols),
        "target_columns": target_cols,
        "families": families,
        "n_dedup_removed_at_target_build": int(n_dedup),
        "n_trainable_rows_(positive_or_confident_negative)": int(len(trainable)),
        "n_excluded_unknown_rows": int(len(df) - len(trainable)),
        "trainable_positive_rate_pct": round(float((trainable["A_any_eligible_breakdown"] == 1).mean() * 100), 2),
    }


def write_feature_dictionary(df):
    lines = []
    lines.append("# Feature & Target Data Dictionary\n")
    lines.append(f"Processed dataset: {df.shape[0]} rows x {df.shape[1]} columns. "
                 "One row = one (machine_id, Date) combination.\n")

    lines.append("\n## Identifier Columns\n")
    lines.append("| Column | Description |")
    lines.append("|---|---|")
    lines.append("| `machine_id` | Normalized machine identifier: 'VRM-1' or 'VRM-2' |")
    lines.append("| `Date` | Calendar date (daily grain; shift-level alignment was ruled out "
                 "-- see methodology report) |")
    lines.append("| `machine_date_id` | `machine_id` + '_' + Date, unique row key |")

    lines.append("\n## Data-Quality / Trust Columns\n")
    lines.append("| Column | Values | Meaning |")
    lines.append("|---|---|---|")
    lines.append("| `operating_status` | operating / idle_logged / unknown_no_log | "
                 "Whether the machine was confirmed running, confirmed idle, or has no log "
                 "at all for that day (genuine reporting gap, not evidence of anything) |")
    lines.append("| `label_confidence` | positive / confident_negative / "
                 "unknown_no_production_record / unknown_discrepancy | Whether the "
                 "machine-day's breakdown target (A) can be trusted: 'positive' = a real "
                 "breakdown event is recorded; 'confident_negative' = Production's "
                 "independently-logged breakdown-downtime columns corroborate zero "
                 "breakdowns; 'unknown_*' = insufficient evidence either way -- EXCLUDE "
                 "these rows from training rather than treating them as negatives |")
    lines.append("| `sensor_reading_count` | int | Number of raw hourly sensor rows logged "
                 "that day (same-day; audit metadata, not a feature) |")
    lines.append("| `prod_shift_record_count` | int | Number of Production shift-records "
                 "logged that day (same-day; audit metadata, not a feature) |")

    lines.append("\n## Target Columns (defined in Step 4 / label_taxonomy.py, re-used here)\n")
    target_cols = [c for c in df.columns if c.startswith(("A_", "B_", "C_", "E_"))]
    lines.append("| Column | Type | Meaning |")
    lines.append("|---|---|---|")
    for c in target_cols:
        if c.startswith("A_"):
            meaning = "Binary: any eligible (Electrical/Mechanical) breakdown that machine-day"
        elif c.startswith("B_"):
            meaning = f"Binary: breakdown occurred within failure family '{c[2:-9]}'"
        elif c.startswith("C_"):
            meaning = f"Integer: number of breakdown events in family '{c[2:-12]}' that machine-day"
        elif c.startswith("E_"):
            meaning = "Downtime minutes (total or max) for eligible breakdowns that machine-day"
        else:
            meaning = ""
        lines.append(f"| `{c}` | {'int' if c.startswith(('A_','B_','C_')) else 'float'} | {meaning} |")

    lines.append("\n## Feature Columns (LEAKAGE-FREE, lagged)\n")
    lines.append("Every predictive feature is a rolling-window AGGREGATE of data from the "
                 "**prior** 1, 3, or 7 days (strictly excluding the prediction day itself), "
                 "named `<original_sensor_or_production_column>_<stat>_lag<N>d_mean`. "
                 "For example `Main_Motor_Power_|_KW_mean_lag3d_mean` is the mean of the "
                 "daily-mean Main Motor Power over the 3 days immediately before the "
                 "prediction date -- NEVER including the prediction date's own readings.\n")
    lag_cols = [c for c in df.columns if "_lag" in c and c.endswith("_mean")]
    lines.append(f"Total lagged feature columns: {len(lag_cols)} (sourced from VRM-1/VRM-2 "
                 f"sensor columns and Production columns, x {{mean,std,min,max,sum}} daily "
                 f"stat x {{1,3,7}}-day lag window).\n")

    lines.append("\n## Same-Day Audit-Only Columns (DO NOT USE AS MODEL FEATURES)\n")
    sameday_cols = [c for c in df.columns if c.startswith("SAMEDAY_AUDIT_ONLY__")]
    lines.append(f"{len(sameday_cols)} columns prefixed `SAMEDAY_AUDIT_ONLY__` contain the "
                 "RAW same-day sensor/production aggregates. They are retained purely for "
                 "data-quality auditing and debugging the lag computation. They describe "
                 "the prediction day itself and **must never be used as a model input** -- "
                 "doing so would leak same-day information into a forecast that is supposed "
                 "to be made before that day's readings exist.\n")

    lines.append("\n## No-Leakage Guarantee (summary)\n")
    lines.append("- All feature columns are shift(1)-then-rolled, i.e. computed from "
                 "`Date <= prediction_date - 1 day` only.\n"
                 "- Target columns (A-E) are computed from breakdown events dated exactly "
                 "`prediction_date`, as required for a same-day target during training/"
                 "evaluation (a 3-day-ahead *forecast* is produced at serving time by "
                 "shifting the feature cutoff forward, done in the modeling stage, not "
                 "here).\n"
                 "- `operating_status` and `label_confidence` use same-day Production/"
                 "sensor *presence* only as metadata flags about the target's trustworthiness "
                 "-- they are never included as model features.\n")

    with open("feature_data_dictionary.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
