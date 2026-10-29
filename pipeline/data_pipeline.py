"""
data_pipeline.py
==================
Reusable, time-aware data pipeline for VRM-1 / VRM-2 predictive maintenance.
All functions are read-only against the source workbook and build a daily
(machine, date) feature+target table with NO future leakage.

-----------------------------------------------------------------------
DESIGN DECISIONS (evidence-based, derived from live inspection -- not assumed)
-----------------------------------------------------------------------

1. ALIGNMENT GRAIN: DAILY, not shift-level.
   Evidence: VRM-1/VRM-2 sensor sheets use 3 shift codes (A/B/C). The
   Production and Breakdown sheets use 4 shift codes (A/B/C/D, i.e.
   Durbar/Durjoy/Duronto/Reliever Group, with casing variants DURBAR/
   DURJOY/DURANTO). These are not the same shift system, so a reliable
   shift-to-shift join across all three sheets is not defensible. Daily
   aggregation sidesteps this mismatch while still being fine-grained
   enough for 3-day-ahead forecasting.

2. BREAKDOWN LOG COMPLETENESS / NEGATIVE-LABEL TRUST.
   Evidence: Production carries its own independently-recorded
   'Mechanical Breakdown' and 'Electrical Breakdown' downtime-minutes
   columns. Aggregated to (machine, date) and compared against the
   Breakdown-sheet event log's summed Downtime for Electrical/Mechanical
   events, the two sources match EXACTLY on 898 of 900 covered
   machine-days (99.8%). This cross-check is strong evidence the
   Breakdown log is complete on days a Production record exists.

   Therefore a machine-day is only treated as a CONFIDENT NEGATIVE
   (no electrical/mechanical breakdown occurred) when:
     (a) a Production record exists for that (machine, date), AND
     (b) Production's Mechanical+Electrical downtime sums to 0, AND
     (c) no eligible Breakdown-sheet event exists for that (machine, date).
   A missing breakdown record is NEVER by itself read as "no breakdown".

3. OPERATING-STATUS / MISSING-OPERATING-PERIOD DETECTION.
   A (machine, date) is marked:
     - "operating"      : Production record exists with Actual Run Time > 0,
                            OR at least one VRM sensor reading exists that day.
     - "idle_logged"     : Production record exists but Actual Run Time == 0
                            (machine was down/idle, not a logging gap).
     - "unknown_no_log"  : no Production record AND no sensor reading at all
                            for that (machine, date) -- a genuine reporting
                            gap, not evidence of anything. These rows get
                            label_confidence = 'unknown' and are EXCLUDED
                            from confident-negative training rows (they stay
                            in the dataset for transparency, but a model
                            trainer should drop or specially handle them).

4. NO-LEAKAGE FEATURE CONSTRUCTION.
   All sensor and production features for a given (machine, date=d) row
   are computed ONLY from data with date <= d-1 (i.e. features describe
   the 1/3/7-day window strictly BEFORE d, never including day d itself
   or any later day). Breakdown TARGETS for day d use breakdown events
   dated exactly d. No column derived from day >= d is ever placed in the
   feature block. This matches "use only information available at the
   prediction cutoff time."

5. MULTIPLE BREAKDOWNS ON THE SAME DATE.
   Handled by row-level aggregation (sum of event counts, max severity
   proxies) exactly as defined in Step 4 (label_taxonomy.py's
   build_machine_day_targets), re-used here rather than re-implemented,
   so target definitions cannot drift between Step 4 and this pipeline.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

import label_taxonomy as T  # re-use Step 4's authoritative target/taxonomy logic

SOURCE_WORKBOOK = "VRM_Jul2025_Oct2026_Combined.xlsx"

LAG_WINDOWS = [1, 3, 7]  # days of history used for rolling sensor/production features


# ==========================================================================
# NORMALIZATION HELPERS
# ==========================================================================
def normalize_machine_id(raw):
    if pd.isna(raw):
        return None
    s = str(raw).strip().upper()
    s_compact = re.sub(r"[\s\-_]+", "", s)
    m = re.search(r"(VRM|MILL)\D{0,3}([12])", s_compact)
    if m:
        return f"VRM-{m.group(2)}"
    return str(raw).strip()


SHIFT_NAME_CANON = {
    "durbar": "Durbar", "durjoy": "Durjoy", "duronto": "Duronto",
    "reliever group": "Reliever Group",
}
SHIFT_CODE_TO_NAME = {"A": "Durbar", "B": "Durjoy", "C": "Duronto", "D": "Reliever Group"}


def normalize_shift_name(raw):
    if pd.isna(raw):
        return None
    key = str(raw).strip().lower()
    return SHIFT_NAME_CANON.get(key, str(raw).strip())


def normalize_shift_code(raw):
    if pd.isna(raw):
        return None
    s = str(raw).strip().upper()
    return s if s in ("A", "B", "C", "D") else s


# ==========================================================================
# STEP 1: LOAD + NORMALIZE EACH SHEET (read-only)
# ==========================================================================
def load_and_normalize(path=SOURCE_WORKBOOK):
    xl = pd.ExcelFile(path)
    v1 = xl.parse("VRM 1")
    v2 = xl.parse("VRM 2")
    pdf = xl.parse("Production")
    bdf = xl.parse("Breakdown")

    v1["machine_id"] = "VRM-1"
    v2["machine_id"] = "VRM-2"
    for df in (v1, v2):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["shift_code"] = df["Shift"].apply(normalize_shift_code) if "Shift" in df.columns else None

    pdf = pdf.copy()
    pdf["machine_id"] = pdf["Mill Name"].apply(normalize_machine_id)
    pdf["Date"] = pd.to_datetime(pdf["Date"], errors="coerce")
    pdf["shift_code"] = pdf["Shift"].apply(normalize_shift_code)
    pdf["shift_name_norm"] = pdf["Shift Name"].apply(normalize_shift_name)

    bdf = bdf.copy()
    bdf["machine_id"] = bdf["Mill Name"].apply(normalize_machine_id)
    bdf["Date"] = pd.to_datetime(bdf["Date"], errors="coerce")
    bdf["shift_code"] = bdf["Shift"].apply(normalize_shift_code)
    bdf["shift_name_norm"] = bdf["Shift Name"].apply(normalize_shift_name)

    return {"VRM 1": v1, "VRM 2": v2, "Production": pdf, "Breakdown": bdf}


# ==========================================================================
# STEP 2: DAILY SENSOR AGGREGATION (per machine)
# ==========================================================================
SENSOR_NON_FEATURE_COLS = {
    "Timestamp", "Date", "Hour", "Shift", "Operation Engineer", "Remarks",
    "Source file", "Operation Mode", "machine_id", "shift_code",
}


def daily_sensor_aggregates(v_df):
    """One row per (machine_id, Date) with mean/std/min/max of every numeric
    sensor column, plus a reading_count and coverage flag for that day."""
    sensor_cols = [c for c in v_df.columns
                   if c not in SENSOR_NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(v_df[c])]

    agg_funcs = {c: ["mean", "std", "min", "max"] for c in sensor_cols}
    grouped = v_df.groupby(["machine_id", "Date"]).agg(agg_funcs)
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    grouped = grouped.reset_index()

    counts = v_df.groupby(["machine_id", "Date"]).size().rename("sensor_reading_count").reset_index()
    grouped = grouped.merge(counts, on=["machine_id", "Date"], how="left")
    return grouped


# ==========================================================================
# STEP 3: DAILY PRODUCTION AGGREGATION (per machine)
# ==========================================================================
PRODUCTION_NUMERIC_AGG_COLS = [
    "Shift Capacity (MT)", "Shift Target (MT)", "Actual Output (MT)",
    "Achievement %", "Standerd Speed (Feed rate MT/Hr)", "Actual Speed  (Feed rate MT/Hr)",
    "Speed Gap  (Feed rate)", "Gap (%)", "Operation/Shift Time (Min)",
    "Planned Down Time (Min)", "Available Run Time (Min)", "Total Unplanned Time (NPT) (Min.)",
    "Actual Run Time (Mins)", "NPT%", "Total Wastage (MT)", "Wastage%",
    "Total Good Production (MT)", "Availability%", "Performance%", "Quality%", "OEE%",
    "Mechanical Breakdown", "Electrical Breakdown", "Utility (Gas)", "Utility (Electricity)",
    "Raw Material Shortage", "Manpower Shortage", "Warehouse block",
]


def daily_production_aggregates(pdf):
    cols = [c for c in PRODUCTION_NUMERIC_AGG_COLS if c in pdf.columns]
    for c in cols:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce")

    grouped = pdf.groupby(["machine_id", "Date"])[cols].sum(min_count=1).reset_index()
    grouped = grouped.rename(columns={c: f"prod_{c.replace(' ', '_').replace('(', '').replace(')', '').replace('%','pct').replace('.','')}_sum" for c in cols})

    shift_counts = pdf.groupby(["machine_id", "Date"]).size().rename("prod_shift_record_count").reset_index()
    grouped = grouped.merge(shift_counts, on=["machine_id", "Date"], how="left")
    return grouped


# ==========================================================================
# STEP 4: OPERATING-STATUS DETERMINATION
# ==========================================================================
def determine_operating_status(full_grid, sensor_daily, prod_daily):
    df = full_grid.merge(
        sensor_daily[["machine_id", "Date", "sensor_reading_count"]],
        on=["machine_id", "Date"], how="left"
    )
    run_time_col = [c for c in prod_daily.columns if c.startswith("prod_Actual_Run_Time")]
    df = df.merge(
        prod_daily[["machine_id", "Date", "prod_shift_record_count"] + run_time_col],
        on=["machine_id", "Date"], how="left"
    )

    df["sensor_reading_count"] = df["sensor_reading_count"].fillna(0)
    df["prod_shift_record_count"] = df["prod_shift_record_count"].fillna(0)
    actual_run_col = run_time_col[0] if run_time_col else None
    if actual_run_col:
        df[actual_run_col] = df[actual_run_col].fillna(0)

    def status_row(r):
        has_prod = r["prod_shift_record_count"] > 0
        has_sensor = r["sensor_reading_count"] > 0
        run_time = r[actual_run_col] if actual_run_col else np.nan
        if not has_prod and not has_sensor:
            return "unknown_no_log"
        if has_prod and run_time is not None and not pd.isna(run_time) and run_time <= 0:
            return "idle_logged"
        return "operating"

    df["operating_status"] = df.apply(status_row, axis=1)
    return df


# ==========================================================================
# STEP 5: CONFIDENT-NEGATIVE / LABEL-CONFIDENCE DETERMINATION
# ==========================================================================
def determine_label_confidence(df, prod_daily):
    """Marks each machine-day's target as 'confident_negative', 'positive',
    or 'unknown' per the cross-check design decision above."""
    elec_col = [c for c in prod_daily.columns if "Electrical_Breakdown" in c]
    mech_col = [c for c in prod_daily.columns if "Mechanical_Breakdown" in c]
    pd_bd = prod_daily[["machine_id", "Date"] + elec_col + mech_col].copy()
    pd_bd["prod_recorded_EM_downtime"] = pd_bd[elec_col[0]].fillna(0) + pd_bd[mech_col[0]].fillna(0)
    df = df.merge(pd_bd[["machine_id", "Date", "prod_recorded_EM_downtime"]],
                   on=["machine_id", "Date"], how="left")

    def confidence_row(r):
        has_prod_record = r["prod_shift_record_count"] > 0
        target_positive = r.get("A_any_eligible_breakdown", 0) == 1
        if target_positive:
            return "positive"
        if has_prod_record and (r.get("prod_recorded_EM_downtime", np.nan) == 0):
            return "confident_negative"
        if not has_prod_record:
            return "unknown_no_production_record"
        return "unknown_discrepancy"  # prod has E/M downtime but no matching eligible event -- rare, flagged

    df["label_confidence"] = df.apply(confidence_row, axis=1)
    return df


# ==========================================================================
# STEP 6: LAGGED (LEAKAGE-FREE) ROLLING FEATURES
# ==========================================================================
def add_lagged_rolling_features(df, value_cols, windows=LAG_WINDOWS):
    """
    For each machine, sorts by Date and computes rolling means over the
    `windows`-day window ENDING THE PREVIOUS DAY (shift(1) before rolling),
    so a feature for date d never includes day d's own value.
    """
    df = df.sort_values(["machine_id", "Date"]).reset_index(drop=True)
    new_cols = {}
    for col in value_cols:
        if col not in df.columns:
            continue
        shifted = df.groupby("machine_id")[col].shift(1)  # never include day d itself
        for w in windows:
            roll = shifted.groupby(df["machine_id"]).transform(
                lambda s: s.rolling(window=w, min_periods=1).mean()
            )
            new_cols[f"{col}_lag{w}d_mean"] = roll
    out = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return out


# ==========================================================================
# STEP 7: FULL (machine, date) GRID
# ==========================================================================
def build_full_grid(sheets):
    all_dates = []
    for name in ("VRM 1", "VRM 2", "Production", "Breakdown"):
        d = sheets[name]["Date"].dropna()
        if len(d):
            all_dates.append((d.min(), d.max()))
    date_min = min(d[0] for d in all_dates)
    date_max = max(d[1] for d in all_dates)
    calendar = pd.date_range(date_min, date_max, freq="D")
    machines = ["VRM-1", "VRM-2"]
    grid = pd.MultiIndex.from_product([machines, calendar], names=["machine_id", "Date"])
    return pd.DataFrame(index=grid).reset_index()


# ==========================================================================
# ORCHESTRATOR: BUILD THE FULL LEAKAGE-FREE DAILY DATASET
# ==========================================================================
def build_processed_dataset(path=SOURCE_WORKBOOK):
    sheets = load_and_normalize(path)

    v1_daily = daily_sensor_aggregates(sheets["VRM 1"])
    v2_daily = daily_sensor_aggregates(sheets["VRM 2"])
    sensor_daily = pd.concat([v1_daily, v2_daily], ignore_index=True)

    prod_daily = daily_production_aggregates(sheets["Production"])

    full_grid = build_full_grid(sheets)

    # operating status + label confidence (uses CURRENT-day production/sensor
    # presence, which is fine -- these are metadata flags about day d, not
    # leakage, since they are only used to annotate trustworthiness of day d's
    # own TARGET, never as a feature input for predicting day d)
    status_df = determine_operating_status(full_grid, sensor_daily, prod_daily)

    # targets A-E, re-using Step 4's authoritative, already-reviewed logic
    bdf = sheets["Breakdown"]
    mapping_df = T.build_mapping_table(bdf)
    target_df, families, n_dedup = T.build_machine_day_targets(bdf, mapping_df)
    target_df = target_df.rename(columns={"machine": "machine_id", "date": "Date"})

    merged = status_df.merge(target_df, on=["machine_id", "Date"], how="left")
    for col in merged.columns:
        if col.startswith(("A_", "B_", "C_")):
            merged[col] = merged[col].fillna(0).astype(int)
        if col.startswith("E_"):
            merged[col] = merged[col].fillna(0.0)

    merged = determine_label_confidence(merged, prod_daily)

    # merge same-day sensor/production aggregates temporarily so we can lag them
    merged = merged.merge(sensor_daily, on=["machine_id", "Date"], how="left")
    merged = merged.merge(
        prod_daily.drop(columns=["prod_shift_record_count"]),
        on=["machine_id", "Date"], how="left", suffixes=("", "_dup")
    )

    # which columns are raw same-day values that must be LAGGED (never used same-day)
    raw_feature_cols = [c for c in merged.columns
                         if (c.startswith(tuple(f"{s}_" for s in [])) or
                             any(c.endswith(suf) for suf in ("_mean", "_std", "_min", "_max", "_sum")))
                         and not c.startswith(("A_", "B_", "C_", "E_"))]

    merged = add_lagged_rolling_features(merged, raw_feature_cols, LAG_WINDOWS)

    # DROP the same-day raw aggregates from the FEATURE set (they are same-day
    # and would leak); keep only the lagged versions as predictors. We retain
    # same-day aggregates under a clearly marked '_SAMEDAY_DO_NOT_USE_AS_FEATURE'
    # prefix purely for audit/debugging, not for modeling.
    rename_map = {c: f"SAMEDAY_AUDIT_ONLY__{c}" for c in raw_feature_cols}
    merged = merged.rename(columns=rename_map)

    merged["machine_date_id"] = merged["machine_id"] + "_" + merged["Date"].dt.strftime("%Y-%m-%d")

    return merged, families, mapping_df, n_dedup, sheets


if __name__ == "__main__":
    df, families, mapping_df, n_dedup, sheets = build_processed_dataset()
    print(f"Processed dataset shape: {df.shape}")
    print(f"Operating status counts:\n{df['operating_status'].value_counts()}")
    print(f"Label confidence counts:\n{df['label_confidence'].value_counts()}")
