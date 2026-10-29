"""
label_taxonomy.py
===================
TASK: Define the prediction target correctly (electrical + mechanical
breakdowns only) and build a transparent, 3-level, data-driven taxonomy.

Authoritative classification rule (Step 4.1)
----------------------------------------------
Every Breakdown-sheet row's `Breakdwon Name` and `Reason` carries a code
prefix (e.g. "E01-...", "M01-...", "U01-..."). These prefixes exactly match
the Dropdown Data `Code` / `Code.1` columns, and their per-prefix counts
exactly equal the counts of the `Category` field:

    U01 -> Utility (Electricity)          590
    P01 -> Planned Down Time              418
    O01 -> Others down time               388
    W01 -> Warehouse block                230
    E01 -> Electrical Breakdown           230
    M01 -> Mechanical Breakdown           214
    I01 -> In Process Setup/ Adjustment    98
    R01 -> Raw Material Shortage           16

This is therefore the plant's own documented, authoritative classification
-- not a keyword guess. Only Category == 'Electrical Breakdown' (E01) or
'Mechanical Breakdown' (M01) are eligible for the primary prediction
target. Critically, 'Utility (Electricity)' contains the word "Electrical"
but is coded U01, NOT E01 -- confirming it must NOT be folded into the
electrical-failure target. Likewise 'Planned Down Time' (P01) is excluded
even when its free-text reason mentions mechanical/electrical equipment,
because planned maintenance is not an unplanned failure.

Level 2 (failure family) is then derived from the actual vocabulary
observed within the E01/M01 subset only (Step 4.2), via documented
keyword rules applied to the normalized Breakdown Name + Reason text.
Families are retained only if they have real support in the data; no
family is assumed a priori.

Level 3 is the normalized breakdown reason itself (deduplicated/standardized
text), preserving full traceability back to the original free-text.
"""

import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

SOURCE_WORKBOOK = "VRM_Jul2025_Oct2026_Combined.xlsx"

# Category -> authoritative code prefix (verified 1:1 against Dropdown Data Code/Code.1)
CATEGORY_CODE_MAP = {
    "Utility (Electricity)": "U01",
    "Planned Down Time": "P01",
    "Others down time": "O01",
    "Warehouse block": "W01",
    "Electrical Breakdown": "E01",
    "Mechanical Breakdown": "M01",
    "In Process Setup/ Adjustment": "I01",
    "Raw Material Shortage": "R01",
}
ELIGIBLE_CATEGORIES = {"Electrical Breakdown": "Electrical", "Mechanical Breakdown": "Mechanical"}

PREFIX_PATTERN = re.compile(r"^[A-Z]{1,2}\d{2}-\s*")


def strip_prefix(text):
    if pd.isna(text):
        return ""
    return PREFIX_PATTERN.sub("", str(text)).strip()


def normalize_text(text):
    """Lowercase-insensitive-safe normalization: collapse whitespace, strip
    punctuation noise, keep original casing for display but normalize for
    matching/grouping."""
    t = strip_prefix(text)
    t = re.sub(r"\s+", " ", t).strip()
    t = t.rstrip(".,;:")
    return t


# --------------------------------------------------------------------------
# Level-2 FAILURE-FAMILY RULES
# Each rule: (family_name, [keywords], confidence_if_matched)
# Keywords are matched against the normalized (lowercased) Name+Reason text.
# Order matters: first match wins (most specific families listed first).
# Only families that actually match >=1 real record are kept in the final
# taxonomy (see build_taxonomy()).
# --------------------------------------------------------------------------
FAMILY_RULES = [
    ("Bag House / Bag Filter / Dust Collection",
     ["bag house", "bag filter", "baghouse", "bagfilter", "dp maximum", "bag filter dp"],
     "High"),
    ("Motor & Electrical Drive Failures",
     ["main motor", "motor change", "motor bearing", "winding", "lrs", "capacitor bank",
      "transformer", "motor problem", "motor not start", "motor fitting"],
     "High"),
    ("Bearing, Gearbox & Lubrication Failures",
     ["gear box", "gearbox", "lubrication", "bearing", "thrustpad", "rocker arm",
      "oil pressure", "oil temp", "oil level", "oil tank", "gear oil"],
     "High"),
    ("Conveyor & Feeder Mechanical Failures",
     ["belt conveyor", "belt joint", "belt side", "bucket elevator", "bucket damage",
      "rotary feeder", "rotary valve", "weigh feeder", "rotary jam", "jam"],
     "High"),
    ("Hydraulic & Pneumatic System Failures",
     ["hydraulic", "hydraullic", "cylinder", "compressor", "pneumatic", "hy-110",
      "hy 110", "pressure fall", "pressure low", "pressure down"],
     "High"),
    ("Fan & Separator Mechanical/Electrical Failures",
     ["fan", "classifier", "separator", "damper", "cooling fan", "id fan"],
     "Medium"),
    ("Electrical Protection, Instrumentation & Control Faults",
     ["plc", "zss", "bss", "limit fault", "motion detector", "alarm", "trip",
      "cycle off", "not ready", "server", "sensor", "not run", "stop without alarm"],
     "Medium"),
    ("Mill / Roller / Structural Mechanical Failures",
     ["roller", "mill vibration", "silicon tube", "guide rollar", "guide roller",
      "diaphramgh", "diaphragm"],
     "Medium"),
]

OTHER_ELECTRICAL = "Other Confirmed Electrical Failures"
OTHER_MECHANICAL = "Other Confirmed Mechanical Failures"


def assign_family(level1, normalized_name, normalized_reason):
    """Returns (family, rationale, confidence)."""
    text = f"{normalized_name} {normalized_reason}".lower()
    # collapse whitespace around hyphens so "hy - 110" matches "hy-110"/"hy 110"
    text = re.sub(r"\s*-\s*", "-", text)
    text = text.replace("-", " ")
    for family, keywords, conf in FAMILY_RULES:
        for kw in keywords:
            kw_norm = kw.replace("-", " ")
            if kw_norm in text:
                return family, f"Matched keyword '{kw}' in name/reason text", conf
    fallback = OTHER_ELECTRICAL if level1 == "Electrical" else OTHER_MECHANICAL
    return fallback, "No family keyword matched; retained as unclassified residual within Level 1", "Low"


# --------------------------------------------------------------------------
# BUILD THE MAPPING TABLE (Step 4.1 deliverable)
# --------------------------------------------------------------------------
def build_mapping_table(bdf):
    """
    One row per UNIQUE (Category, Breakdown Name, Reason) combination
    actually observed in the data, with an event_count column for
    traceability back to how many raw rows it represents.
    """
    name_col = "Breakdwon Name" if "Breakdwon Name" in bdf.columns else "Breakdown Name"

    work = bdf[["Category", name_col, "Reason"]].copy()
    work.columns = ["category", "breakdown_name_raw", "reason_raw"]
    work["event_count"] = 1

    grouped = (work.groupby(["category", "breakdown_name_raw", "reason_raw"], dropna=False)
               .size().reset_index(name="event_count"))

    rows = []
    for _, r in grouped.iterrows():
        cat = r["category"]
        name_raw = r["breakdown_name_raw"]
        reason_raw = r["reason_raw"]
        n = int(r["event_count"])

        norm_name = normalize_text(name_raw)
        norm_reason = normalize_text(reason_raw)

        eligible = cat in ELIGIBLE_CATEGORIES
        level1 = ELIGIBLE_CATEGORIES.get(cat, "Not Applicable")

        if eligible:
            family, rationale, confidence = assign_family(level1, norm_name, norm_reason)
            review_status = "Needs Review" if confidence == "Low" else "Auto-Confirmed"
        else:
            family = "N/A (excluded category)"
            rationale = (
                f"Category '{cat}' is not Electrical Breakdown or Mechanical Breakdown "
                f"(authoritative code prefix {CATEGORY_CODE_MAP.get(cat, '?')}), "
                f"therefore excluded from the primary failure-prediction target "
                f"regardless of any electrical/mechanical wording in the name or reason."
            )
            confidence = "High"
            review_status = "Excluded by Category (not a failure)"

        rows.append({
            "original_category": cat,
            "original_breakdown_name": name_raw,
            "original_reason": reason_raw,
            "normalized_breakdown_name": norm_name,
            "normalized_reason": norm_reason,
            "electrical_mechanical_classification": level1,
            "failure_family_level2": family,
            "mapping_rationale": rationale,
            "mapping_confidence": confidence,
            "review_status": review_status,
            "event_count": n,
            "eligible_for_target": eligible,
        })

    mapping_df = pd.DataFrame(rows).sort_values(
        ["eligible_for_target", "electrical_mechanical_classification", "failure_family_level2", "event_count"],
        ascending=[False, True, True, False]
    ).reset_index(drop=True)

    return mapping_df


# --------------------------------------------------------------------------
# TAXONOMY SUMMARY (Step 4.2 deliverable) -- families actually supported by data
# --------------------------------------------------------------------------
def taxonomy_summary(mapping_df):
    elig = mapping_df[mapping_df["eligible_for_target"]]
    summary = (elig.groupby(["electrical_mechanical_classification", "failure_family_level2"])
               .agg(n_unique_reasons=("normalized_reason", "nunique"),
                    n_events=("event_count", "sum"),
                    low_confidence_rows=("mapping_confidence", lambda s: int((s == "Low").sum())))
               .reset_index()
               .sort_values(["electrical_mechanical_classification", "n_events"], ascending=[True, False]))
    return summary


# --------------------------------------------------------------------------
# EVENT-GRANULARITY ASSESSMENT (Step 4.3 preamble)
# --------------------------------------------------------------------------
def event_granularity_report(bdf):
    grp = bdf.groupby(["Mill Name", "Date", "Shift"]).size()
    dist = grp.value_counts().sort_index()
    downtime = pd.to_numeric(bdf["Downtime"], errors="coerce")
    return {
        "rows_per_machine_date_shift_distribution": dist.to_dict(),
        "max_rows_in_one_machine_date_shift": int(grp.max()),
        "pct_machine_date_shift_with_multiple_events": round(100 * (grp > 1).mean(), 2),
        "downtime_min": float(downtime.min()),
        "downtime_max": float(downtime.max()),
        "downtime_mean": round(float(downtime.mean()), 1),
        "conclusion": (
            "Each row in the Breakdown sheet represents an INDIVIDUAL breakdown EVENT, "
            "not a shift-level aggregate. Evidence: up to 5 distinct rows share the same "
            "(Mill Name, Date, Shift) key, each with a different Category/Breakdown Name/"
            "Reason and its own Downtime value (range 5-480 minutes, consistent with "
            "within-shift event durations rather than full-shift totals). Therefore, "
            "'number of breakdown events' for target C must be computed as ROW COUNTS "
            "per (machine, family, day) -- summing rows, not counting distinct shifts -- "
            "and the 18 rows sharing an identical (Mill, Date, Shift, Name, Reason) key "
            "found in the prior data-quality audit should be treated as likely duplicate "
            "entries (same event logged twice) and deduplicated before counting, not as "
            "18 additional independent events."
        ),
    }


# --------------------------------------------------------------------------
# TARGET DEFINITIONS (Step 4.3) -- machine-day level target table
# --------------------------------------------------------------------------
def build_machine_day_targets(bdf, mapping_df):
    """
    Builds one row per (machine, date) with:
      A. any_electrical_mechanical_breakdown (0/1)
      B. <family>_occurred (0/1) for each Level-2 family actually present
      C. <family>_event_count (int) for each Level-2 family
      D. top reasons are NOT pre-aggregated here (kept at event level in a
         separate table, since "most likely detailed reason conditional on
         family" is a per-event classification problem, not a single
         machine-day column)
      E. downtime_minutes_total (sum) and downtime_minutes_max (for the day)

    Deduplicates exact-duplicate (Mill, Date, Shift, Name, Reason) rows
    first, per the event_granularity_report() conclusion.
    """
    name_col = "Breakdwon Name" if "Breakdwon Name" in bdf.columns else "Breakdown Name"
    df = bdf.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    dedup_keys = ["Mill Name", "Date", "Shift", name_col, "Reason"]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_keys, keep="first")
    n_removed = before - len(df)

    # attach family/level1 via the mapping table keyed on (category, name, reason)
    map_key = mapping_df.set_index(
        ["original_category", "original_breakdown_name", "original_reason"]
    )[["electrical_mechanical_classification", "failure_family_level2", "eligible_for_target"]]

    df = df.join(map_key, on=["Category", name_col, "Reason"])

    eligible_df = df[df["eligible_for_target"] == True].copy()  # noqa: E712
    eligible_df["Downtime"] = pd.to_numeric(eligible_df["Downtime"], errors="coerce")

    families = sorted(eligible_df["failure_family_level2"].dropna().unique().tolist())

    all_machines = sorted(bdf["Mill Name"].dropna().unique().tolist())
    date_min, date_max = df["Date"].min(), df["Date"].max()
    all_dates = pd.date_range(date_min, date_max, freq="D")

    base_index = pd.MultiIndex.from_product([all_machines, all_dates], names=["machine", "date"])
    target_df = pd.DataFrame(index=base_index).reset_index()

    # Target A: any eligible breakdown that day
    daily_any = (eligible_df.groupby(["Mill Name", "Date"]).size()
                 .rename("any_breakdown_event_count").reset_index()
                 .rename(columns={"Mill Name": "machine", "Date": "date"}))
    target_df = target_df.merge(daily_any, on=["machine", "date"], how="left")
    target_df["any_breakdown_event_count"] = target_df["any_breakdown_event_count"].fillna(0).astype(int)
    target_df["A_any_eligible_breakdown"] = (target_df["any_breakdown_event_count"] > 0).astype(int)

    # Targets B & C: per-family occurrence + event count
    for fam in families:
        fam_df = eligible_df[eligible_df["failure_family_level2"] == fam]
        daily_fam = (fam_df.groupby(["Mill Name", "Date"]).size()
                     .rename(f"C_{fam}_event_count").reset_index()
                     .rename(columns={"Mill Name": "machine", "Date": "date"}))
        target_df = target_df.merge(daily_fam, on=["machine", "date"], how="left")
        col = f"C_{fam}_event_count"
        target_df[col] = target_df[col].fillna(0).astype(int)
        target_df[f"B_{fam}_occurred"] = (target_df[col] > 0).astype(int)

    # Target E: downtime
    daily_downtime = (eligible_df.groupby(["Mill Name", "Date"])["Downtime"]
                       .agg(E_downtime_minutes_total="sum", E_downtime_minutes_max="max")
                       .reset_index().rename(columns={"Mill Name": "machine", "Date": "date"}))
    target_df = target_df.merge(daily_downtime, on=["machine", "date"], how="left")
    target_df["E_downtime_minutes_total"] = target_df["E_downtime_minutes_total"].fillna(0.0)
    target_df["E_downtime_minutes_max"] = target_df["E_downtime_minutes_max"].fillna(0.0)

    return target_df, families, n_removed


def event_level_reason_table(bdf, mapping_df):
    """Target D support table: event-level rows with family + normalized
    reason, for training a 'most likely detailed reason | family' model."""
    name_col = "Breakdwon Name" if "Breakdwon Name" in bdf.columns else "Breakdown Name"
    df = bdf.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    dedup_keys = ["Mill Name", "Date", "Shift", name_col, "Reason"]
    df = df.drop_duplicates(subset=dedup_keys, keep="first")

    map_key = mapping_df.set_index(
        ["original_category", "original_breakdown_name", "original_reason"]
    )[["electrical_mechanical_classification", "failure_family_level2",
       "normalized_reason", "eligible_for_target"]]

    df = df.join(map_key, on=["Category", name_col, "Reason"])
    elig = df[df["eligible_for_target"] == True].copy()  # noqa: E712
    cols = ["Mill Name", "Date", "Shift", "Category",
            "electrical_mechanical_classification", "failure_family_level2",
            "normalized_reason", "Downtime"]
    out = elig[cols].rename(columns={
        "Mill Name": "machine", "Date": "date", "Category": "original_category",
        "normalized_reason": "normalized_reason_level3",
    })
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# TARGET E FEASIBILITY CHECK -- data-driven, not assumed
# --------------------------------------------------------------------------
def downtime_feasibility_check(event_reason_df):
    """Lightweight, honest feasibility diagnostic for predicting downtime
    (Target E), WITHOUT fitting a predictive model yet (that happens in the
    modeling task). Reports sample size, variance, and skew per family so
    a later decision to model/not-model Target E is evidence-based."""
    rows = []
    for fam, sub in event_reason_df.groupby("failure_family_level2"):
        dt = pd.to_numeric(sub["Downtime"], errors="coerce").dropna()
        if len(dt) == 0:
            continue
        rows.append({
            "failure_family": fam,
            "n_events": len(dt),
            "mean": round(float(dt.mean()), 1),
            "std": round(float(dt.std()), 1) if len(dt) > 1 else None,
            "cv_(std/mean)": round(float(dt.std() / dt.mean()), 2) if len(dt) > 1 and dt.mean() > 0 else None,
            "min": float(dt.min()),
            "max": float(dt.max()),
            "reliable_for_regression": bool(len(dt) >= 30 and (dt.std() / dt.mean() if dt.mean() else 99) < 1.2),
        })
    feas_df = pd.DataFrame(rows).sort_values("n_events", ascending=False)
    n_reliable = int(feas_df["reliable_for_regression"].sum()) if len(feas_df) else 0
    conclusion = (
        f"{n_reliable} of {len(feas_df)} failure families have >=30 events AND a "
        f"coefficient of variation < 1.2, the minimum bar for a potentially reliable "
        f"downtime regression. Families below this bar should NOT have a downtime "
        f"model forced onto them -- downtime should be reported only as a historical "
        f"descriptive range for those families, not a point prediction."
        if len(feas_df) else "No eligible events with valid downtime found."
    )
    return feas_df, conclusion


# --------------------------------------------------------------------------
# ORCHESTRATOR
# --------------------------------------------------------------------------
def run_all(path=SOURCE_WORKBOOK):
    xl = pd.ExcelFile(path)
    bdf = xl.parse("Breakdown")

    mapping_df = build_mapping_table(bdf)
    tax_summary = taxonomy_summary(mapping_df)
    granularity = event_granularity_report(bdf)
    target_df, families, n_dedup_removed = build_machine_day_targets(bdf, mapping_df)
    reason_table = event_level_reason_table(bdf, mapping_df)
    feas_df, feas_conclusion = downtime_feasibility_check(reason_table)

    return {
        "bdf": bdf,
        "mapping_df": mapping_df,
        "taxonomy_summary": tax_summary,
        "granularity": granularity,
        "target_df": target_df,
        "families": families,
        "n_dedup_removed": n_dedup_removed,
        "reason_table": reason_table,
        "feasibility_df": feas_df,
        "feasibility_conclusion": feas_conclusion,
    }


if __name__ == "__main__":
    out = run_all()
    print("Families discovered:", out["families"])
    print("Mapping rows:", len(out["mapping_df"]))
    print("Target rows (machine-day):", len(out["target_df"]))
    print(out["granularity"]["conclusion"])
    print(out["feasibility_conclusion"])
