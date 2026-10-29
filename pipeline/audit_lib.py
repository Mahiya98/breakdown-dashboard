"""
audit_lib.py
=============
Reusable, read-only data-audit functions for the VRM predictive-maintenance
workbook. Imported by 01_data_audit.ipynb and run_audit.py.

This module NEVER writes to the source workbook. It only reads it with
pandas.ExcelFile / pd.read_excel and returns/derives new objects in memory
or writes separate output files (xlsx/md/json) under ./audit_v2_output/.

No records, labels, or statistics in this module are invented. Every
number returned comes directly from an operation on the loaded DataFrame.
"""

import re
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

SOURCE_WORKBOOK = "VRM_Jul2025_Oct2026_Combined.xlsx"
SHEET_NAMES = ["VRM 1", "VRM 2", "Production", "Breakdown", "Dropdown Data"]

# Columns that are clearly identifiers / free text / metadata, not sensor
# measurements -- used to separate "machine parameters" from context columns.
NON_PARAMETER_COLS = {
    "Timestamp", "Date", "Hour", "Shift", "Operation Engineer", "Remarks",
    "Source file", "Operation Mode",
}

# Best-effort unit extraction: VRM1/VRM2 columns encode units in the column
# name itself, e.g. "Main Motor Power | KW", "Mill Vibration AVG. | mm/sec".
UNIT_PATTERN = re.compile(r"\|\s*([A-Za-z0-9/%°.\s]+)$")


# --------------------------------------------------------------------------
def load_workbook(path=SOURCE_WORKBOOK):
    """Read-only load. Returns dict[sheet_name] -> DataFrame (copies)."""
    xl = pd.ExcelFile(path)
    missing = set(SHEET_NAMES) - set(xl.sheet_names)
    if missing:
        raise ValueError(f"Missing expected sheets: {missing}. Found: {xl.sheet_names}")
    return {name: xl.parse(name) for name in SHEET_NAMES}, xl.sheet_names


# --------------------------------------------------------------------------
# 1. WORKBOOK / SHEET INVENTORY
# --------------------------------------------------------------------------
def sheet_inventory(sheets):
    rows = []
    for name, df in sheets.items():
        rows.append({
            "sheet": name,
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. DATA DICTIONARY (every column, every sheet)
# --------------------------------------------------------------------------
def extract_unit(col_name):
    m = UNIT_PATTERN.search(str(col_name))
    if m:
        return m.group(1).strip()
    return ""


def data_dictionary(sheets):
    rows = []
    for sheet_name, df in sheets.items():
        for col in df.columns:
            s = df[col]
            is_param = (sheet_name in ("VRM 1", "VRM 2")) and (col not in NON_PARAMETER_COLS)
            rows.append({
                "sheet": sheet_name,
                "column": col,
                "dtype": str(s.dtype),
                "is_machine_parameter": is_param,
                "unit_(parsed_from_column_name)": extract_unit(col) if is_param else "",
                "n_rows": len(s),
                "n_non_null": int(s.notna().sum()),
                "n_missing": int(s.isna().sum()),
                "missing_pct": round(float(s.isna().mean() * 100), 2),
                "n_unique": int(s.nunique(dropna=True)),
                "sample_values": "; ".join(str(v) for v in s.dropna().unique()[:3]),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. MACHINE PARAMETER LIST + UNITS (VRM 1 / VRM 2 only)
# --------------------------------------------------------------------------
def machine_parameter_list(sheets):
    out = {}
    for mname in ("VRM 1", "VRM 2"):
        df = sheets[mname]
        params = [c for c in df.columns if c not in NON_PARAMETER_COLS]
        out[mname] = pd.DataFrame({
            "parameter": params,
            "unit": [extract_unit(c) for c in params],
            "dtype": [str(df[c].dtype) for c in params],
            "missing_pct": [round(float(df[c].isna().mean() * 100), 2) for c in params],
        })
    return out


# --------------------------------------------------------------------------
# 4. RECORD COUNTS + DATE COVERAGE PER SHEET
# --------------------------------------------------------------------------
def date_coverage(sheets):
    rows = []
    for name, df in sheets.items():
        date_col = "Date" if "Date" in df.columns else ("Timestamp" if "Timestamp" in df.columns else None)
        rec = {"sheet": name, "n_records": len(df), "date_column": date_col}
        if date_col:
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            valid = parsed.dropna()
            rec["date_min"] = str(valid.min()) if len(valid) else None
            rec["date_max"] = str(valid.max()) if len(valid) else None
            rec["n_distinct_dates"] = int(valid.dt.date.nunique()) if len(valid) else 0
            rec["n_date_missing"] = int(df[date_col].isna().sum())
            rec["n_date_unparseable"] = int(parsed.isna().sum() - df[date_col].isna().sum())
            full_range = pd.date_range(valid.min(), valid.max(), freq="D") if len(valid) else []
            present_dates = set(valid.dt.date)
            gap_dates = [d.date() for d in full_range if d.date() not in present_dates]
            rec["n_calendar_days_in_range"] = len(full_range)
            rec["n_days_with_zero_records"] = len(gap_dates)
        else:
            rec.update({"date_min": None, "date_max": None, "n_distinct_dates": None,
                        "n_date_missing": None, "n_date_unparseable": None,
                        "n_calendar_days_in_range": None, "n_days_with_zero_records": None})
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. MISSING VALUE PERCENTAGES (summary per sheet)
# --------------------------------------------------------------------------
def missingness_summary(sheets):
    rows = []
    for name, df in sheets.items():
        miss = df.isna().mean() * 100
        rows.append({
            "sheet": name,
            "n_cols": df.shape[1],
            "mean_missing_pct": round(float(miss.mean()), 2),
            "max_missing_pct": round(float(miss.max()), 2),
            "col_with_max_missing": miss.idxmax(),
            "n_cols_fully_populated": int((miss == 0).sum()),
            "n_cols_over_50pct_missing": int((miss > 50).sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 6. DUPLICATE RECORDS + INCONSISTENT TIMESTAMPS
# --------------------------------------------------------------------------
def duplicate_and_timestamp_audit(sheets):
    results = {}

    # VRM1/VRM2: duplicate on (Date, Hour, Shift) -- the natural reading key
    for mname in ("VRM 1", "VRM 2"):
        df = sheets[mname]
        key_cols = [c for c in ["Date", "Hour", "Shift"] if c in df.columns]
        full_dupes = int(df.duplicated().sum())
        key_dupes = int(df.duplicated(subset=key_cols).sum()) if key_cols else None
        bad_ts = None
        if "Timestamp" in df.columns:
            ts = pd.to_datetime(df["Timestamp"], errors="coerce")
            bad_ts = int(ts.isna().sum() - df["Timestamp"].isna().sum())
        results[mname] = {
            "full_row_duplicates": full_dupes,
            "duplicate_key_cols": key_cols,
            "duplicates_on_key": key_dupes,
            "unparseable_timestamps": bad_ts,
        }

    # Production: duplicate on (Mill Name, Date, Shift)
    pdf = sheets["Production"]
    key_cols = [c for c in ["Mill Name", "Date", "Shift"] if c in pdf.columns]
    results["Production"] = {
        "full_row_duplicates": int(pdf.duplicated().sum()),
        "duplicate_key_cols": key_cols,
        "duplicates_on_key": int(pdf.duplicated(subset=key_cols).sum()) if key_cols else None,
    }

    # Breakdown: duplicate on (Mill Name, Date, Shift, Breakdwon Name, Reason)
    bdf = sheets["Breakdown"]
    name_col = "Breakdwon Name" if "Breakdwon Name" in bdf.columns else "Breakdown Name"
    key_cols = [c for c in ["Mill Name", "Date", "Shift", name_col, "Reason"] if c in bdf.columns]
    results["Breakdown"] = {
        "full_row_duplicates": int(bdf.duplicated().sum()),
        "duplicate_key_cols": key_cols,
        "duplicates_on_key": int(bdf.duplicated(subset=key_cols).sum()) if key_cols else None,
    }

    return results


# --------------------------------------------------------------------------
# 7. BREAKDOWN CATEGORY / REASON FREQUENCIES
# --------------------------------------------------------------------------
def breakdown_frequencies(sheets):
    bdf = sheets["Breakdown"]
    name_col = "Breakdwon Name" if "Breakdwon Name" in bdf.columns else "Breakdown Name"
    out = {}
    if "Category" in bdf.columns:
        out["category_counts"] = bdf["Category"].value_counts(dropna=False).to_dict()
    if name_col in bdf.columns:
        out["breakdown_name_counts"] = bdf[name_col].value_counts(dropna=False).to_dict()
    if "Reason" in bdf.columns:
        out["reason_counts"] = bdf["Reason"].value_counts(dropna=False).to_dict()
    if "Mill Name" in bdf.columns:
        out["mill_name_counts"] = bdf["Mill Name"].value_counts(dropna=False).to_dict()
    return out


# --------------------------------------------------------------------------
# 8. CROSS-REFERENCE: Breakdown <-> Dropdown Data
# --------------------------------------------------------------------------
def crossref_breakdown_dropdown(sheets):
    bdf = sheets["Breakdown"]
    ddf = sheets["Dropdown Data"]
    name_col = "Breakdwon Name" if "Breakdwon Name" in bdf.columns else "Breakdown Name"

    # Dropdown Data has two logical blocks: Breakdown Dropdown (col) and Reason Dropdown (col)
    valid_breakdown_names = set()
    if "Breakdown Dropdown" in ddf.columns:
        valid_breakdown_names = set(ddf["Breakdown Dropdown"].dropna().astype(str).str.strip())
    elif "Breakdown" in ddf.columns:
        valid_breakdown_names = set(ddf["Breakdown"].dropna().astype(str).str.strip())

    valid_reasons = set()
    if "Reason Dropdown" in ddf.columns:
        valid_reasons = set(ddf["Reason Dropdown"].dropna().astype(str).str.strip())
    elif "Reason" in ddf.columns:
        valid_reasons = set(ddf["Reason"].dropna().astype(str).str.strip())

    actual_names = set(bdf[name_col].dropna().astype(str).str.strip()) if name_col in bdf.columns else set()
    actual_reasons = set(bdf["Reason"].dropna().astype(str).str.strip()) if "Reason" in bdf.columns else set()

    unmapped_names = sorted(actual_names - valid_breakdown_names)
    unmapped_reasons = sorted(actual_reasons - valid_reasons)

    return {
        "n_valid_dropdown_breakdown_names": len(valid_breakdown_names),
        "n_valid_dropdown_reasons": len(valid_reasons),
        "n_actual_breakdown_names_used": len(actual_names),
        "n_actual_reasons_used": len(actual_reasons),
        "n_breakdown_names_not_in_dropdown": len(unmapped_names),
        "breakdown_names_not_in_dropdown": unmapped_names,
        "n_reasons_not_in_dropdown": len(unmapped_reasons),
        "reasons_not_in_dropdown": unmapped_reasons,
    }


# --------------------------------------------------------------------------
# 9. INVALID / UNMAPPED / INCONSISTENT BREAKDOWN NAME REPORT
# --------------------------------------------------------------------------
def inconsistent_breakdown_names(sheets):
    bdf = sheets["Breakdown"]
    name_col = "Breakdwon Name" if "Breakdwon Name" in bdf.columns else "Breakdown Name"
    if name_col not in bdf.columns:
        return {}

    raw = bdf[name_col].dropna().astype(str)
    trimmed_diff = raw[raw != raw.str.strip()]
    case_groups = {}
    for v in raw.unique():
        key = re.sub(r"\s+", " ", v.strip().lower())
        case_groups.setdefault(key, set()).add(v)
    inconsistent_casing = {k: sorted(v) for k, v in case_groups.items() if len(v) > 1}

    empty_or_placeholder = raw[raw.str.strip().isin(["", "-", "N/A", "NA", "nil", "Nil", "none", "None"])]

    return {
        "n_total_nonnull_names": int(len(raw)),
        "n_names_with_leading_trailing_whitespace": int(len(trimmed_diff)),
        "n_distinct_normalized_names_with_casing_variants": len(inconsistent_casing),
        "casing_variant_examples": {k: v for k, v in list(inconsistent_casing.items())[:15]},
        "n_placeholder_or_empty_like_values": int(len(empty_or_placeholder)),
        "placeholder_values_found": sorted(empty_or_placeholder.unique().tolist()),
    }


# --------------------------------------------------------------------------
# 10. LINKABILITY ASSESSMENT: Breakdown <-> VRM params <-> Production
# --------------------------------------------------------------------------
def normalize_machine_id(raw):
    if pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    s_compact = re.sub(r"[\s\-_]+", "", s)
    m = re.search(r"(vrm|mill)\D{0,3}([12])", s_compact)
    if m:
        return f"VRM-{m.group(2)}"
    return str(raw).strip()


def linkability_assessment(sheets):
    bdf = sheets["Breakdown"].copy()
    pdf = sheets["Production"].copy()
    v1 = sheets["VRM 1"].copy()
    v2 = sheets["VRM 2"].copy()

    result = {}

    # Breakdown -> Production link via (machine, date, shift)
    if {"Mill Name", "Date", "Shift"}.issubset(bdf.columns) and {"Mill Name", "Date", "Shift"}.issubset(pdf.columns):
        bdf["_m"] = bdf["Mill Name"].apply(normalize_machine_id)
        pdf["_m"] = pdf["Mill Name"].apply(normalize_machine_id)
        b_keys = set(zip(bdf["_m"], pd.to_datetime(bdf["Date"], errors="coerce").dt.date, bdf["Shift"]))
        p_keys = set(zip(pdf["_m"], pd.to_datetime(pdf["Date"], errors="coerce").dt.date, pdf["Shift"]))
        matched = b_keys & p_keys
        result["breakdown_to_production_link"] = {
            "n_breakdown_keys": len(b_keys),
            "n_production_keys": len(p_keys),
            "n_matched_keys": len(matched),
            "match_rate_pct": round(100 * len(matched) / max(1, len(b_keys)), 1),
        }

    # Breakdown -> VRM parameter sheet link via (machine, date) -- do readings exist on/around breakdown date?
    for mname, pdf_params, canon in [("VRM 1", v1, "VRM-1"), ("VRM 2", v2, "VRM-2")]:
        if "Date" in pdf_params.columns:
            param_dates = set(pd.to_datetime(pdf_params["Date"], errors="coerce").dropna().dt.date)
            sub_bdf = bdf[bdf["_m"] == canon] if "_m" in bdf.columns else pd.DataFrame()
            if len(sub_bdf):
                bdown_dates = set(pd.to_datetime(sub_bdf["Date"], errors="coerce").dropna().dt.date)
                has_same_day_reading = sum(1 for d in bdown_dates if d in param_dates)
                has_prior_day_reading = sum(
                    1 for d in bdown_dates if any((d - pd.Timedelta(days=k)) in param_dates for k in (1, 2, 3))
                )
                result[f"{mname}_breakdown_to_param_link"] = {
                    "n_breakdown_dates": len(bdown_dates),
                    "n_with_same_day_param_reading": has_same_day_reading,
                    "n_with_param_reading_in_prior_3_days": has_prior_day_reading,
                    "same_day_coverage_pct": round(100 * has_same_day_reading / max(1, len(bdown_dates)), 1),
                    "prior_3day_coverage_pct": round(100 * has_prior_day_reading / max(1, len(bdown_dates)), 1),
                }

    return result


# --------------------------------------------------------------------------
# 11. DATA LIMITATIONS (derived, not assumed)
# --------------------------------------------------------------------------
def derive_limitations(sheets, results_bundle):
    limitations = []

    miss = results_bundle["missingness_summary"]
    for _, row in miss.iterrows():
        if row["max_missing_pct"] > 50:
            limitations.append(
                f"Sheet '{row['sheet']}' has a column ('{row['col_with_max_missing']}') "
                f"with {row['max_missing_pct']}% missing values, which may limit its use as a feature."
            )

    dup = results_bundle["duplicate_and_timestamp_audit"]
    for sheet, info in dup.items():
        dk = info.get("duplicates_on_key")
        if dk:
            limitations.append(
                f"'{sheet}' has {dk} rows sharing the same key "
                f"({info.get('duplicate_key_cols')}), which risks double-counting if joined naively."
            )

    cross = results_bundle["crossref_breakdown_dropdown"]
    if cross.get("n_breakdown_names_not_in_dropdown", 0) > 0:
        limitations.append(
            f"{cross['n_breakdown_names_not_in_dropdown']} distinct breakdown names used in the "
            f"Breakdown sheet do not appear in the Dropdown Data reference list, indicating free-text "
            f"entry or drift from the standard taxonomy."
        )

    link = results_bundle["linkability_assessment"]
    for key in ("VRM 1_breakdown_to_param_link", "VRM 2_breakdown_to_param_link"):
        if key in link and link[key]["prior_3day_coverage_pct"] < 90:
            limitations.append(
                f"For {key.split('_')[0]} {key.split('_')[1]}, only {link[key]['prior_3day_coverage_pct']}% "
                f"of breakdown events have a machine-parameter reading in the 3 days prior, which limits "
                f"the feature signal available for prediction on the remainder."
            )

    cov = results_bundle["date_coverage"]
    for _, row in cov.iterrows():
        if row["n_days_with_zero_records"] and row["n_days_with_zero_records"] > 0:
            limitations.append(
                f"Sheet '{row['sheet']}' has {row['n_days_with_zero_records']} calendar days within its "
                f"date range with zero records, indicating gaps in reporting/logging continuity."
            )

    if not limitations:
        limitations.append("No major structural data-quality limitations were detected by the automated checks.")

    return limitations


# --------------------------------------------------------------------------
# ORCHESTRATOR
# --------------------------------------------------------------------------
def run_full_audit(path=SOURCE_WORKBOOK):
    sheets, sheet_names = load_workbook(path)

    results = {}
    results["sheet_inventory"] = sheet_inventory(sheets)
    results["data_dictionary"] = data_dictionary(sheets)
    results["machine_parameters"] = machine_parameter_list(sheets)
    results["date_coverage"] = date_coverage(sheets)
    results["missingness_summary"] = missingness_summary(sheets)
    results["duplicate_and_timestamp_audit"] = duplicate_and_timestamp_audit(sheets)
    results["breakdown_frequencies"] = breakdown_frequencies(sheets)
    results["crossref_breakdown_dropdown"] = crossref_breakdown_dropdown(sheets)
    results["inconsistent_breakdown_names"] = inconsistent_breakdown_names(sheets)
    results["linkability_assessment"] = linkability_assessment(sheets)
    results["limitations"] = derive_limitations(sheets, results)

    return sheets, results
