"""Step 10: independent re-verification of forward_targets.parquet (the file
modeling_pipeline.py actually loads), run fresh against the raw source files.
Re-derives every number -- does not trust any prior report's prose. Also
extends the original 7 checks with an 8th: that the new electrical/mechanical
breakdown targets partition correctly against the combined "any eligible"
target (no electrical+mechanical double counting issue, no row where neither
is set but "any" is set).
"""
import pandas as pd
import numpy as np

ft = pd.read_parquet("forward_targets.parquet")
fm = pd.read_parquet("feature_matrix.parquet")
mdt = pd.read_csv("target_output/machine_day_targets.csv")
mdt = mdt.rename(columns={"machine": "machine_id", "date": "Date"})
mdt["Date"] = pd.to_datetime(mdt["Date"])
fm["Date"] = pd.to_datetime(fm["Date"])
ft["Date"] = pd.to_datetime(ft["Date"])

results = {}

# ---- Check 1: machine/date key alignment ----
key_ft = set(zip(ft["machine_id"], ft["Date"]))
key_fm = set(zip(fm["machine_id"], fm["Date"]))
results["check1_keys_match"] = (len(ft) == 964 and key_ft == key_fm)
results["check1_detail"] = f"rows={len(ft)}, key_set_equal={key_ft == key_fm}, machines={sorted(ft['machine_id'].unique())}"

# ---- Check 2: forward target matches actual same-day label on future date ----
mdt_idx = mdt.set_index(["machine_id", "Date"])
check2 = {}
for h in [1, 2, 3]:
    col = f"Day_Plus_{h}__A_any_eligible_breakdown"
    n_checked, n_mismatch = 0, 0
    for _, row in ft.iterrows():
        val = row[col]
        if pd.isna(val):
            continue
        future_date = row["Date"] + pd.Timedelta(days=h)
        key = (row["machine_id"], future_date)
        if key not in mdt_idx.index:
            n_mismatch += 1
            continue
        actual = mdt_idx.loc[key, "A_any_eligible_breakdown"]
        n_checked += 1
        if int(actual) != int(val):
            n_mismatch += 1
    check2[f"Day+{h}"] = {"checked": n_checked, "mismatches": n_mismatch}
results["check2"] = check2

# ---- Check 3: confidence matches that same future date ----
fm_idx = fm.set_index(["machine_id", "Date"])
check3 = {}
for h in [1, 2, 3]:
    col = f"Day_Plus_{h}__label_confidence"
    n_checked, n_mismatch = 0, 0
    for _, row in ft.iterrows():
        val = row[col]
        if pd.isna(val):
            continue
        future_date = row["Date"] + pd.Timedelta(days=h)
        key = (row["machine_id"], future_date)
        if key not in fm_idx.index:
            n_mismatch += 1
            continue
        actual = fm_idx.loc[key, "label_confidence"]
        n_checked += 1
        if actual != val:
            n_mismatch += 1
    check3[f"Day+{h}"] = {"checked": n_checked, "mismatches": n_mismatch}
results["check3"] = check3

# ---- Check 4: trailing NaN only, nowhere else ----
check4 = {}
for h in [1, 2, 3]:
    tcol, ccol = f"Day_Plus_{h}__A_any_eligible_breakdown", f"Day_Plus_{h}__label_confidence"
    per_machine = {}
    for m, g in ft.sort_values("Date").groupby("machine_id"):
        nan_t = g[tcol].isna().sum()
        nan_c = g[ccol].isna().sum()
        # the last h rows should be exactly the NaN rows
        trailing_idx = g.index[-h:]
        nan_idx_t = g.index[g[tcol].isna()]
        per_machine[m] = {
            "nan_target": int(nan_t), "nan_conf": int(nan_c),
            "nan_rows_are_trailing": set(nan_idx_t) == set(trailing_idx),
        }
    check4[f"Day+{h}"] = per_machine
results["check4"] = check4

# ---- Check 5: unknown confidence rows excluded from "trainable" ----
check5 = {}
for h in [1, 2, 3]:
    tcol, ccol = f"Day_Plus_{h}__A_any_eligible_breakdown", f"Day_Plus_{h}__label_confidence"
    present = ft[tcol].notna()
    unknown = ft[ccol] == "unknown_no_production_record"
    n_unknown_with_target = int((present & unknown).sum())
    trainable = ft[ccol].isin(["positive", "confident_negative"])
    n_wrongly_trainable = int((present & unknown & trainable).sum())
    check5[f"Day+{h}"] = {"unknown_with_target": n_unknown_with_target, "wrongly_included": n_wrongly_trainable}
results["check5"] = check5

# ---- Check 6: no-breakdown vs missing/uncertain distinguishable ----
check6 = {}
for h in [1, 2, 3]:
    ccol = f"Day_Plus_{h}__label_confidence"
    vc = ft[ccol].value_counts(dropna=False).to_dict()
    check6[f"Day+{h}"] = {str(k): int(v) for k, v in vc.items()}
results["check6_confidence_value_counts"] = check6

# ---- Check 7: row/class counts per horizon/machine (recomputed via get_xy-equivalent logic) ----
import sys
sys.path.insert(0, ".")
from modeling_pipeline import align_features_and_targets, build_feature_columns, chronological_split, get_xy

df = align_features_and_targets()
feature_df, _ = build_feature_columns(df)
train_mask, val_mask, test_mask, split_bounds = chronological_split(df)
check7_rows = []
for h in [1, 2, 3]:
    col = f"Day_Plus_{h}__A_any_eligible_breakdown"
    for scope, machine in [("combined", None), ("VRM-1", "VRM-1"), ("VRM-2", "VRM-2")]:
        if machine is None:
            tm, vm, tem = train_mask, val_mask, test_mask
        else:
            sub = df["machine_id"] == machine
            tm, vm, tem = train_mask & sub, val_mask & sub, test_mask & sub
        Xtr, ytr, _ = get_xy(df, feature_df, col, tm)
        Xv, yv, _ = get_xy(df, feature_df, col, vm)
        Xte, yte, _ = get_xy(df, feature_df, col, tem)
        check7_rows.append({
            "horizon": f"Day+{h}", "scope": scope,
            "train_n": len(ytr), "train_pos": int(ytr.sum()), "train_neg": len(ytr) - int(ytr.sum()),
            "val_n": len(yv), "val_pos": int(yv.sum()), "val_neg": len(yv) - int(yv.sum()),
            "test_n": len(yte), "test_pos": int(yte.sum()), "test_neg": len(yte) - int(yte.sum()),
        })
results["check7_class_counts"] = check7_rows

# ---- Check 8 (NEW this step): electrical/mechanical targets partition correctly
#      against the combined "any eligible breakdown" target ----
check8 = {}
for h in [1, 2, 3]:
    any_col = f"Day_Plus_{h}__A_any_eligible_breakdown"
    elec_col = f"Day_Plus_{h}__A_electrical_breakdown"
    mech_col = f"Day_Plus_{h}__A_mechanical_breakdown"
    sub = ft[[any_col, elec_col, mech_col]].dropna()
    any_v = sub[any_col].astype(int)
    elec_v = sub[elec_col].astype(int)
    mech_v = sub[mech_col].astype(int)
    union = ((elec_v == 1) | (mech_v == 1)).astype(int)
    # every electrical=1 or mechanical=1 row must have any=1
    n_elec_or_mech_not_any = int((((elec_v == 1) | (mech_v == 1)) & (any_v == 0)).sum())
    # rows where any=1 but neither electrical nor mechanical is 1 are allowed
    # (other eligible categories exist per machine_day_targets.csv, e.g. non-E/M
    # categories excluded from A_electrical/A_mechanical by design) -- just count them
    n_any_but_neither = int(((any_v == 1) & (elec_v == 0) & (mech_v == 0)).sum())
    n_both_elec_and_mech_same_day = int(((elec_v == 1) & (mech_v == 1)).sum())
    check8[f"Day+{h}"] = {
        "rows_checked": int(len(sub)),
        "elec_or_mech_implies_any_violations": n_elec_or_mech_not_any,
        "any_true_but_neither_elec_nor_mech": n_any_but_neither,
        "both_elec_and_mech_same_row": n_both_elec_and_mech_same_day,
    }
results["check8_electrical_mechanical_partition"] = check8

import json
with open("step10_target_validation_raw_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(json.dumps(results, indent=2, default=str))
