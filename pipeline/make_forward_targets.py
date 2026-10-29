"""Build leakage-free forward targets for Day+1, +2, +3 prediction horizons.

Inputs:
- feature_matrix.parquet: existing feature matrix with same-day targets only
- machine_day_targets.csv: A/B/C targets built by label_taxonomy.py
- target_definition_report.md: authoritative level-1/2 family mapping
- label_taxonomy.py: same mapping in code form (used here)

Outputs:
- forward_targets.parquet: one row per (machine, date), aligned to
  feature_matrix.parquet's own rows, with:
    - machine_id, Date (inherited from feature_matrix.parquet)
    - operating_status, label_confidence (inherited, used for masking, not features)
    - Day_Plus_N__A_any_eligible_breakdown: binary, for N in {1, 2, 3}
    - Day_Plus_N__A_electrical_breakdown: binary, for N in {1, 2, 3}
    - Day_Plus_N__A_mechanical_breakdown: binary, for N in {1, 2, 3}
    - Day_Plus_N__B_<family>_occurred: binary per level-2 family, for N in {1, 2, 3}
    - Day_Plus_N__C_<family>_event_count: int per level-2 family, for N in {1, 2, 3}

No-leakage discipline:
1. We start from feature_matrix.parquet's existing 964 rows -- no future rows
   are introduced.
2. Forward targets are built by sorting machine_day_targets.csv by
   (machine, date) and taking a plain per-machine groupby().shift(-h): row
   i's Day_Plus_h value is the SAME column's value h rows later IN THAT
   MACHINE'S OWN date-sorted sequence. Since the source table is one row per
   calendar day with no gaps (built from a full date_range), "h rows later"
   is equivalent to "h days later" here -- this is checked below.
3. Rows that don't have a day h days later (the last h rows of each machine)
   get explicit NaNs, never a coerced value.
4. operating_status and label_confidence are carried through purely for
   masking at evaluation time; they must never be used as model features.

BUG FIX (this version): an earlier version of this function used
`.unstack("machine").shift(-h).stack().reindex(...)`, which silently
produced an all-NaN result for every Day_Plus_N column -- the unstack/
stack index levels and the reindex target's level order and dtypes did not
line up, and pandas reindex failed closed (NaN) rather than raising. This
was caught by inspecting forward_targets.parquet's actual values (100% NaN)
rather than trusting that the code "looked right." The fix below avoids
unstack/stack entirely and uses a single, auditable per-machine shift.
"""
import pandas as pd

FEATURE_MATRIX_PATH = "/home/claude/work/pdm_system/feature_matrix.parquet"
MACHINE_DAY_TARGETS_PATH = "/home/claude/work/pdm_system/target_output/machine_day_targets.csv"
REPORT_PATH = "/home/claude/work/pdm_system/target_definition_report.md"

# Hard-coded Level1->Level2 family mapping from label_taxonomy.py / the
# target definition report, to avoid a circular import. If the taxonomy
# changes, update it here too.
LEVEL2_ELECTRICAL_FAMILIES = [
    "Fan & Separator Mechanical/Electrical Failures",
    "Electrical Protection, Instrumentation & Control Faults",
    "Bag House / Bag Filter / Dust Collection",
    "Motor & Electrical Drive Failures",
    "Conveyor & Feeder Mechanical Failures",
    "Hydraulic & Pneumatic System Failures",
    "Bearing, Gearbox & Lubrication Failures",
    "Other Confirmed Electrical Failures",
    "Mill / Roller / Structural Mechanical Failures",
]

LEVEL2_MECHANICAL_FAMILIES = [
    "Bearing, Gearbox & Lubrication Failures",
    "Conveyor & Feeder Mechanical Failures",
    "Bag House / Bag Filter / Dust Collection",
    "Hydraulic & Pneumatic System Failures",
    "Other Confirmed Mechanical Failures",
    "Mill / Roller / Structural Mechanical Failures",
    "Electrical Protection, Instrumentation & Control Faults",
    "Motor & Electrical Drive Failures",
]


def make_forward_targets(feature_matrix_path, machine_day_targets_path, horizons=(1, 2, 3)):
    df = pd.read_parquet(feature_matrix_path)
    df["Date"] = pd.to_datetime(df["Date"])
    targets = pd.read_csv(machine_day_targets_path)
    targets["date"] = pd.to_datetime(targets["date"])
    targets = targets.sort_values(["machine", "date"]).reset_index(drop=True)

    # Sanity check: one row per calendar day per machine, no gaps -- this is
    # what makes "shift by h rows" equivalent to "shift by h days" below.
    for m, g in targets.groupby("machine"):
        full_range = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        assert len(g) == len(full_range), (
            f"machine_day_targets.csv has a date gap for {m}: "
            f"{len(g)} rows vs {len(full_range)} expected calendar days"
        )

    # BUG FIX (Step 9 audit, defect #6/10): label_confidence and operating_status
    # live in feature_matrix.parquet (df), NOT in machine_day_targets.csv, and were
    # previously carried through UNSHIFTED -- i.e. "label_confidence_targ" was always
    # identical to the same-day "label_confidence_feat", not the future day's value,
    # even though get_xy() used it as if it were the future day's confidence. Fix:
    # shift label_confidence/operating_status forward per-machine the exact same way
    # the A_/B_/C_ target columns are shifted below, from df itself (not `targets`).
    conf_src = df[["machine_id", "Date", "label_confidence", "operating_status"]].copy()
    conf_src = conf_src.sort_values(["machine_id", "Date"]).reset_index(drop=True)
    for m, g in conf_src.groupby("machine_id"):
        full_range = pd.date_range(g["Date"].min(), g["Date"].max(), freq="D")
        assert len(g) == len(full_range), (
            f"feature_matrix.parquet has a date gap for {m}: "
            f"{len(g)} rows vs {len(full_range)} expected calendar days"
        )
    conf_shifted_frames = []
    for h in horizons:
        prefix = f"Day_Plus_{h}__"
        shifted = conf_src.groupby("machine_id", sort=False)[["label_confidence", "operating_status"]].shift(-h)
        shifted.columns = [f"{prefix}label_confidence", f"{prefix}operating_status"]
        conf_shifted_frames.append(shifted)
    conf_forward = pd.concat([conf_src[["machine_id", "Date"]]] + conf_shifted_frames, axis=1)

    # Derive Level-1 electrical/mechanical occurrence (same-day), before shifting.
    elec_cols = [f"B_{f}_occurred" for f in LEVEL2_ELECTRICAL_FAMILIES]
    mech_cols = [f"B_{f}_occurred" for f in LEVEL2_MECHANICAL_FAMILIES]
    targets["A_electrical_breakdown"] = targets[elec_cols].any(axis=1).astype(int)
    targets["A_mechanical_breakdown"] = targets[mech_cols].any(axis=1).astype(int)

    family_b_cols = [c for c in targets.columns if c.startswith("B_")]
    family_c_cols = [f"C_{c[2:-9]}_event_count" for c in family_b_cols]
    all_target_cols = ["A_any_eligible_breakdown", "A_electrical_breakdown",
                        "A_mechanical_breakdown"] + family_b_cols + family_c_cols

    shifted_frames = []
    for h in horizons:
        prefix = f"Day_Plus_{h}__"
        shifted = targets.groupby("machine", sort=False)[all_target_cols].shift(-h)
        shifted.columns = [f"{prefix}{c}" for c in all_target_cols]
        shifted_frames.append(shifted)

    forward_cols = pd.concat(shifted_frames, axis=1)
    targets_with_forward = pd.concat([targets[["machine", "date"]], forward_cols], axis=1)

    # Re-key onto feature_matrix.parquet's own (machine_id, Date) rows.
    out = df[["machine_id", "Date", "operating_status", "label_confidence"]].copy()
    merged = out.merge(
        targets_with_forward.rename(columns={"machine": "machine_id", "date": "Date"}),
        on=["machine_id", "Date"], how="left",
    )
    assert len(merged) == len(out), "Row count changed during forward-target merge"

    # Merge in the newly-shifted future label_confidence/operating_status (fix for
    # Step 9 defect #6/10). conf_forward is already keyed on (machine_id, Date) and
    # was derived from df itself, so this merge is 1:1 with no row-count change.
    merged = merged.merge(conf_forward, on=["machine_id", "Date"], how="left")
    assert len(merged) == len(out), "Row count changed during confidence-shift merge"
    for h in horizons:
        col = f"Day_Plus_{h}__label_confidence"
        assert col in merged.columns, f"Missing shifted confidence column: {col}"

    return merged


if __name__ == "__main__":
    forward_targets = make_forward_targets(
        feature_matrix_path=FEATURE_MATRIX_PATH,
        machine_day_targets_path=MACHINE_DAY_TARGETS_PATH,
    )
    print(f"Forward target matrix: {forward_targets.shape}")

    # Verification: confirm the fix actually produced non-NaN values and
    # that a spot-checked row's Day+1 target matches the next day's actual
    # same-day target directly from machine_day_targets.csv.
    n_non_na = forward_targets["Day_Plus_1__A_any_eligible_breakdown"].notna().sum()
    assert n_non_na > 0, "Day_Plus_1 target is still all-NaN -- fix did not work"
    print(f"Day_Plus_1__A_any_eligible_breakdown non-NaN rows: {n_non_na} / {len(forward_targets)}")

    raw_targets = pd.read_csv(MACHINE_DAY_TARGETS_PATH)
    raw_targets["date"] = pd.to_datetime(raw_targets["date"])
    sample = forward_targets.dropna(subset=["Day_Plus_1__A_any_eligible_breakdown"]).iloc[10]
    expected_date = sample["Date"] + pd.Timedelta(days=1)
    expected_row = raw_targets[(raw_targets["machine"] == sample["machine_id"]) & (raw_targets["date"] == expected_date)]
    assert len(expected_row) == 1
    assert int(expected_row["A_any_eligible_breakdown"].iloc[0]) == int(sample["Day_Plus_1__A_any_eligible_breakdown"]), \
        "Spot check failed: Day_Plus_1 value does not match next day's actual same-day target"
    print("Spot check passed: Day_Plus_1 target matches the next calendar day's actual same-day label.")

    for h in (1, 2, 3):
        col = f"Day_Plus_{h}__A_any_eligible_breakdown"
        print(f"{col}: {forward_targets[col].notna().sum()} non-NaN, "
              f"positive rate among non-NaN = {forward_targets[col].mean():.2%}")

    # Verification of the Step 9 defect fix: Day_Plus_1__label_confidence must NOT
    # be identical to the same-day label_confidence (if it were, the shift silently
    # failed again). Spot-check one row directly against feature_matrix.parquet.
    fm_raw = pd.read_parquet(FEATURE_MATRIX_PATH)[["machine_id", "Date", "label_confidence"]]
    fm_raw["Date"] = pd.to_datetime(fm_raw["Date"])
    same_day_vs_shifted = (forward_targets["label_confidence"] == forward_targets["Day_Plus_1__label_confidence"]).mean()
    print(f"\nlabel_confidence == Day_Plus_1__label_confidence for {same_day_vs_shifted:.1%} of rows "
          f"(should be well below 100% now that the shift is fixed)")
    assert same_day_vs_shifted < 0.99, "Day_Plus_1__label_confidence still looks unshifted -- fix did not work"

    conf_sample = forward_targets.dropna(subset=["Day_Plus_1__label_confidence"]).iloc[15]
    expected_conf_date = conf_sample["Date"] + pd.Timedelta(days=1)
    expected_conf_row = fm_raw[(fm_raw["machine_id"] == conf_sample["machine_id"]) & (fm_raw["Date"] == expected_conf_date)]
    assert len(expected_conf_row) == 1
    assert expected_conf_row["label_confidence"].iloc[0] == conf_sample["Day_Plus_1__label_confidence"], \
        "Spot check failed: Day_Plus_1__label_confidence does not match the next calendar day's actual confidence"
    print("Spot check passed: Day_Plus_1__label_confidence matches the next calendar day's actual label_confidence.")

    for h in (1, 2, 3):
        col = f"Day_Plus_{h}__label_confidence"
        trainable = forward_targets[col].isin(["positive", "confident_negative"])
        print(f"{col}: {forward_targets[col].notna().sum()} non-NaN, "
              f"{trainable.sum()} trainable (positive/confident_negative)")

    forward_targets.to_parquet("forward_targets.parquet", index=False)

    with open("forward_targets_report.md", "w") as f:
        f.write(f"""# Forward Targets Report

Objective: build leakage-free Day+1, Day+2, Day+3 targets, aligned to feature_matrix.parquet's existing rows, to enable 1/2/3-day-ahead forecasting.

## Bug found and fixed in this stage

The first version of this script used `.unstack("machine").shift(-h).stack().reindex(...)` to
produce the forward-shifted columns. This silently produced an **all-NaN** result for every
single `Day_Plus_N__*` column -- confirmed by directly inspecting `forward_targets.parquet`'s
values (100% NaN), not just by re-reading the code and assuming it was correct. Downstream, this
would have caused every model-comparison run to be skipped for lack of a usable target, which is
exactly what happened on the first `run_modeling.py` attempt (9/9 "SKIPPED: train_rows=0").

**Fix**: the function now sorts `machine_day_targets.csv` by (machine, date) and applies a plain
`groupby("machine").shift(-h)` -- no unstack/stack/reindex gymnastics. A sanity assertion confirms
each machine's rows are one-per-calendar-day with no gaps (so "shift by h rows" is equivalent to
"shift by h days"), and a spot check below confirms a sampled row's `Day_Plus_1` value exactly
matches the next calendar day's actual same-day `A_any_eligible_breakdown` value.

## Methodology

1. Start from feature_matrix.parquet's existing {forward_targets.shape[0]}-row frame (one row per
   historical prediction point). No future rows are introduced.
2. For each same-day target column (`A_any_eligible_breakdown`, `B_<family>_occurred`,
   `C_<family>_event_count`), create a Day+N version via `groupby(machine).shift(-N)` on the
   date-sorted target table, then re-key onto the feature matrix's own rows.
3. The last N rows of each machine's history have no day N days later within the dataset, so they
   are explicit NaNs for a Day+N target, never a coerced value.
4. Electrical/Mechanical Day+N targets are derived from the authoritative Level1->Level2 family
   mapping (hard-coded from label_taxonomy.py / the target definition report):

   Electrical families: {LEVEL2_ELECTRICAL_FAMILIES}
   Mechanical families: {LEVEL2_MECHANICAL_FAMILIES}

   The Day+N Electrical target is 1 if ANY of that day's electrical-family `B_*_occurred` columns
   is 1 (else 0). Likewise for Mechanical.

## Verification performed

- Row count is unchanged after the merge ({forward_targets.shape[0]} rows in, {forward_targets.shape[0]} rows out).
- `Day_Plus_1__A_any_eligible_breakdown` has {n_non_na} non-NaN values (not all-NaN as in the buggy version).
- Spot check: a sampled row's `Day_Plus_1__A_any_eligible_breakdown` value was confirmed to exactly
  equal the next calendar day's actual same-day `A_any_eligible_breakdown` value, read directly from
  `machine_day_targets.csv`.

## Forward Target Matrix

{forward_targets.shape[0]} rows x {forward_targets.shape[1]} columns, aligned to feature_matrix.parquet's existing rows.

| Horizon | Non-NaN rows | Positive rate (non-NaN) |
|---|---|---|
""" + "\n".join(
            f"| Day+{h} | {forward_targets[f'Day_Plus_{h}__A_any_eligible_breakdown'].notna().sum()} | "
            f"{forward_targets[f'Day_Plus_{h}__A_any_eligible_breakdown'].mean():.2%} |"
            for h in (1, 2, 3)
        ) + """

## Columns produced

- `Day_Plus_N__A_any_eligible_breakdown`: binary, N in {1, 2, 3}
- `Day_Plus_N__A_electrical_breakdown`: binary, N in {1, 2, 3}
- `Day_Plus_N__A_mechanical_breakdown`: binary, N in {1, 2, 3}
- `Day_Plus_N__B_<family>_occurred`: binary per level-2 family, N in {1, 2, 3}
- `Day_Plus_N__C_<family>_event_count`: int per level-2 family, N in {1, 2, 3}
""")
    print("Wrote forward_targets_report.md")
