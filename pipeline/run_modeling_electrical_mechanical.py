"""Driver: same pipeline as run_modeling.py (baselines -> L1-selected,
train-only-scaled Logistic Regression -> tree ensembles), but for the
electrical-only and mechanical-only Day+1/+2/+3 targets instead of the
combined "any eligible breakdown" target.

Targets used (already present in forward_targets.parquet, built from the
authoritative Category/Code mapping in label_taxonomy.py -- NOT keyword
guessing, and already fixed by the Step 9/Phase 1 label_confidence shift
correction, since forward_targets.parquet is the corrected file):
  Day_Plus_{1,2,3}__A_electrical_breakdown
  Day_Plus_{1,2,3}__A_mechanical_breakdown

Reuses run_one_target() unchanged from run_modeling.py (same exclusion
list, same train-only imputer/selector/scaler fitting, same metrics).
No new feature-selection method set introduced -- identical config to the
already-validated combined-target run.
"""
import json
import pandas as pd
from modeling_pipeline import align_features_and_targets, build_feature_columns, chronological_split
from run_modeling import run_one_target

HORIZONS = []
for h in [1, 2, 3]:
    for cat in ["electrical", "mechanical"]:
        HORIZONS.append(f"Day_Plus_{h}__A_{cat}_breakdown")


def main():
    df = align_features_and_targets()
    feature_df, excluded_log = build_feature_columns(df)
    print(f"Feature matrix for modeling: {feature_df.shape}")

    train_mask, val_mask, test_mask, split_bounds = chronological_split(df)
    print("Split bounds:", split_bounds)

    all_results = []
    for target_col in HORIZONS:
        print(f"\n=== {target_col} (combined) ===")
        all_results.extend(run_one_target(df, feature_df, target_col, train_mask, val_mask, test_mask, split_bounds))
        for machine in ["VRM-1", "VRM-2"]:
            print(f"=== {target_col} ({machine} only) ===")
            all_results.extend(run_one_target(df, feature_df, target_col, train_mask, val_mask, test_mask,
                                               split_bounds, machine_subset=machine))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("step10_electrical_mechanical_results.csv", index=False)
    print("\nWrote step10_electrical_mechanical_results.csv")
    print(results_df[["model", "scope", "target_col", "n_features", "val_roc_auc", "val_pr_auc",
                       "test_roc_auc", "test_pr_auc"]].to_string())


if __name__ == "__main__":
    main()
