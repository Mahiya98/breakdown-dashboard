"""Driver: baselines -> feature selection -> traditional-ML comparison,
for the Day+1 / Day+2 / Day+3 "any eligible breakdown" target.

Per the brief:
  1. Majority-class baseline, training-frequency baseline, DummyClassifier,
     regularized Logistic Regression.
  2. Decision Tree, Random Forest, Extra Trees, HistGradientBoosting.
     (XGBoost/LightGBM/CatBoost were checked and are NOT installed in this
     environment -- skipped rather than installing a large new dependency
     stack, per the brief's instruction not to install unnecessary packages.)
  3. Feature selection is fit on the TRAIN split only, for every model that
     isn't itself regularized/tree-based with built-in feature handling.
  4. Combined (both machines, machine_id as a feature) vs separate per-machine
     models -- only attempted if each machine's split leaves enough rows.
"""
import json
import re
import pandas as pd
from modeling_pipeline import (
    align_features_and_targets, build_feature_columns, chronological_split,
    get_xy, fit_imputer, apply_imputer, fit_scaler, apply_scaler, evaluate_classifier,
    select_features_train_only, RANDOM_SEED,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier

TARGET = "Day_Plus_1__A_any_eligible_breakdown"  # primary target for this run
HORIZONS = ["Day_Plus_1__A_any_eligible_breakdown",
            "Day_Plus_2__A_any_eligible_breakdown",
            "Day_Plus_3__A_any_eligible_breakdown"]


def run_one_target(df, feature_df, target_col, train_mask, val_mask, test_mask, split_bounds, machine_subset=None):
    if machine_subset is not None:
        sub_mask = df["machine_id"] == machine_subset
        tr_m, va_m, te_m = train_mask & sub_mask, val_mask & sub_mask, test_mask & sub_mask
    else:
        tr_m, va_m, te_m = train_mask, val_mask, test_mask

    X_train_raw, y_train, _ = get_xy(df, feature_df, target_col, tr_m)
    X_val_raw, y_val, _ = get_xy(df, feature_df, target_col, va_m)
    X_test_raw, y_test, _ = get_xy(df, feature_df, target_col, te_m)

    scope = machine_subset or "combined"
    horizon_match = re.match(r"Day_Plus_(\d+)__", target_col)
    horizon_label = f"Day_Plus_{horizon_match.group(1)}" if horizon_match else "unknown"
    if len(y_train) < 30 or y_train.nunique() < 2 or y_val.nunique() < 2:
        return [{"model": "SKIPPED", "feature_selection": "n/a", "n_features": None,
                 "hyperparameters": f"insufficient data for scope={scope}: "
                                     f"train_rows={len(y_train)}, train_classes={y_train.nunique()}, "
                                     f"val_rows={len(y_val)}, val_classes={y_val.nunique()}",
                 "random_seed": RANDOM_SEED, "train_rows": len(y_train), "train_seconds": 0,
                 "split": split_bounds, "scope": scope,
                 "horizon": horizon_label, "target_col": target_col}]

    imp, impute_cols, dropped_all_nan = fit_imputer(X_train_raw)
    X_train = apply_imputer(imp, impute_cols, X_train_raw)
    X_val = apply_imputer(imp, impute_cols, X_val_raw)
    X_test = apply_imputer(imp, impute_cols, X_test_raw)

    results = []

    # ---- 1. Baselines ----
    results.append(evaluate_classifier(
        "Majority-class baseline", DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_selection="none (baseline)", n_features=0, split_bounds=split_bounds))

    results.append(evaluate_classifier(
        "Training-frequency baseline", DummyClassifier(strategy="prior", random_state=RANDOM_SEED),
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_selection="none (baseline)", n_features=0, split_bounds=split_bounds))

    results.append(evaluate_classifier(
        "Dummy classifier (stratified)", DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_selection="none (baseline)", n_features=0, split_bounds=split_bounds))

    # Regularized Logistic Regression: feature-selected via train-only L1 pre-pass,
    # then SCALED (train-only StandardScaler -- logistic regression is scale-sensitive,
    # tree models below are not and are deliberately left unscaled), then refit with L2.
    Xtr_sel, Xval_sel, Xtest_sel, sel_cols = select_features_train_only(
        X_train, y_train, X_val, X_test, method="l1_logistic")
    scaler = fit_scaler(Xtr_sel)
    Xtr_scaled = apply_scaler(scaler, Xtr_sel)
    Xval_scaled = apply_scaler(scaler, Xval_sel)
    Xtest_scaled = apply_scaler(scaler, Xtest_sel)
    results.append(evaluate_classifier(
        "Logistic Regression (L1-selected, scaled, L2-fit)",
        LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
        Xtr_scaled, y_train, Xval_scaled, y_val, Xtest_scaled, y_test,
        feature_selection="L1 logistic (train-only)", n_features=len(sel_cols), split_bounds=split_bounds,
        scaling="StandardScaler (train-only fit)"))

    # ---- 2. Traditional ML comparison, all on the same L1-selected feature set
    #          (keeps the comparison apples-to-apples and tractable at 900 rows) ----
    tree_models = [
        ("Decision Tree", DecisionTreeClassifier(max_depth=4, min_samples_leaf=10,
                                                   class_weight="balanced", random_state=RANDOM_SEED)),
        ("Random Forest", RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                                   class_weight="balanced", random_state=RANDOM_SEED)),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                               class_weight="balanced", random_state=RANDOM_SEED)),
        ("HistGradientBoosting", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                                                   max_iter=200, random_state=RANDOM_SEED)),
    ]
    for name, clf in tree_models:
        results.append(evaluate_classifier(
            name, clf, Xtr_sel, y_train, Xval_sel, y_val, Xtest_sel, y_test,
            feature_selection="L1 logistic (train-only)", n_features=len(sel_cols), split_bounds=split_bounds,
            scaling="none (tree-based, scale-invariant)"))

    # Step 9 audit fix (finding #5): tag every result row with exactly which
    # horizon/target it came from, so the results CSV is self-describing and
    # doesn't require re-deriving script loop order to interpret.
    for r in results:
        r["scope"] = scope
        r["horizon"] = horizon_label
        r["target_col"] = target_col
    return results


def main():
    df = align_features_and_targets()
    feature_df, excluded_log = build_feature_columns(df)
    print(f"Feature matrix for modeling: {feature_df.shape}")
    for k, v in excluded_log.items():
        print(f"  excluded[{k}]: {len(v)} columns")

    train_mask, val_mask, test_mask, split_bounds = chronological_split(df)
    print("Split bounds:", split_bounds)
    print("Machine counts by split:")
    for name, m in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        print(f"  {name}:", df.loc[m, "machine_id"].value_counts().to_dict())

    all_results = []
    for target_col in HORIZONS:
        print(f"\n=== {target_col} (combined) ===")
        all_results.extend(run_one_target(df, feature_df, target_col, train_mask, val_mask, test_mask, split_bounds))

        for machine in ["VRM-1", "VRM-2"]:
            print(f"=== {target_col} ({machine} only) ===")
            all_results.extend(run_one_target(df, feature_df, target_col, train_mask, val_mask, test_mask,
                                                split_bounds, machine_subset=machine))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("model_comparison_results.csv", index=False)
    with open("excluded_columns_log.json", "w") as f:
        json.dump(excluded_log, f, indent=2)
    print("\nWrote model_comparison_results.csv and excluded_columns_log.json")
    print(results_df[["model", "scope", "n_features", "val_roc_auc", "val_pr_auc", "test_roc_auc", "test_pr_auc"]]
          .to_string())


if __name__ == "__main__":
    main()
