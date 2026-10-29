"""Pilot run: ONE target (Day_Plus_1__A_any_eligible_breakdown), ONE scope
(combined), to confirm the corrected pipeline (confidence-fixed masking +
train-only scaler for Logistic Regression) works end-to-end before scaling
to all 3 horizons x 3 scopes. Per instruction: confirm before scaling up.
"""
import pandas as pd
from modeling_pipeline import (
    align_features_and_targets, build_feature_columns, chronological_split,
    get_xy, fit_imputer, apply_imputer, fit_scaler, apply_scaler,
    evaluate_classifier, select_features_train_only, RANDOM_SEED,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier

TARGET_COL = "Day_Plus_1__A_any_eligible_breakdown"
SCOPE = "combined"


def main():
    df = align_features_and_targets()
    feature_df, excluded_log = build_feature_columns(df)
    print(f"Feature matrix for modeling: {feature_df.shape}")
    for k, v in excluded_log.items():
        print(f"  excluded[{k}]: {len(v)} columns")
    # Leakage re-check: confirm nothing excluded reached X
    leaked = [c for c in feature_df.columns
              if c.startswith(("SAMEDAY_AUDIT_ONLY__", "A_", "B_", "C_", "E_", "Day_Plus_"))]
    assert not leaked, f"Leakage columns reached feature matrix: {leaked}"
    assert "machine_id" not in feature_df.columns and "Date" not in feature_df.columns
    assert "operating_status" not in feature_df.columns and "label_confidence" not in feature_df.columns
    print("Leakage re-check passed: no excluded-category column reached the feature matrix.\n")

    train_mask, val_mask, test_mask, split_bounds = chronological_split(df)
    print("Split bounds:", split_bounds)

    X_train_raw, y_train, _ = get_xy(df, feature_df, TARGET_COL, train_mask)
    X_val_raw, y_val, _ = get_xy(df, feature_df, TARGET_COL, val_mask)
    X_test_raw, y_test, _ = get_xy(df, feature_df, TARGET_COL, test_mask)
    print(f"\n{TARGET_COL} / {SCOPE}:")
    print(f"  train n={len(y_train)} pos={int(y_train.sum())} ({y_train.mean():.1%})")
    print(f"  val   n={len(y_val)} pos={int(y_val.sum())} ({y_val.mean():.1%})")
    print(f"  test  n={len(y_test)} pos={int(y_test.sum())} ({y_test.mean():.1%})")

    imp, impute_cols, dropped_all_nan = fit_imputer(X_train_raw)
    X_train = apply_imputer(imp, impute_cols, X_train_raw)
    X_val = apply_imputer(imp, impute_cols, X_val_raw)
    X_test = apply_imputer(imp, impute_cols, X_test_raw)

    results = []

    # ---- Baselines (reused logic, same as before) ----
    for name, strategy in [("Majority-class baseline", "most_frequent"),
                            ("Training-frequency baseline", "prior"),
                            ("Dummy classifier (stratified)", "stratified")]:
        r = evaluate_classifier(name, DummyClassifier(strategy=strategy, random_state=RANDOM_SEED),
                                 X_train, y_train, X_val, y_val, X_test, y_test,
                                 feature_selection="none (baseline)", n_features=0, split_bounds=split_bounds)
        r.update(target_col=TARGET_COL, horizon="Day_Plus_1", scope=SCOPE)
        results.append(r)

    # ---- Regularized Logistic Regression: L1-selected (train-only) -> SCALED (train-only) -> L2 fit ----
    Xtr_sel, Xval_sel, Xtest_sel, sel_cols = select_features_train_only(
        X_train, y_train, X_val, X_test, method="l1_logistic")
    scaler = fit_scaler(Xtr_sel)
    Xtr_scaled = apply_scaler(scaler, Xtr_sel)
    Xval_scaled = apply_scaler(scaler, Xval_sel)
    Xtest_scaled = apply_scaler(scaler, Xtest_sel)
    r = evaluate_classifier(
        "Logistic Regression (L1-selected, scaled, L2-fit)",
        LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
        Xtr_scaled, y_train, Xval_scaled, y_val, Xtest_scaled, y_test,
        feature_selection="L1 logistic (train-only)", n_features=len(sel_cols), split_bounds=split_bounds,
        scaling="StandardScaler (train-only fit)")
    r.update(target_col=TARGET_COL, horizon="Day_Plus_1", scope=SCOPE)
    results.append(r)

    # ---- Traditional ML comparison, same L1-selected feature set, UNSCALED (trees don't need it) ----
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
        r = evaluate_classifier(name, clf, Xtr_sel, y_train, Xval_sel, y_val, Xtest_sel, y_test,
                                 feature_selection="L1 logistic (train-only)", n_features=len(sel_cols),
                                 split_bounds=split_bounds, scaling="none (tree-based, scale-invariant)")
        r.update(target_col=TARGET_COL, horizon="Day_Plus_1", scope=SCOPE)
        results.append(r)

    results_df = pd.DataFrame(results)
    print("\n" + results_df[["model", "target_col", "horizon", "scope", "scaling", "n_features",
                               "train_rows", "val_roc_auc", "val_pr_auc",
                               "test_roc_auc", "test_pr_auc"]].to_string(index=False))
    results_df.to_csv("pilot_day1_combined_results.csv", index=False)
    print("\nWrote pilot_day1_combined_results.csv")
    print("\nSchema check -- required columns present:",
          all(c in results_df.columns for c in ["target_col", "horizon", "scope", "split",
                                                  "train_rows", "val_n", "test_n"]))


if __name__ == "__main__":
    main()
