"""Generate per-row predictions for the dashboard.

Model selection rule (validation-only, baselines excluded from serving
eligibility since a Dummy/majority-class strategy is not an operational
model): for each (target category in {any, electrical, mechanical}) x
(horizon in {Day+1,2,3}), combined scope, pick the feature-based model with
the best val_roc_auc from the already-completed, already-validated
comparisons (model_comparison_results.csv / step10_electrical_mechanical_results.csv).
Refit that exact model (train-only imputer/selector/scaler, same as the
original run) and score every row in the dataset (train+val+test) to get a
real predicted probability per (machine, feature_date, horizon, category).

No fabricated numbers: every score below comes from an actual sklearn
.predict_proba() call on a model fit only on the train split, scored on
feature rows that went through the exact same leakage-safe pipeline already
validated in Step 9/10. Where the future-day label_confidence is unknown
(masked), 'actual' is reported as null/"unknown", never coerced to 0 -- same
rule as the rest of the project.
"""
import json
import pandas as pd
import numpy as np
from modeling_pipeline import (
    align_features_and_targets, build_feature_columns, chronological_split,
    get_xy, fit_imputer, apply_imputer, fit_scaler, apply_scaler,
    select_features_train_only, RANDOM_SEED,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

MODEL_REGISTRY = {
    "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                                  class_weight="balanced", random_state=RANDOM_SEED),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                                       class_weight="balanced", random_state=RANDOM_SEED),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=10,
                                                      class_weight="balanced", random_state=RANDOM_SEED),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                                                      max_iter=200, random_state=RANDOM_SEED),
    "Logistic Regression (L1-selected, scaled, L2-fit)": "logreg",  # special-cased below
}

# Selected models -- chosen above on VALIDATION roc_auc only, baselines excluded from eligibility.
SELECTIONS = {
    ("any", "Day_Plus_1"): ("Day_Plus_1__A_any_eligible_breakdown", "Extra Trees"),
    ("any", "Day_Plus_2"): ("Day_Plus_2__A_any_eligible_breakdown", "Extra Trees"),
    ("any", "Day_Plus_3"): ("Day_Plus_3__A_any_eligible_breakdown", "Extra Trees"),
    ("electrical", "Day_Plus_1"): ("Day_Plus_1__A_electrical_breakdown", "Extra Trees"),
    ("electrical", "Day_Plus_2"): ("Day_Plus_2__A_electrical_breakdown", "Extra Trees"),
    ("electrical", "Day_Plus_3"): ("Day_Plus_3__A_electrical_breakdown", "Extra Trees"),
    ("mechanical", "Day_Plus_1"): ("Day_Plus_1__A_mechanical_breakdown", "Logistic Regression (L1-selected, scaled, L2-fit)"),
    ("mechanical", "Day_Plus_2"): ("Day_Plus_2__A_mechanical_breakdown", "Random Forest"),
    ("mechanical", "Day_Plus_3"): ("Day_Plus_3__A_mechanical_breakdown", "Decision Tree"),
}


def fit_and_score(df, feature_df, target_col, model_name, train_mask, val_mask, test_mask):
    X_train_raw, y_train, _ = get_xy(df, feature_df, target_col, train_mask)
    imp, impute_cols, _ = fit_imputer(X_train_raw)
    X_train = apply_imputer(imp, impute_cols, X_train_raw)

    # Score EVERY row of feature_df (not just trainable ones) so the dashboard can show
    # a prediction for every (machine, date), while 'actual' stays null for masked rows.
    X_all_raw = feature_df.copy()
    X_all = apply_imputer(imp, impute_cols, X_all_raw)

    if model_name == "Logistic Regression (L1-selected, scaled, L2-fit)":
        Xtr_sel, _, _, sel_cols = select_features_train_only(X_train, y_train, X_train, X_train, method="l1_logistic")
        scaler = fit_scaler(Xtr_sel)
        Xtr_scaled = apply_scaler(scaler, Xtr_sel)
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
        clf.fit(Xtr_scaled, y_train)
        X_all_sel = X_all[sel_cols]
        X_all_scaled = apply_scaler(scaler, X_all_sel)
        proba = clf.predict_proba(X_all_scaled)[:, 1]
    else:
        Xtr_sel, _, _, sel_cols = select_features_train_only(X_train, y_train, X_train, X_train, method="l1_logistic")
        clf = MODEL_REGISTRY[model_name]()
        clf.fit(Xtr_sel, y_train)
        X_all_sel = X_all[sel_cols]
        proba = clf.predict_proba(X_all_sel)[:, 1]

    return proba


def load_actual_reasons():
    """Actual historical breakdown reason per (machine, date), ground truth only.

    Source: target_output/event_level_reason_table.csv, built from the audited
    label taxonomy. Multiple shift-level breakdowns on the same machine/day are
    collapsed into one row: all distinct reasons for that day are kept, each
    with its own downtime-minutes weight, ranked by downtime so the dashboard
    can show up to the top 3 contributing reasons for that actual day.
    Days with no logged breakdown simply have no entry here (not a 'zero').
    """
    r = pd.read_csv("target_output/event_level_reason_table.csv")
    r["date"] = pd.to_datetime(r["date"]).dt.strftime("%Y-%m-%d")
    out = {}
    for (machine, date), g in r.groupby(["machine", "date"]):
        g = g.sort_values("Downtime", ascending=False)
        reasons = [
            {
                "classification": row["electrical_mechanical_classification"],
                "reason_group": row["failure_family_level2"],
                "reason_detail": row["normalized_reason_level3"],
                "downtime_minutes": None if pd.isna(row["Downtime"]) else float(row["Downtime"]),
            }
            for _, row in g.head(3).iterrows()
        ]
        out[(machine, date)] = reasons
    return out


def main():
    df = align_features_and_targets()
    feature_df, _ = build_feature_columns(df)
    train_mask, val_mask, test_mask, split_bounds = chronological_split(df)
    actual_reasons = load_actual_reasons()

    split_label = pd.Series("train", index=df.index)
    split_label[val_mask] = "val"
    split_label[test_mask] = "test"

    records = []
    pred_cols = {}  # (category, horizon) -> np.array of probabilities aligned to df.index

    for (category, horizon), (target_col, model_name) in SELECTIONS.items():
        print(f"Fitting {model_name} for {category}/{horizon} (target={target_col}) ...")
        proba = fit_and_score(df, feature_df, target_col, model_name, train_mask, val_mask, test_mask)
        pred_cols[(category, horizon)] = proba

        conf_col = f"{horizon}__label_confidence"
        future_conf = df[conf_col]
        trainable = future_conf.isin(["positive", "confident_negative"])
        actual = df[target_col]

        for i in range(len(df)):
            actual_val = actual.iloc[i]
            is_trainable = bool(trainable.iloc[i])
            if pd.isna(actual_val) or not is_trainable:
                actual_out = None
            else:
                actual_out = int(actual_val)

            machine_id = df["machine_id"].iloc[i]
            target_date = (df["Date"].iloc[i] + pd.Timedelta(days=int(horizon.split("_")[-1]))).strftime("%Y-%m-%d")
            # Show every actual reason logged for that day, regardless of this row's
            # category. We deliberately do NOT filter to classification==category here:
            # the electrical/mechanical *target labels* (A_electrical_breakdown /
            # A_mechanical_breakdown in make_forward_targets.py) are defined from
            # overlapping failure-family lists -- most families count toward BOTH
            # labels -- so a day can be labeled electrical_breakdown=1 purely because
            # of a mechanically-classified event. Filtering reasons by category would
            # then hide the true logged reason behind a label-definition artifact.
            # Each reason carries its own real classification so that mismatch (e.g.
            # a "Mechanical"-tagged reason on an "electrical" category row) stays
            # visible rather than silently blank.
            reasons = actual_reasons.get((machine_id, target_date), [])

            records.append({
                "machine_id": machine_id,
                "feature_date": df["Date"].iloc[i].strftime("%Y-%m-%d"),
                "forecast_horizon": horizon,
                "forecast_target_date": target_date,
                "category": category,
                "model_used": model_name,
                "predicted_score": round(float(proba[i]), 4),
                "predicted_label": int(proba[i] >= 0.5),
                "actual_label": actual_out,
                "split": split_label.iloc[i],
                "actual_reasons": reasons,  # ground-truth only; [] if no breakdown logged for that day
            })

    pred_df = pd.DataFrame(records)
    # CSV export: flatten reasons to a readable string (JSON bundle below keeps full structure).
    csv_df = pred_df.copy()
    csv_df["actual_reasons"] = csv_df["actual_reasons"].apply(
        lambda rs: "; ".join(f"{r['reason_group']} - {r['reason_detail']} ({r['downtime_minutes']}m)" for r in rs) if rs else ""
    )
    csv_df.to_csv("dashboard_predictions.csv", index=False)
    print(f"\nWrote dashboard_predictions.csv: {pred_df.shape}")

    # Attach accuracy metrics (already computed, validated) for the selected models only.
    a = pd.read_csv("model_comparison_results.csv")
    b = pd.read_csv("step10_electrical_mechanical_results.csv")
    metrics_rows = []
    for (category, horizon), (target_col, model_name) in SELECTIONS.items():
        src = a if category == "any" else b
        row = src[(src["scope"] == "combined") & (src["horizon"] == horizon) &
                   (src["model"] == model_name) & (src["target_col"] == target_col)]
        if len(row) != 1:
            raise RuntimeError(f"Expected exactly 1 matching row for {category}/{horizon}, got {len(row)}")
        r = row.iloc[0]
        metrics_rows.append({
            "category": category, "horizon": horizon, "model_used": model_name,
            "val_roc_auc": r["val_roc_auc"], "val_pr_auc": r["val_pr_auc"],
            "val_precision": r["val_precision"], "val_recall": r["val_recall"], "val_f1": r["val_f1"],
            "test_roc_auc": r["test_roc_auc"], "test_pr_auc": r["test_pr_auc"],
            "test_precision": r["test_precision"], "test_recall": r["test_recall"], "test_f1": r["test_f1"],
            "test_balanced_accuracy": r["test_balanced_accuracy"],
            "test_support_pos": int(r["test_support_pos"]), "test_support_neg": int(r["test_support_neg"]),
            "val_test_gap_roc_auc": r["val_test_gap_roc_auc"],
        })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("dashboard_accuracy_metrics.csv", index=False)
    print(f"Wrote dashboard_accuracy_metrics.csv: {metrics_df.shape}")

    # Bundle everything the dashboard needs into one JSON for embedding.
    bundle = {
        "predictions": pred_df.to_dict(orient="records"),
        "metrics": metrics_df.to_dict(orient="records"),
        "split_bounds": split_bounds,
        "generated_note": "Predictions are real model outputs (train-only-fit models, scored on every "
                           "row). Model selection used VALIDATION roc_auc only (test set never used for "
                           "selection). Baseline/dummy strategies were excluded from selection eligibility. "
                           "actual_label is null where the future day's label_confidence is "
                           "unknown_no_production_record -- never coerced to 0.",
    }
    with open("dashboard_data.json", "w") as f:
        json.dump(bundle, f)
    print("Wrote dashboard_data.json")


if __name__ == "__main__":
    main()
