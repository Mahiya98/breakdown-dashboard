"""Predict the LIKELY breakdown reason group for every (machine, date, category,
horizon) row -- not just the ground-truth reason on days a breakdown actually
happened (that's generate_dashboard_predictions.py / the "actual reason" column).

This is a CONDITIONAL classifier: "if an electrical/mechanical breakdown occurs on
this target date, which failure-family GROUP is it most likely to be?" It is
trained only on rows where that category's breakdown actually occurred (same
leakage-safe train-only-fit discipline as the probability models), then scored
on EVERY row -- including rows where no breakdown is predicted or known -- so the
dashboard can show a predicted reason group for any date, with a confidence score.

This is a genuinely hard, small-sample multiclass problem: electrical/mechanical
breakdowns happen on a small fraction of days, and several failure-family groups
have only a handful of historical events. Rare groups (fewer than MIN_CLASS_COUNT
total occurrences) are merged into "Other" so the model isn't asked to learn a
class it has seen 1-2 times. Reported val/test accuracy should be read with that
caveat -- see dashboard note text.
"""
import json
import numpy as np
import pandas as pd
from modeling_pipeline import (
    align_features_and_targets, build_feature_columns, chronological_split,
    fit_imputer, apply_imputer, fit_scaler, apply_scaler, RANDOM_SEED,
)
from make_forward_targets import LEVEL2_ELECTRICAL_FAMILIES, LEVEL2_MECHANICAL_FAMILIES
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score

SELECT_K = 40  # small on purpose: train sets here are tens of rows, not hundreds


def select_features_multiclass(X_train, y_train, X_val, X_test, k=SELECT_K):
    """ANOVA F-test feature selection, fit on TRAIN ONLY -- supports multiclass
    targets natively (unlike modeling_pipeline's l1_logistic path, which is
    binary-only via coef_[0])."""
    kbest = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
    kbest.fit(X_train, y_train)
    cols = X_train.columns[kbest.get_support()]
    return X_train[cols], X_val[cols], X_test[cols], list(cols)

# MIN_CLASS_COUNT is now set per-loop, not global
HORIZONS = (1, 2, 3)
CATEGORY_FAMILIES = {"electrical": LEVEL2_ELECTRICAL_FAMILIES, "mechanical": LEVEL2_MECHANICAL_FAMILIES}

CANDIDATES = {
    "Logistic Regression": lambda: LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                                                       class_weight="balanced", random_state=RANDOM_SEED),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=1,
                                                    class_weight="balanced_subsample", random_state=RANDOM_SEED),
    "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_leaf=1,
                                                class_weight="balanced_subsample", random_state=RANDOM_SEED),
}


def winning_family(row, h, families):
    """The family with the most events for this row's Day+h, among this
    category's family list -- None if none of them occurred."""
    best_fam, best_count = None, 0
    for fam in families:
        col = f"Day_Plus_{h}__C_{fam}_event_count"
        c = row.get(col, 0)
        c = 0 if pd.isna(c) else c
        if c > best_count:
            best_fam, best_count = fam, c
    return best_fam


def collapse_rare(labels, min_count):
    counts = labels.value_counts()
    rare = set(counts[counts < min_count].index)
    return labels.apply(lambda v: "Other" if v in rare else v)


def main():
    df = align_features_and_targets()
    feature_df, _ = build_feature_columns(df)
    train_mask, val_mask, test_mask, _ = chronological_split(df)
    split_label = pd.Series("train", index=df.index)
    split_label[val_mask] = "val"
    split_label[test_mask] = "test"

    results_rows = []        # metrics per (category, horizon)
    predictions = {}         # (category, horizon) -> np.array of (pred_group, confidence) per df row

    for category, families in CATEGORY_FAMILIES.items():
        for h in HORIZONS:
            MIN_CLASS_COUNT = 12  # Was 8; increasing to merge more rare classes
            occ_col = f"Day_Plus_{h}__A_{category}_breakdown"
            occurred = df[occ_col]
            raw_labels = df.apply(lambda r: winning_family(r, h, families), axis=1)
            labels = collapse_rare(raw_labels.dropna(), MIN_CLASS_COUNT)
            labels = labels.reindex(df.index)

            pos_mask = (occurred == 1) & labels.notna()
            train_pos = pos_mask & train_mask
            val_pos = pos_mask & val_mask
            test_pos = pos_mask & test_mask

            MIN_CLASS_COUNT = 12  # Was 8; increasing to merge more rare classes
            labels = collapse_rare(labels[pos_mask], MIN_CLASS_COUNT)
            n_train, n_val, n_test = train_pos.sum(), val_pos.sum(), test_pos.sum()
            print(f"{category}/Day+{h}: train={n_train} val={n_val} test={n_test} "
                  f"COLLAPSED to {sorted(labels[pos_mask].unique())}")

            if n_train < 10 or labels[train_pos].nunique() < 2:
                print(f"  -> skipping (too few training rows / classes) for {category}/Day+{h}")
                results_rows.append({
                    "category": category, "horizon": f"Day_Plus_{h}", "model_used": "insufficient_data",
                    "n_train": int(n_train), "n_val": int(n_val), "n_test": int(n_test),
                    "val_accuracy": None, "test_accuracy": None, "val_macro_f1": None, "test_macro_f1": None,
                    "classes": sorted(labels[pos_mask].unique().tolist()),
                })
                predictions[(category, h)] = [(None, None)] * len(df)
                continue

            X_train_raw = feature_df[train_pos]
            y_train = labels[train_pos]
            imp, impute_cols, _ = fit_imputer(X_train_raw)
            X_train = apply_imputer(imp, impute_cols, X_train_raw)
            X_val = apply_imputer(imp, impute_cols, feature_df[val_pos])
            y_val = labels[val_pos]

            Xtr_sel, Xv_sel, _, sel_cols = select_features_multiclass(X_train, y_train, X_val, X_val)

            best_name, best_clf, best_val_f1 = None, None, -1
            for name, make in CANDIDATES.items():
                clf = make()
                try:
                    clf.fit(Xtr_sel, y_train)
                    val_pred = clf.predict(Xv_sel) if len(y_val) else []
                    val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0) if len(y_val) else 0
                except Exception as e:
                    print(f"  {name} failed: {e}")
                    continue
                if val_f1 > best_val_f1:
                    best_name, best_clf, best_val_f1 = name, clf, val_f1

            X_test = apply_imputer(imp, impute_cols, feature_df[test_pos])
            Xt_sel = X_test[sel_cols]
            y_test = labels[test_pos]
            test_pred = best_clf.predict(Xt_sel) if len(y_test) else []
            test_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0) if len(y_test) else None
            test_acc = accuracy_score(y_test, test_pred) if len(y_test) else None
            val_acc = accuracy_score(y_val, best_clf.predict(Xv_sel)) if len(y_val) else None

            results_rows.append({
                "category": category, "horizon": f"Day_Plus_{h}", "model_used": best_name,
                "n_train": int(n_train), "n_val": int(n_val), "n_test": int(n_test),
                "val_accuracy": round(float(val_acc), 4) if val_acc is not None else None,
                "test_accuracy": round(float(test_acc), 4) if test_acc is not None else None,
                "val_macro_f1": round(float(best_val_f1), 4),
                "test_macro_f1": round(float(test_f1), 4) if test_f1 is not None else None,
                "classes": sorted(y_train.unique().tolist()),
            })

            # Score EVERY row (not just positive ones) so the dashboard always has a
            # "most likely reason if a breakdown occurs" prediction, with confidence.
            X_all = apply_imputer(imp, impute_cols, feature_df)
            Xall_sel = X_all[sel_cols]
            proba = best_clf.predict_proba(Xall_sel)
            classes = best_clf.classes_
            top_idx = proba.argmax(axis=1)
            row_preds = [(classes[i], round(float(proba[r, i]), 4)) for r, i in enumerate(top_idx)]
            predictions[(category, h)] = row_preds
            print(f"  -> selected {best_name}, val_macro_f1={best_val_f1:.3f}, test_macro_f1={test_f1}")

    # Build output rows aligned to df, one per (machine, date, category, horizon)
    out_records = []
    for i in range(len(df)):
        machine_id = df["machine_id"].iloc[i]
        feature_date = df["Date"].iloc[i].strftime("%Y-%m-%d")
        for category in CATEGORY_FAMILIES:
            for h in HORIZONS:
                pred_group, conf = predictions[(category, h)][i]
                target_date = (df["Date"].iloc[i] + pd.Timedelta(days=h)).strftime("%Y-%m-%d")
                out_records.append({
                    "machine_id": machine_id, "feature_date": feature_date,
                    "forecast_horizon": f"Day_Plus_{h}", "forecast_target_date": target_date,
                    "category": category,
                    "predicted_reason_group": pred_group, "confidence": conf,
                    "split": split_label.iloc[i],
                })

    out_df = pd.DataFrame(out_records)
    out_df.to_csv("reason_predictions.csv", index=False)
    print(f"\nWrote reason_predictions.csv: {out_df.shape}")

    metrics_df = pd.DataFrame(results_rows)
    metrics_df.to_csv("reason_model_metrics.csv", index=False)
    print(f"Wrote reason_model_metrics.csv: {metrics_df.shape}")
    print(metrics_df[["category", "horizon", "model_used", "n_train", "val_macro_f1", "test_macro_f1"]])


if __name__ == "__main__":
    main()
