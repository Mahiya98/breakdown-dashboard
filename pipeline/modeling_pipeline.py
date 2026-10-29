"""Leakage-free modeling pipeline for VRM breakdown prediction (Day+1/+2/+3).

Inputs:
- feature_matrix.parquet: engineered features, already leakage-free (every
  feature is lagged relative to its own row's date).
- forward_targets.parquet: forward-looking Day+1/+2/+3 targets aligned to
  the SAME (machine_id, Date) rows as feature_matrix.parquet.

This script is intentionally conservative about what counts as a feature:
- Excludes every `SAMEDAY_AUDIT_ONLY__*` column (same-day raw aggregates,
  explicitly flagged in feature_data_dictionary.md as leakage if used).
- Excludes the original same-day target columns (`A_`, `B_`, `C_`, `E_`
  prefixes) that live in feature_matrix.parquet — those describe the
  feature row's OWN day, not a future day, but leaving them in would still
  let a model see "did today have a breakdown" while predicting whether a
  FUTURE day will, which is a strong, unrealistic proxy at serving time if
  today's own label isn't actually known yet in some deployments. Dropped
  to be safe and because the brief says do not use them as predictors.
- Excludes `operating_status` and `label_confidence` as model inputs
  (used only for masking rows, never as a feature), per the brief.
- Excludes `machine_id`/`Date` from X, but keeps a one-hot `machine_id`
  dummy as the single allowed "machine identity" feature for the combined
  model (Section 8 of the brief).
- Keeps only numeric dtypes after the above exclusions; logs anything else
  dropped so nothing disappears silently.

A hard assertion checks no excluded-pattern column reaches the final X.
"""
import re
import time
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, recall_score, precision_score,
                              f1_score, confusion_matrix, balanced_accuracy_score)

FEATURE_MATRIX_PATH = "/home/claude/work/pdm_system/feature_matrix.parquet"
FORWARD_TARGETS_PATH = "/home/claude/work/pdm_system/forward_targets.parquet"
RANDOM_SEED = 42

EXCLUDE_PREFIXES = ("SAMEDAY_AUDIT_ONLY__", "A_", "B_", "C_", "E_", "Day_Plus_")
EXCLUDE_EXACT = {"machine_id", "Date", "operating_status", "label_confidence",
                  "operating_status_feat", "label_confidence_feat",
                  "operating_status_targ", "label_confidence_targ",
                  "any_breakdown_event_count"}


def align_features_and_targets():
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    targets = pd.read_parquet(FORWARD_TARGETS_PATH)
    features["Date"] = pd.to_datetime(features["Date"])
    targets["Date"] = pd.to_datetime(targets["Date"])
    features = features.set_index(["machine_id", "Date"])
    targets = targets.set_index(["machine_id", "Date"])
    df = features.join(targets, lsuffix="_feat", rsuffix="_targ")
    return df.reset_index()


def build_feature_columns(df, include_machine_dummy=True):
    """Return the exact list of column names allowed into X, plus a log of
    what was excluded and why, so the exclusion is auditable rather than
    implicit."""
    excluded = {"same_day_audit_only": [], "same_day_targets": [],
                "metadata": [], "forward_targets": [], "non_numeric": []}
    keep = []
    for c in df.columns:
        if c in EXCLUDE_EXACT:
            excluded["metadata"].append(c)
            continue
        if c.startswith("SAMEDAY_AUDIT_ONLY__"):
            excluded["same_day_audit_only"].append(c)
            continue
        if c.startswith("Day_Plus_"):
            excluded["forward_targets"].append(c)
            continue
        if c.startswith(("A_", "B_", "C_", "E_")):
            excluded["same_day_targets"].append(c)
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            excluded["non_numeric"].append(c)
            continue
        keep.append(c)

    feature_df = df[keep].copy()
    if include_machine_dummy:
        feature_df["machine_is_VRM2"] = (df["machine_id"] == "VRM-2").astype(int)

    # Hard leakage assertion
    for c in feature_df.columns:
        assert not c.startswith(EXCLUDE_PREFIXES[:-1]), f"Leakage column slipped through: {c}"

    return feature_df, excluded


def chronological_split(df, train_frac=0.70, val_frac=0.15):
    """Split by calendar date (not row percentile) so both machines share
    identical period boundaries -- no calendar period leaks across the
    train/test boundary via one machine's rows while the other machine's
    same-period rows sit in a different split."""
    dates = np.sort(df["Date"].unique())
    n = len(dates)
    train_end = dates[int(n * train_frac) - 1]
    val_end = dates[int(n * (train_frac + val_frac)) - 1]

    train_mask = df["Date"] <= train_end
    val_mask = (df["Date"] > train_end) & (df["Date"] <= val_end)
    test_mask = df["Date"] > val_end
    bounds = {"train_end": str(pd.Timestamp(train_end).date()),
              "val_end": str(pd.Timestamp(val_end).date()),
              "train_rows": int(train_mask.sum()), "val_rows": int(val_mask.sum()),
              "test_rows": int(test_mask.sum())}
    return train_mask, val_mask, test_mask, bounds


def get_xy(df, feature_df, target_col, mask):
    """Rows restricted to: in-split, target non-null, AND the future day's
    own label_confidence is trainable (positive/confident_negative) -- an
    unknown future label must not be coerced to a negative.

    Step 9 audit fix: the future day's confidence column is horizon-specific
    (Day_Plus_1__label_confidence / Day_Plus_2__ / Day_Plus_3__), produced by
    make_forward_targets.py's per-machine forward shift of label_confidence.
    The previous version used "label_confidence_targ", which -- despite its
    name and this docstring's original claim -- was never actually shifted
    and was 100% identical to the same-day label_confidence. This derives the
    correct horizon-matched confidence column from target_col itself, so a
    Day+2 target is never masked using a Day+1 (or same-day) confidence."""
    m = re.match(r"(Day_Plus_\d+__)", target_col)
    if m:
        conf_col = f"{m.group(1)}label_confidence"
        if conf_col not in df.columns:
            raise KeyError(
                f"Expected shifted confidence column {conf_col!r} for target {target_col!r} "
                f"not found -- re-run make_forward_targets.py to regenerate forward_targets.parquet."
            )
        future_conf = df[conf_col]
    else:
        # target_col has no Day_Plus_N__ prefix (e.g. a same-day target) -- fall
        # back to the row's own same-day label_confidence, which IS correct here.
        future_conf = df["label_confidence_feat"] if "label_confidence_feat" in df.columns else df["label_confidence"]
    trainable_future = future_conf.isin(["positive", "confident_negative"])
    row_mask = mask & df[target_col].notna() & trainable_future
    X = feature_df.loc[row_mask]
    y = df.loc[row_mask, target_col].astype(int)
    return X, y, row_mask


def fit_imputer(X_train):
    # pct_change-derived rate-of-change columns can be +/-inf when a prior
    # value was exactly 0 (division by zero). Treat inf as missing, same as
    # NaN, rather than letting it reach the imputer/model as a huge finite-
    # looking number.
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    cols_all_nan = X_train.columns[X_train.isna().all()]
    imp = SimpleImputer(strategy="median")
    X_train_clean = X_train.drop(columns=cols_all_nan)
    imp.fit(X_train_clean)
    return imp, list(X_train_clean.columns), list(cols_all_nan)


def apply_imputer(imp, cols, X):
    X = X[cols].replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(imp.transform(X), columns=cols, index=X.index)


def fit_scaler(X_train):
    """Fit a StandardScaler on TRAIN ONLY. For use before scale-sensitive models
    (e.g. L2-regularized Logistic Regression) -- tree-based models (Decision
    Tree, Random Forest, Extra Trees, HistGradientBoosting) are scale-invariant
    and should NOT be passed through this; apply it only to the logistic
    regression branch."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler, X):
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


def safe_roc_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def safe_pr_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_score)


def evaluate_classifier(name, clf, X_train, y_train, X_val, y_val, X_test, y_test,
                         feature_selection="none", n_features=None, seed=RANDOM_SEED, split_bounds=None,
                         scaling="none"):
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_seconds = time.time() - t0

    def score(X, y):
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[:, 1]
        else:
            proba = clf.decision_function(X)
        pred = clf.predict(X)
        # Confusion matrix is computed with explicit labels=[0,1] so a split with
        # only one class present still returns a full 2x2 matrix (zeros in the
        # missing cells) rather than silently collapsing to 1x1.
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        return {
            "roc_auc": safe_roc_auc(y, proba),
            "pr_auc": safe_pr_auc(y, proba),
            "recall": recall_score(y, pred, zero_division=0),
            "precision": precision_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred) if len(np.unique(y)) > 1 else np.nan,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "support_pos": n_pos, "support_neg": n_neg,
            "positive_rate": float(y.mean()),
            "n": len(y),
        }

    val_scores = score(X_val, y_val)
    test_scores = score(X_test, y_test)

    # Validation/test generalization gap: test minus val, for the two headline
    # ranking metrics. A large positive value (test >> val) is suspicious in the
    # OTHER direction (test looking better than validation ever suggested), and
    # a large negative value is the usual overfit-to-validation symptom; both are
    # worth surfacing explicitly rather than leaving the reader to subtract columns.
    val_test_gap_roc_auc = (test_scores["roc_auc"] - val_scores["roc_auc"]
                             if pd.notna(test_scores["roc_auc"]) and pd.notna(val_scores["roc_auc"]) else np.nan)
    val_test_gap_pr_auc = (test_scores["pr_auc"] - val_scores["pr_auc"]
                            if pd.notna(test_scores["pr_auc"]) and pd.notna(val_scores["pr_auc"]) else np.nan)

    return {
        "model": name,
        "feature_selection": feature_selection,
        "n_features": n_features if n_features is not None else X_train.shape[1],
        "hyperparameters": str(clf.get_params()) if hasattr(clf, "get_params") else "n/a",
        "random_seed": seed,
        "train_rows": len(y_train),
        "train_seconds": round(train_seconds, 3),
        "scaling": scaling,
        "split": split_bounds,
        **{f"val_{k}": v for k, v in val_scores.items()},
        **{f"test_{k}": v for k, v in test_scores.items()},
        "val_test_gap_roc_auc": val_test_gap_roc_auc,
        "val_test_gap_pr_auc": val_test_gap_pr_auc,
    }


def select_features_train_only(X_train, y_train, X_val, X_test, method, k=150):
    """Fit a feature-selection method on TRAIN ONLY, then apply the same
    transform to val/test. Returns (Xtr, Xval, Xtest, selected_columns)."""
    if method == "variance_threshold":
        vt = VarianceThreshold(threshold=0.0)
        vt.fit(X_train)
        cols = X_train.columns[vt.get_support()]
    elif method == "mutual_info_kbest":
        kbest = SelectKBest(mutual_info_classif, k=min(k, X_train.shape[1]))
        kbest.fit(X_train, y_train)
        cols = X_train.columns[kbest.get_support()]
    elif method == "l1_logistic":
        # liblinear's L1 solver scales poorly past a few thousand columns at
        # this row count (674 train rows x 13k+ columns is too slow to run
        # 9 times). Pre-reduce with a fast, vectorized univariate filter
        # (ANOVA F-test, train-only) to <=800 columns first, THEN run L1 for
        # the final sparse selection. This keeps the selection fit entirely
        # on training data; it only changes the speed of getting there.
        if X_train.shape[1] > 800:
            pre = SelectKBest(f_classif, k=800)
            pre.fit(X_train, y_train)
            pre_cols = X_train.columns[pre.get_support()]
        else:
            pre_cols = X_train.columns
        l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.05,
                                 max_iter=1000, random_state=RANDOM_SEED)
        l1.fit(X_train[pre_cols], y_train)
        cols = pre_cols[l1.coef_[0] != 0]
        if len(cols) == 0:
            cols = pre_cols  # fallback: L1 zeroed everything out
    else:
        raise ValueError(method)
    return X_train[cols], X_val[cols], X_test[cols], list(cols)
