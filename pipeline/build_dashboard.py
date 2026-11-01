"""Build dashboard.html from dashboard_predictions.csv / dashboard_accuracy_metrics.csv.

Run this after generate_dashboard_predictions.py. Produces a compact indexed-array JSON
bundle (not one JSON object per row -- that bloated an earlier version to 2.4MB and broke
loading in the artifact sandbox) and embeds it into dashboard.template.html in place of the
__DASHBOARD_DATA_JSON__ placeholder, writing the result to dashboard.html.
"""
import json
import pandas as pd

PRED_CSV = "dashboard_predictions.csv"
METRICS_CSV = "dashboard_accuracy_metrics.csv"
DATA_JSON = "dashboard_data.json"
REASON_PRED_CSV = "reason_predictions.csv"              # optional: generate_reason_predictions.py
REASON_METRICS_CSV = "reason_model_metrics.csv"          # optional: generate_reason_predictions.py
TEMPLATE = "dashboard.template.html"
OUTPUT = "dashboard.html"


def main():
    df = pd.read_csv(PRED_CSV)
    metrics = pd.read_csv(METRICS_CSV)

    # actual_reasons has real structure (list of dicts) only in the JSON bundle --
    # the CSV flattens it to a string. Pull it from JSON, aligned by row order
    # (generate_dashboard_predictions.py writes both from the same pred_df, in the
    # same order, so positional alignment is safe here).
    with open(DATA_JSON) as f:
        bundle_in = json.load(f)
    reasons_by_row = [rec.get("actual_reasons", []) for rec in bundle_in["predictions"]]
    assert len(reasons_by_row) == len(df), "dashboard_data.json row count must match dashboard_predictions.csv"

    machines = sorted(df["machine_id"].unique().tolist())
    dates = sorted(df["feature_date"].unique().tolist())
    cats = sorted(df["category"].unique().tolist())
    models = sorted(df["model_used"].unique().tolist())
    horizons = ["Day_Plus_1", "Day_Plus_2", "Day_Plus_3"]
    splits = ["train", "val", "test"]

    m_idx = {v: i for i, v in enumerate(machines)}
    d_idx = {v: i for i, v in enumerate(dates)}
    c_idx = {v: i for i, v in enumerate(cats)}
    mo_idx = {v: i for i, v in enumerate(models)}
    h_idx = {v: i for i, v in enumerate(horizons)}
    s_idx = {v: i for i, v in enumerate(splits)}

    # Reason groups/details referenced across all actual_reasons, as index tables
    # so each row only stores small integers, not repeated strings.
    reason_groups = []
    rg_idx = {}
    reason_details = []
    rd_idx = {}

    def _rg(name):
        if name not in rg_idx:
            rg_idx[name] = len(reason_groups)
            reason_groups.append(name)
        return rg_idx[name]

    def _rd(name):
        if name not in rd_idx:
            rd_idx[name] = len(reason_details)
            reason_details.append(name)
        return rd_idx[name]

    # Optional: predicted (not ground-truth) reason group per row, from
    # generate_reason_predictions.py -- "if a breakdown occurs, which failure-family
    # group is it most likely to be", available for every row regardless of whether
    # an actual breakdown is known. Keyed the same way dashboard_predictions.csv rows
    # are identified, so it's a plain lookup at row-build time below.
    reason_pred_lookup = {}
    reason_pred_metrics = []
    try:
        rp = pd.read_csv(REASON_PRED_CSV)
        for r in rp.itertuples(index=False):
            if pd.isna(r.predicted_reason_group):
                continue
            key = (r.machine_id, r.feature_date, r.forecast_horizon, r.category)
            reason_pred_lookup[key] = (_rg(r.predicted_reason_group), round(float(r.confidence), 4))
        rm = pd.read_csv(REASON_METRICS_CSV)
        for r in rm.itertuples(index=False):
            reason_pred_metrics.append({
                "category": r.category, "horizon": r.horizon, "model_used": r.model_used,
                "n_train": int(r.n_train), "n_val": int(r.n_val), "n_test": int(r.n_test),
                "val_accuracy": None if pd.isna(r.val_accuracy) else round(float(r.val_accuracy), 4),
                "test_accuracy": None if pd.isna(r.test_accuracy) else round(float(r.test_accuracy), 4),
                "val_macro_f1": None if pd.isna(r.val_macro_f1) else round(float(r.val_macro_f1), 4),
                "test_macro_f1": None if pd.isna(r.test_macro_f1) else round(float(r.test_macro_f1), 4),
            })
        print(f"Loaded {len(reason_pred_lookup)} reason predictions, {len(reason_pred_metrics)} reason-model metric rows")
    except FileNotFoundError:
        print(f"{REASON_PRED_CSV} not found -- building without predicted reason groups "
              f"(run generate_reason_predictions.py first to include them)")

    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        actual = r.actual_label
        actual = None if pd.isna(actual) else int(actual)

        reasons_enc = []
        for reason in reasons_by_row[i]:
            reasons_enc.append([
                _rg(reason["reason_group"]),
                _rd(reason["reason_detail"]),
                reason["downtime_minutes"],
                0 if reason["classification"] == "Electrical" else 1,
            ])

        pred_reason = reason_pred_lookup.get((r.machine_id, r.feature_date, r.forecast_horizon, r.category))
        pred_reason_group_idx = pred_reason[0] if pred_reason else None
        pred_reason_conf = pred_reason[1] if pred_reason else None

        rows.append([
            m_idx[r.machine_id], d_idx[r.feature_date], h_idx[r.forecast_horizon],
            c_idx[r.category], mo_idx[r.model_used], round(float(r.predicted_score), 4),
            actual, s_idx[r.split], reasons_enc,
            pred_reason_group_idx, pred_reason_conf,
        ])

    metrics_rows = []
    for r in metrics.itertuples(index=False):
        metrics_rows.append({
            "category": r.category, "horizon": r.horizon, "model_used": r.model_used,
            "val_roc_auc": round(float(r.val_roc_auc), 4), "test_roc_auc": round(float(r.test_roc_auc), 4),
            "test_precision": round(float(r.test_precision), 4), "test_recall": round(float(r.test_recall), 4),
            "test_f1": round(float(r.test_f1), 4), "test_balanced_accuracy": round(float(r.test_balanced_accuracy), 4),
            "test_support_pos": int(r.test_support_pos), "test_support_neg": int(r.test_support_neg),
        })

    bundle = {
        "machines": machines, "dates": dates, "cats": cats, "models": models,
        "horizons": horizons, "splits": splits,
        "reason_groups": reason_groups, "reason_details": reason_details,
        "rows": rows,
        "metrics": metrics_rows,
        "reason_model_metrics": reason_pred_metrics,
        "generated_note": (
            "Predictions are real model outputs (train-only-fit models, scored on every row). "
            "Model selection used VALIDATION roc_auc only (test set never used for selection). "
            "Baseline/dummy strategies were excluded from selection eligibility. actual is null "
            "where the future day label_confidence is unknown_no_production_record -- never "
            "coerced to 0. actual_reasons (per row, up to top 3 by downtime) are the real logged "
            "breakdown reason/grouping for that forecast_target_date -- ground truth, not a "
            "prediction; empty when no breakdown was logged that day, or none in this row's "
            "category. Each row's 9th/10th values are a PREDICTED reason group and its confidence "
            "(0-1) -- 'if a breakdown occurs on this target date, which failure-family group is it "
            "most likely to be', from a small, genuinely hard multiclass model "
            "(generate_reason_predictions.py). It is a best guess, not ground truth, and its "
            "test macro-F1 (reason_model_metrics, ~0.10-0.28) is close to chance for several "
            "classes -- shown with low-confidence styling in the UI."
        ),
    }
    data_json = json.dumps(bundle, separators=(",", ":"))

    with open(TEMPLATE) as f:
        html = f.read()
    assert html.count("__DASHBOARD_DATA_JSON__") == 1, "template placeholder missing or duplicated"
    html = html.replace("__DASHBOARD_DATA_JSON__", data_json)

    with open(OUTPUT, "w") as f:
        f.write(html)

    print(f"Wrote {OUTPUT}: {len(html)} bytes, {len(rows)} prediction rows, {len(metrics_rows)} metric rows")


if __name__ == "__main__":
    main()
