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
            ])

        rows.append([
            m_idx[r.machine_id], d_idx[r.feature_date], h_idx[r.forecast_horizon],
            c_idx[r.category], mo_idx[r.model_used], round(float(r.predicted_score), 4),
            actual, s_idx[r.split], reasons_enc,
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
        "generated_note": (
            "Predictions are real model outputs (train-only-fit models, scored on every row). "
            "Model selection used VALIDATION roc_auc only (test set never used for selection). "
            "Baseline/dummy strategies were excluded from selection eligibility. actual is null "
            "where the future day label_confidence is unknown_no_production_record -- never "
            "coerced to 0. actual_reasons (per row, up to top 3 by downtime) are the real logged "
            "breakdown reason/grouping for that forecast_target_date -- ground truth, not a "
            "prediction; empty when no breakdown was logged that day, or none in this row's "
            "category."
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
