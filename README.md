# VRM-1 / VRM-2 Breakdown Prediction Dashboard

**Live dashboard:** https://mahiya98.github.io/breakdown-dashboard/vrm-prediction/

Predicts the probability of an electrical or mechanical breakdown on VRM-1 / VRM-2, 1, 2, or 3 days
ahead, and shows the real logged breakdown reason (group + specific reason) alongside each
prediction wherever one actually occurred.

## Repository layout

| Path | Purpose |
| --- | --- |
| `vrm-prediction/index.html` | The published dashboard (this is the file GitHub Pages serves). |
| `pipeline/` | The full retrain pipeline: source workbook, feature engineering, target definition, model training, scoring, and dashboard build scripts. |
| `deploy_and_publish.sh` | One-command retrain + republish (run from the repo root). |

## Retraining and republishing

```bash
./deploy_and_publish.sh
```

This re-runs the full pipeline against `pipeline/VRM_Jul2025_Oct2026_Combined.xlsx`, rebuilds the
dashboard, copies it to `vrm-prediction/index.html`, and commits + pushes — only if the output
actually changed.

## Pipeline stages (run by `pipeline/run_pipeline.sh`)

1. `data_pipeline.py` — builds the feature matrix from the source workbook.
2. `run_target_definition.py` — builds the audited breakdown-reason taxonomy and per-day targets.
3. `make_forward_targets.py` — shifts targets forward to Day+1/2/3.
4. `run_modeling.py`, `run_modeling_electrical_mechanical.py` — trains and compares candidate models per category/horizon, selecting on validation ROC-AUC only.
5. `generate_dashboard_predictions.py` — refits the selected models and scores every row, joining in the real breakdown reason per day.
6. `generate_reason_predictions.py` — trains small multiclass models predicting the likely failure-family group for every row (low confidence; see dashboard note).
7. `build_dashboard.py` — compiles everything into `pipeline/dashboard.html`.

See `pipeline/README.md` for details on each script.
