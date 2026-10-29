# VRM Breakdown Prediction Pipeline

Full feature/target/model pipeline for the VRM-1/VRM-2 predictive maintenance dashboard
(the live page is at `../vrm-prediction/index.html`, served via GitHub Pages).

## To retrain with new data

1. Replace `VRM_Jul2025_Oct2026_Combined.xlsx` in this folder with the updated workbook
   (same filename).
2. From the repo root, run: `./deploy_and_publish.sh`
   This installs dependencies, re-runs every pipeline step in order, rebuilds the
   dashboard, and pushes it live -- only if the output actually changed.

## Pipeline order (see `run_pipeline.sh`)

1. `data_pipeline.py` -- rebuilds the feature matrix from the source workbook
2. `run_target_definition.py` -- rebuilds per-day breakdown labels
3. `make_forward_targets.py` -- builds Day+1/2/3 forward-shifted targets
4. `run_modeling.py` -- trains/evaluates the combined "any eligible breakdown" models
5. `run_modeling_electrical_mechanical.py` -- trains/evaluates electrical + mechanical models
6. `generate_dashboard_predictions.py` -- scores every row with the selected models
7. `build_dashboard.py` -- embeds the results into `dashboard.template.html` -> `dashboard.html`

## Design rules this pipeline follows (do not change without re-reading these)

- Chronological train/val/test split only -- never shuffled, never randomly sampled.
- Models are fit on the train split only; model selection uses validation ROC-AUC only,
  never the test split.
- A future day's label is only used when its `label_confidence` is `positive` or
  `confident_negative` -- `unknown_no_production_record` days are excluded from
  training/evaluation and reported as `actual: null`, never coerced to 0/1.
- No downtime duration is predicted anywhere -- only whether/what kind of breakdown.
- Baseline/dummy strategies (majority-class, training-frequency, stratified dummy) are
  evaluated for comparison but excluded from the model actually selected for serving in
  the dashboard, even when one scores higher on validation ROC-AUC than every real model
  (see `step10_electrical_mechanical_results_report.pdf` for why, and the full comparison).
