#!/usr/bin/env bash
# Full retrain pipeline, in order. Run this after replacing VRM_Jul2025_Oct2026_Combined.xlsx
# with updated data. Each step's output feeds the next; stops on first failure (set -e).
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/6] Rebuilding feature matrix from source workbook..."
python3 data_pipeline.py

echo "[2/6] Rebuilding per-day breakdown targets..."
python3 run_target_definition.py

echo "[3/6] Rebuilding Day+1/2/3 forward targets..."
python3 make_forward_targets.py

echo "[4/6] Retraining/evaluating combined 'any eligible breakdown' models..."
python3 run_modeling.py

echo "[5/6] Retraining/evaluating electrical + mechanical models..."
python3 run_modeling_electrical_mechanical.py

echo "[6/6] Re-scoring every row and rebuilding dashboard..."
python3 generate_dashboard_predictions.py
python3 build_dashboard.py

echo "Pipeline complete. Updated: dashboard.html, dashboard_predictions.csv, dashboard_accuracy_metrics.csv"
