#!/usr/bin/env bash
# Full retrain + republish, run from the repo root.
# 1) Re-runs the entire modeling pipeline against pipeline/VRM_Jul2025_Oct2026_Combined.xlsx
# 2) Rebuilds the dashboard
# 3) Copies it to vrm-prediction/index.html (the live GitHub Pages path)
# 4) Commits and pushes, only if something actually changed
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Installing dependencies ==="
pip install -r pipeline/requirements.txt --break-system-packages --quiet

echo "=== Running pipeline ==="
(cd pipeline && ./run_pipeline.sh)

echo "=== Publishing dashboard ==="
cp pipeline/dashboard.html vrm-prediction/index.html

echo "=== Committing ==="
git add vrm-prediction/index.html pipeline/dashboard_predictions.csv pipeline/dashboard_accuracy_metrics.csv \
        pipeline/model_comparison_results.csv pipeline/step10_electrical_mechanical_results.csv 2>/dev/null || \
git add vrm-prediction/index.html

if git diff --cached --quiet; then
  echo "No changes detected -- data/model outputs are identical to last run. Nothing to publish."
  exit 0
fi

git -c user.email="pipeline@vrm-breakdown.local" -c user.name="VRM Pipeline Bot" commit -m "$(cat <<'EOF'
Automated retrain: refresh dashboard with latest VRM data

Re-ran the full feature/target/model pipeline against the current
source workbook and republished the dashboard.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
git push origin main

echo "=== Done. Live at https://mahiya98.github.io/breakdown-dashboard/vrm-prediction/ (allow a minute to propagate) ==="
