#!/usr/bin/env bash
# run_pipeline.sh — GAPC full pipeline (run on VPS)
# Usage: bash run_pipeline.sh [--start STEP]

set -euo pipefail

VENV="${HOME}/gasp/.venv"
GAPC_DIR="${HOME}/gapc"

source "$VENV/bin/activate"
cd "$GAPC_DIR"

START_STEP="${2:-1}"

run_step() {
    local step="$1"
    local script="$2"
    if [ "$step" -ge "$START_STEP" ]; then
        echo ""
        echo "══════════════════════════════════════"
        echo "  Step ${step}: ${script}"
        echo "══════════════════════════════════════"
        PYTHONUNBUFFERED=1 python "pipeline/${script}" \
            2>&1 | tee "logs/$(printf '%02d' $step)_$(basename ${script} .py).log"
    fi
}

mkdir -p logs

run_step 1 "01_verify_setup.py"
run_step 2 "02_download_sso.py"
run_step 3 "03_filter_quality.py"
run_step 4 "04_fit_hg1g2.py"
run_step 5 "05_crossmatch_gasp.py"
run_step 6 "06_validate.py"
run_step 7 "compute_stats.py"

echo ""
echo "✅  Pipeline complete."
