#!/usr/bin/env bash
# compare_hook.sh
#
# Runs the ibex/nangate45 flow twice from the same placement checkpoint:
#   1. Baseline: CTS through finish, no hook
#   2. Hook:     CTS with POST_CTS_TCL, then finish
# Then prints both pr_metrics.py tables side by side for comparison.
#
# Usage (from flow/):
#   util/compare_hook.sh
#
# All make targets run inside the Docker container via util/docker_shell.

set -euo pipefail

DESIGN_CONFIG=designs/nangate45/ibex/config.mk
HOOK=/work/scripts/post_cts_timing_repair.tcl
RESULTS=results/nangate45/ibex/base
REPORTS=reports/nangate45/ibex/base
LOGS=logs/nangate45/ibex/base
BASELINE_DIR=/tmp/ibex_hook_baseline_reports
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

clean_downstream() {
    rm -f "$RESULTS"/4_* "$RESULTS"/5_* "$RESULTS"/6_*
}

echo "======================================================"
echo " compare_hook.sh — ibex/nangate45 before/after"
echo "======================================================"

# --- Baseline run ---
echo ""
echo "[1/4] Cleaning CTS and downstream..."
clean_downstream

echo "[2/4] Running baseline (no hook) through finish..."
util/docker_shell make finish DESIGN_CONFIG="$DESIGN_CONFIG"

echo "      Saving baseline reports to $BASELINE_DIR"
rm -rf "$BASELINE_DIR"
cp -r "$REPORTS" "$BASELINE_DIR"

# --- Hook run ---
echo ""
echo "[3/4] Cleaning CTS and downstream..."
clean_downstream

echo "      Running CTS with POST_CTS_TCL hook..."
util/docker_shell make cts \
    DESIGN_CONFIG="$DESIGN_CONFIG" \
    POST_CTS_TCL="$HOOK"

echo "      Running route and finish..."
util/docker_shell make finish DESIGN_CONFIG="$DESIGN_CONFIG"

# --- Compare ---
echo ""
echo "[4/4] Results"
echo ""
echo "--- BASELINE (no hook) ---"
python3 "$SCRIPT_DIR/pr_metrics.py" \
    --reports-dir "$BASELINE_DIR" \
    --logs-dir    "$LOGS"

echo "--- WITH HOOK ---"
python3 "$SCRIPT_DIR/pr_metrics.py" \
    --platform nangate45 --design ibex --tag base
