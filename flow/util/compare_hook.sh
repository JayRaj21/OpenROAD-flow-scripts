#!/usr/bin/env bash
# compare_hook.sh
#
# Runs a design twice from the same placement checkpoint:
#   1. Baseline: CTS through finish, no hook
#   2. Hook:     CTS with POST_CTS_TCL, then finish
# Then prints both pr_metrics.py tables side by side for comparison.
#
# Usage (from flow/):
#   util/compare_hook.sh --platform nangate45 --design ibex
#   util/compare_hook.sh --platform nangate45 --design aes --tag base
#
# All make targets run inside the Docker container via util/docker_shell.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PLATFORM=nangate45
DESIGN=ibex
TAG=base
CTS_HOOK=/work/scripts/post_cts_timing_repair.tcl
GRT_HOOK=/work/scripts/post_grt_timing_repair.tcl

usage() {
    cat <<EOF
Usage: util/compare_hook.sh --platform <platform> --design <design> [options]

Options:
  --tag <tag>          Flow tag (default: base)
  --cts-hook <path>    Container path to POST_CTS hook   (default: $CTS_HOOK)
  --grt-hook <path>    Container path to POST_GRT hook   (default: $GRT_HOOK)
  --no-cts-hook        Disable the POST_CTS hook
  --no-grt-hook        Disable the POST_GRT hook
EOF
    exit 1
}

USE_CTS_HOOK=1
USE_GRT_HOOK=1

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)     PLATFORM="$2";  shift 2 ;;
        --design)       DESIGN="$2";    shift 2 ;;
        --tag)          TAG="$2";       shift 2 ;;
        --cts-hook)     CTS_HOOK="$2";  shift 2 ;;
        --grt-hook)     GRT_HOOK="$2";  shift 2 ;;
        --no-cts-hook)  USE_CTS_HOOK=0; shift ;;
        --no-grt-hook)  USE_GRT_HOOK=0; shift ;;
        -h|--help)      usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ -z "$PLATFORM" || -z "$DESIGN" ]]; then
    echo "ERROR: --platform and --design are required."
    usage
fi

DESIGN_CONFIG=designs/$PLATFORM/$DESIGN/config.mk
RESULTS=results/$PLATFORM/$DESIGN/$TAG
REPORTS=reports/$PLATFORM/$DESIGN/$TAG
LOGS=logs/$PLATFORM/$DESIGN/$TAG
BASELINE_DIR=/tmp/${PLATFORM}_${DESIGN}_${TAG}_hook_baseline

clean_downstream() {
    rm -f "$RESULTS"/4_* "$RESULTS"/5_* "$RESULTS"/6_*
}

echo "======================================================"
echo " compare_hook.sh — $PLATFORM/$DESIGN/$TAG before/after"
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

echo "      Running CTS$([ "$USE_CTS_HOOK" = 1 ] && echo " with POST_CTS hook" || echo " (no CTS hook)")..."
if [[ "$USE_CTS_HOOK" = 1 ]]; then
    util/docker_shell make cts \
        DESIGN_CONFIG="$DESIGN_CONFIG" \
        POST_CTS_TCL="$CTS_HOOK"
else
    util/docker_shell make cts DESIGN_CONFIG="$DESIGN_CONFIG"
fi

echo "      Running route and finish$([ "$USE_GRT_HOOK" = 1 ] && echo " with POST_GRT hook" || echo "")..."
if [[ "$USE_GRT_HOOK" = 1 ]]; then
    util/docker_shell make finish \
        DESIGN_CONFIG="$DESIGN_CONFIG" \
        POST_GLOBAL_ROUTE_TCL="$GRT_HOOK"
else
    util/docker_shell make finish DESIGN_CONFIG="$DESIGN_CONFIG"
fi

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
    --platform "$PLATFORM" --design "$DESIGN" --tag "$TAG"
