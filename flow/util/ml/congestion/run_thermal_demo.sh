#!/usr/bin/env bash
# End-to-end demo of the thermal-prediction track: test, train, visualise, predict.
#
# Usage (from anywhere; the script moves itself to flow/):
#   bash util/ml/congestion/run_thermal_demo.sh [OPTIONS]
#
# Options:
#   --epochs N     Training epochs (default: 100)
#   --extract      If no thermal labels exist yet, run the Docker extraction
#                  first (needs routed designs under results/ and the
#                  openroad/orfs-ml:latest image; this is the slow step)
#   --no-open      Do not try to open the HTML report in a browser
#   -h, --help     Show this help
#
# What it does:
#   1. Runs the model smoke tests (synthetic data, a few seconds)
#   2. Checks that extracted features + thermal labels exist in data/
#   3. Trains the thermal U-Net, saving checkpoints/thermal_best.pt
#   4. Writes a self-contained HTML report comparing predicted and
#      ground-truth thermal maps for every design
#   5. Runs single-design inference on the first design as a usage example
#
# Everything it writes (checkpoints/, experiments/, data/*.npz) is gitignored.

set -euo pipefail
cd "$(dirname "$0")/../../.."   # always run from flow/

EPOCHS=100
EXTRACT=false
OPEN=true

usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)  EPOCHS="$2"; shift 2 ;;
        --extract) EXTRACT=true; shift ;;
        --no-open) OPEN=false; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

ML_DIR="util/ml/congestion"
DATA_DIR="$ML_DIR/data"
CKPT_DIR="$ML_DIR/checkpoints"
OUT_DIR="$ML_DIR/experiments"
REPORT="$OUT_DIR/thermal_report.html"

# Number of designs that have both features and thermal labels.
count_designs() {
    local n=0 f
    for f in "$DATA_DIR"/*_features.npz; do
        [[ -e "$f" ]] || continue
        [[ -e "${f%_features.npz}_thermal_labels.npz" ]] && n=$((n + 1))
    done
    echo "$n"
}

echo "============================================================"
echo " Thermal prediction demo"
echo "============================================================"

echo ""
echo "[1/5] Running model smoke tests..."
TEST_LOG=$(mktemp)
if python3 "$ML_DIR/tests/test_models.py" -v >"$TEST_LOG" 2>&1; then
    grep -E "^Ran " "$TEST_LOG" | sed 's/^/      /'
    echo "      All tests passed."
else
    cat "$TEST_LOG" >&2
    echo "ERROR: smoke tests failed (output above)." >&2
    rm -f "$TEST_LOG"
    exit 1
fi
rm -f "$TEST_LOG"

echo ""
echo "[2/5] Checking for extracted data in $DATA_DIR ..."
if [[ $(count_designs) -eq 0 ]]; then
    if $EXTRACT; then
        echo "      None found; running extraction (this can take a long time)."
        OR_IMAGE=openroad/orfs-ml:latest \
            bash "$ML_DIR/data_collection/extract_thermal_batch.sh" </dev/null
    else
        echo "ERROR: no designs with both *_features.npz and *_thermal_labels.npz." >&2
        echo "       Re-run with --extract (needs routed designs under results/" >&2
        echo "       and Docker), or copy .npz files into $DATA_DIR." >&2
        exit 1
    fi
fi
N=$(count_designs)
echo "      Found $N design(s) with features and thermal labels."
if [[ $N -lt 4 ]]; then
    echo "      Warning: with only $N design(s) the train/val/test split is tiny;"
    echo "      the demo will run but the numbers are not meaningful."
fi

echo ""
echo "[3/5] Training the thermal U-Net ($EPOCHS epochs)..."
mkdir -p "$CKPT_DIR" "$OUT_DIR"
python3 "$ML_DIR/training/train_thermal.py" \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --epochs "$EPOCHS"

echo ""
echo "[4/5] Writing the HTML report..."
python3 "$ML_DIR/inference/visualize_thermal.py" \
    --data-dir "$DATA_DIR" \
    --checkpoint "$CKPT_DIR/thermal_best.pt" \
    --out "$REPORT"
echo "      Report: $REPORT"

echo ""
echo "[5/5] Single-design inference example..."
FIRST=$(ls "$DATA_DIR"/*_features.npz | head -1)
python3 "$ML_DIR/inference/predict_thermal.py" \
    --features "$FIRST" \
    --checkpoint "$CKPT_DIR/thermal_best.pt" \
    --out "$OUT_DIR/predicted_thermal.npz"
python3 - "$OUT_DIR/predicted_thermal.npz" "$FIRST" <<'PY'
import sys
import numpy as np

pred = np.load(sys.argv[1])["thermal_pred_norm"]
print(f"      Input:  {sys.argv[2]}")
print(f"      Output: predicted map {pred.shape}, "
      f"relative range [{pred.min():.3f}, {pred.max():.3f}] (1.0 = hottest point)")
PY

echo ""
echo "============================================================"
echo " Done."
echo "   Checkpoint: $CKPT_DIR/thermal_best.pt"
echo "   Report:     $REPORT"
echo "============================================================"

if $OPEN; then
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$REPORT" >/dev/null 2>&1 || true
    fi
fi
