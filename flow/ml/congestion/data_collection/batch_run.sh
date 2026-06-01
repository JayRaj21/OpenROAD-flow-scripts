#!/usr/bin/env bash
# Run all built-in ORFS designs through placement + global routing,
# then extract placement grids and congestion vectors for ML training.
#
# Usage (from repo root):
#   bash flow/ml/congestion/data_collection/batch_run.sh
#
# Outputs land in flow/ml/data/:
#   <platform>_<design>_placement.npy   -- 2D cell density grid
#   <platform>_<design>_congestion.npy  -- per-layer congestion vector

set -e

FLOW_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="$FLOW_DIR/ml/data"
EXTRACT_CONG="$FLOW_DIR/ml/congestion/data_collection/extract_congestion.py"
GRID_SIZE=64

mkdir -p "$DATA_DIR"

# Designs to run: "platform design" pairs
# Skip very large designs (black_parrot, bp_quad) by default to save time
DESIGNS=(
    "nangate45 gcd"
    "nangate45 aes"
    "nangate45 ibex"
    "nangate45 jpeg"
    "nangate45 swerv"
    "nangate45 tinyRocket"
    "nangate45 dynamic_node"
    "sky130hd gcd"
    "sky130hd aes"
    "sky130hd ibex"
    "sky130hd jpeg"
    "sky130hd riscv32i"
)

run_design() {
    local platform="$1"
    local design="$2"
    local tag="${platform}_${design}"
    local place_out="$DATA_DIR/${tag}_placement.npy"
    local cong_out="$DATA_DIR/${tag}_congestion.npy"

    if [[ -f "$place_out" && -f "$cong_out" ]]; then
        echo "[SKIP] $tag — data already exists"
        return 0
    fi

    echo ""
    echo "=============================="
    echo " Running: $tag"
    echo "=============================="

    local config="/work/designs/${platform}/${design}/config.mk"
    local odb_place="/work/results/${platform}/${design}/base/3_5_place_dp.odb"

    # Run through detailed placement
    cd "$FLOW_DIR"
    util/docker_shell make \
        DESIGN_CONFIG="$config" \
        DESIGN_HOME=/work/designs \
        place 2>&1 | tail -5

    # Run global routing to get congestion data
    util/docker_shell make \
        DESIGN_CONFIG="$config" \
        DESIGN_HOME=/work/designs \
        grt 2>&1 | tail -5

    # Extract placement grid (runs inside Docker, OpenROAD Python API required)
    util/docker_shell openroad -python /work/ml/congestion/data_collection/extract_placement.py \
        --odb "$odb_place" \
        --out "/work/ml/data/${tag}_placement.npy" \
        --grid "$GRID_SIZE"

    # Extract congestion vector from GRT log (pure Python, runs on host)
    python3 "$EXTRACT_CONG" \
        --log "$FLOW_DIR/logs/${platform}/${design}/base/5_1_grt.log" \
        --out "$cong_out"

    echo "[DONE] $tag"
}

for entry in "${DESIGNS[@]}"; do
    platform="${entry%% *}"
    design="${entry##* }"
    run_design "$platform" "$design" || echo "[ERROR] $platform/$design failed, continuing..."
done

echo ""
echo "=============================="
echo " Batch run complete"
echo " Data files in: $DATA_DIR"
ls "$DATA_DIR"/*.npy 2>/dev/null | wc -l | xargs echo " Total .npy files:"
echo "=============================="
