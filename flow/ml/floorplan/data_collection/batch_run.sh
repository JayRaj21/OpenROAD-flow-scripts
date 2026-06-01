#!/usr/bin/env bash
# Generate floorplan training data from all macro-containing designs.
#
# Usage (from flow/ directory):
#   bash ml/floorplan/data_collection/batch_run.sh

set -e

FLOW_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="$FLOW_DIR/ml/data"

mkdir -p "$DATA_DIR"

# "platform design nickname" — nickname is the DESIGN_NICKNAME from config.mk,
# which determines the results/ subdirectory name (may differ from design folder name)
DESIGNS=(
    "nangate45 ariane133 ariane133"
    "nangate45 ariane136 ariane136"
    "nangate45 black_parrot bp"
    "nangate45 bp_be_top bp_be"
    "nangate45 bp_fe_top bp_fe"
    "nangate45 mempool_group mempool_group"
    "sky130hd microwatt microwatt"
)

run_design() {
    local platform="$1"
    local design="$2"
    local nickname="$3"
    local tag="${platform}_${design}"
    local graph_out="$DATA_DIR/${tag}_graph.npz"
    local fp_out="$DATA_DIR/${tag}_floorplan.npz"

    if [[ -f "$graph_out" && -f "$fp_out" ]]; then
        echo "[SKIP] $tag — data already exists"
        return 0
    fi

    echo ""
    echo "=============================="
    echo " Running: $tag"
    echo "=============================="

    local config="/work/designs/${platform}/${design}/config.mk"
    # ODB paths use the nickname (DESIGN_NICKNAME), not the design folder name
    local odb_synth="/work/results/${platform}/${nickname}/base/1_synth.odb"
    local odb_fp="/work/results/${platform}/${nickname}/base/2_floorplan.odb"

    cd "$FLOW_DIR"

    # Run through floorplan (includes synthesis)
    util/docker_shell make \
        DESIGN_CONFIG="$config" \
        DESIGN_HOME=/work/designs \
        floorplan 2>&1 | tail -5

    # Extract netlist graph from synthesis ODB
    util/docker_shell openroad -python /work/ml/floorplan/data_collection/extract_netlist_graph.py \
        --odb "$odb_synth" \
        --out "/work/ml/data/${tag}_graph.npz"

    # Extract macro positions from floorplan ODB
    util/docker_shell openroad -python /work/ml/floorplan/data_collection/extract_floorplan.py \
        --odb "$odb_fp" \
        --out "/work/ml/data/${tag}_floorplan.npz"

    echo "[DONE] $tag"
}

for entry in "${DESIGNS[@]}"; do
    read -r platform design nickname <<< "$entry"
    run_design "$platform" "$design" "$nickname" || echo "[ERROR] $platform/$design failed, continuing..."
done

echo ""
echo "=============================="
echo " Batch run complete"
ls "$DATA_DIR"/*_graph.npz 2>/dev/null | wc -l | xargs echo " Graph files:"
ls "$DATA_DIR"/*_floorplan.npz 2>/dev/null | wc -l | xargs echo " Floorplan files:"
echo "=============================="
