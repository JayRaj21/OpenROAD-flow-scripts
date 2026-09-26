#!/usr/bin/env bash
# Extract placement features, thermal labels and placement QoR for the
# density variants produced by knob_variants.sh.
#
# A parameterised copy of data_collection/extract_thermal_batch.sh: same
# skip / timeout / </dev/null contract, but outputs go to the gitignored
# experiments/thermal_loop/data/ instead of util/ml/congestion/data/. Variant
# files must never be written to data/: ThermalDataset globs that directory
# and laplacian_sweep assumes a `_base` suffix, so they would silently
# pollute training and break every existing LODO result.
#
# Usage (from flow/):
#   OR_IMAGE=openroad/orfs-ml:latest bash util/ml/congestion/loop/extract_variant_labels.sh \
#       --design sky130hd/riscv32i [--tags dn_005,dn_015,...] [--timeout 3600]
#
# --tags     comma-separated variant tags (default: every dn_* dir of the
#            design that has a 3_place.odb)
# --timeout  per-extractor timeout in seconds (default 3600)
#
# Writes <pdk>_<design>_<tag>_{features,thermal_labels}.npz and
# <pdk>_<design>_<tag>_qor.json. Existing files are skipped (idempotent).
# Every thermal_labels.npz is checked after extraction (thermal_metrics.py
# --check-thermal-npz: keys, 64x64, finite, ptp > 1e-3 C); a file that fails
# is deleted and counted as failed, so it is re-extracted on the next run.

set -euo pipefail
cd "$(dirname "$0")/../../../.."   # → flow/

TIMEOUT_S=3600
DESIGN=""
TAGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --design)  DESIGN="$2"; shift 2 ;;
        --tags)    TAGS="$2"; shift 2 ;;
        --timeout) TIMEOUT_S="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$DESIGN" ]; then
    echo "Usage: $0 --design <pdk>/<design> [--tags a,b] [--timeout N]" >&2
    exit 1
fi

OUT_REL="util/ml/congestion/experiments/thermal_loop/data"
DATA_DIR="/work/${OUT_REL}"
FEAT_SCRIPT="/work/util/ml/congestion/data_collection/extract_features.py"
THERM_SCRIPT="/work/util/ml/congestion/data_collection/extract_thermal_labels.py"
QOR_SCRIPT="/work/util/ml/congestion/loop/placement_qor.py"
METRICS_SCRIPT="util/ml/congestion/loop/thermal_metrics.py"

if [ "$(realpath -m "$OUT_REL")" = "$(realpath -m util/ml/congestion/data)" ]; then
    echo "Refusing to write variant files to util/ml/congestion/data (ThermalDataset globs it)" >&2
    exit 1
fi

mkdir -p "$OUT_REL"

if [ -n "$TAGS" ]; then
    IFS=',' read -ra tag_list <<< "$TAGS"
else
    tag_list=()
    while IFS= read -r odb; do
        tag="${odb%/3_place.odb}"
        tag_list+=("${tag##*/}")
    done < <(find "results/${DESIGN}" -path "*/dn_*/3_place.odb" | sort)
fi

if [ "${#tag_list[@]}" -eq 0 ]; then
    echo "No dn_* variants with a 3_place.odb found for ${DESIGN}" >&2
    exit 1
fi

pass=0; fail=0; skip=0

check_thermal_output() {
    python3 "$METRICS_SCRIPT" --check-thermal-npz "$1"
}

# run_extractor <name> <host-output> <check-fn|-> <cmd...>
# Skips if <host-output> exists. After a zero exit, <check-fn> <host-output>
# must also succeed, otherwise the output is deleted and counted as failed.
run_extractor() {
    local name="$1" host_out="$2" check="$3"
    shift 3
    if [ -f "$host_out" ]; then
        echo "  [SKIP] ${name} already extracted"
        ((skip++)) || true
        return
    fi
    echo "  [RUN]  ${name} (timeout ${TIMEOUT_S}s)"
    local ec=0
    timeout "$TIMEOUT_S" "$@" </dev/null || ec=$?
    if [ "$ec" -eq 0 ] && [ "$check" != "-" ] && ! "$check" "$host_out"; then
        echo "  [FAIL]   ${name} wrote an unusable ${host_out} — deleting it"
        rm -f "$host_out"
        ((fail++)) || true
    elif [ "$ec" -eq 0 ]; then
        echo "  [OK]   ${host_out}"
        ((pass++)) || true
    elif [ "$ec" -eq 124 ]; then
        # A killed run can leave a half-written output that the skip check
        # above would otherwise treat as done on the next run.
        echo "  [TIMEOUT] ${name} exceeded ${TIMEOUT_S}s — skipping"
        rm -f "$host_out"
        ((fail++)) || true
    else
        echo "  [FAIL]   ${name} failed (exit ${ec})"
        rm -f "$host_out"
        ((fail++)) || true
    fi
}

for tag in "${tag_list[@]}"; do
    host_odb="results/${DESIGN}/${tag}/3_place.odb"
    label="$(echo "${DESIGN}/${tag}" | tr '/' '_')"
    cont_odb="/work/${host_odb}"

    echo "========================================="
    echo "Design: ${label}"

    if [ ! -f "$host_odb" ]; then
        echo "  [FAIL]   ${host_odb} not found"
        ((fail++)) || true
        continue
    fi

    run_extractor extract_features.py "${OUT_REL}/${label}_features.npz" - \
        util/docker_shell openroad -python "$FEAT_SCRIPT" \
        --odb "$cont_odb" --out "${DATA_DIR}/${label}_features.npz"

    run_extractor extract_thermal_labels.py "${OUT_REL}/${label}_thermal_labels.npz" check_thermal_output \
        util/docker_shell openroad -python "$THERM_SCRIPT" \
        --odb "$cont_odb" --out "${DATA_DIR}/${label}_thermal_labels.npz"

    run_extractor placement_qor.py "${OUT_REL}/${label}_qor.json" - \
        util/docker_shell openroad -python "$QOR_SCRIPT" \
        --odb "$cont_odb" --out "${DATA_DIR}/${label}_qor.json"
done

echo ""
echo "========================================="
echo "Done.  passed=${pass}  failed=${fail}  skipped=${skip}"

# A driver must be able to detect failure from the exit status.
[ "$fail" -eq 0 ] || exit 1
