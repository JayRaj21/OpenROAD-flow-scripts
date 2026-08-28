#!/usr/bin/env bash
# Batch IR-drop (and placement feature) extraction from existing ORFS result dirs.
#
# Iterates every 6_final.odb already on disk (routed, RCX-extracted stage)
# with a matching 6_final.spef and runs:
#   1. extract_features.py         -> util/ml/congestion/data/<label>_features.npz
#   2. extract_irdrop_labels.py    -> util/ml/congestion/data/<label>_irdrop_labels.npz
#
# Unlike extract_thermal_batch.sh, IR-drop extraction uses OpenROAD's native
# PDNSim (analyze_power_grid), so the *stock* openroad/orfs:latest image is
# enough — no HotSpot / -ml suffix image is needed.
#
# Usage (from flow/):
#   export OR_IMAGE=openroad/orfs:latest
#   bash util/ml/congestion/data_collection/extract_irdrop_batch.sh [--timeout 600] [--force] \
#       [--net VDD]
#
# --force  : re-extract IR-drop labels even if they already exist (use after
#            changing extract_irdrop_labels.py)
#
# Already-extracted files are skipped (idempotent).
# Each design is given TIMEOUT_S seconds per extractor (default 3600).
#
# Liberty and nominal voltage are resolved **per design**, not per platform,
# via `make DESIGN_CONFIG=designs/<platform>/<design>/config.mk print-X`
# (variables.mk's generic `print-%` target) rather than by grepping the
# platform config.mk directly. This matters because:
#   - asap7's LIB_FILES is built from corner/VT-placeholder-substituted make
#     variables ($($(CORNER)_$(LIB_MODEL)_LIB_FILES)), not a plain
#     "export LIB_FILES = ..." line — a text-parsing approach silently
#     resolves to nothing or garbage on asap7. `make print-LIB_FILES`
#     resolves it the same way the real flow does.
#   - Nominal supply voltage varies by platform (nangate45 1.1V, sky130hd
#     1.8V, asap7 0.77V) and is not a fixed CLI default — resolved from
#     `print-PWR_NETS_VOLTAGES` (a universal "<net> <voltage>" dict format
#     also used by scripts/final_outputs.tcl), matched against --net.
# Designs whose config can't be resolved are skipped with a warning rather
# than silently analyzed against the wrong supply voltage.

set -euo pipefail
cd "$(dirname "$0")/../../.."   # -> flow/

TIMEOUT_S=3600
FORCE=0
NET="VDD"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout) TIMEOUT_S="$2"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        --net)     NET="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

DATA_DIR="/work/util/ml/congestion/data"
FEAT_SCRIPT="/work/util/ml/congestion/data_collection/extract_features.py"
IRDROP_SCRIPT="/work/util/ml/congestion/data_collection/extract_irdrop_labels.py"

pass=0; fail=0; skip=0

while IFS= read -r HOST_ODB; do
    # HOST_ODB is a relative path from flow/: results/<platform>/<design>/<tag>/6_final.odb
    rel="${HOST_ODB#results/}"                 # platform/design/tag/6_final.odb
    rel="${rel%/6_final.odb}"                  # platform/design/tag
    label="$(echo "$rel" | tr '/' '_')"        # platform_design_tag
    platform="$(echo "$rel" | cut -d/ -f1)"
    design="$(echo "$rel" | cut -d/ -f2)"

    host_spef="results/${rel}/6_final.spef"
    if [ ! -f "$host_spef" ]; then
        echo "Design: ${label}  [SKIP] no 6_final.spef (RCX not enabled for this platform?)"
        ((skip++)) || true
        continue
    fi

    design_cfg="designs/${platform}/${design}/config.mk"
    if [ ! -f "$design_cfg" ]; then
        echo "Design: ${label}  [SKIP] no design config at ${design_cfg}"
        ((skip++)) || true
        continue
    fi

    cont_odb="/work/results/${rel}/6_final.odb"
    cont_spef="/work/results/${rel}/6_final.spef"

    feat_out="${DATA_DIR}/${label}_features.npz"
    irdrop_out="${DATA_DIR}/${label}_irdrop_labels.npz"

    feat_host="util/ml/congestion/data/${label}_features.npz"
    irdrop_host="util/ml/congestion/data/${label}_irdrop_labels.npz"

    echo "========================================="
    echo "Design: ${label}"

    # -- Features --------------------------------------------------------
    if [ -f "$feat_host" ]; then
        echo "  [SKIP] features already extracted"
        ((skip++)) || true
    else
        echo "  [RUN]  extract_features.py (timeout ${TIMEOUT_S}s)"
        if timeout "$TIMEOUT_S" util/docker_shell openroad -python "$FEAT_SCRIPT" \
               --odb "$cont_odb" --out "$feat_out" </dev/null; then
            echo "  [OK]   ${label}_features.npz"
            ((pass++)) || true
        else
            ec=$?
            if [ $ec -eq 124 ]; then
                echo "  [TIMEOUT] features extraction exceeded ${TIMEOUT_S}s — skipping"
            else
                echo "  [FAIL]   features extraction failed (exit $ec)"
            fi
            ((fail++)) || true
        fi
    fi

    # -- IR-drop labels ----------------------------------------------------
    if [ -f "$irdrop_host" ] && [ "$FORCE" -eq 0 ]; then
        echo "  [SKIP] IR-drop labels already extracted"
        ((skip++)) || true
    else
        # docker_shell's `docker run -i` reads from stdin until EOF; without
        # </dev/null here it drains the outer `while read` loop's process
        # substitution after the first iteration, silently ending the batch
        # after one design.
        if ! make_out="$(util/docker_shell -- "make DESIGN_CONFIG=./${design_cfg} print-LIB_FILES print-PWR_NETS_VOLTAGES" 2>/dev/null </dev/null)"; then
            echo "  [SKIP] 'make print-LIB_FILES print-PWR_NETS_VOLTAGES' failed for ${design_cfg}"
            ((skip++)) || true
            continue
        fi
        lib_files="$(echo "$make_out" | sed -n 's/^LIB_FILES: *//p')"
        pwr_nets_voltages="$(echo "$make_out" | sed -n 's/^PWR_NETS_VOLTAGES: *//p')"

        if [ -z "$lib_files" ]; then
            echo "  [SKIP] LIB_FILES did not resolve via 'make print-LIB_FILES' for ${design_cfg}"
            ((skip++)) || true
            continue
        fi
        # LIB_FILES paths from `make print-` are absolute host paths already
        # (make resolved $(PLATFORM_DIR) etc for real) — just remap the flow/
        # prefix to the container's /work mount.
        cont_libs=""
        for lib in $lib_files; do
            cont_libs="${cont_libs} /work/${lib#*/flow/}"
        done

        # PWR_NETS_VOLTAGES is a "<net1> <voltage1> <net2> <voltage2> ..." dict
        # (same format scripts/final_outputs.tcl parses) — find the voltage for
        # the requested --net.
        voltage=""
        set -- $pwr_nets_voltages
        while [ "$#" -ge 2 ]; do
            if [ "$1" = "$NET" ]; then voltage="$2"; fi
            shift 2
        done
        if [ -z "$voltage" ]; then
            echo "  [SKIP] no voltage for net ${NET} in PWR_NETS_VOLTAGES ('${pwr_nets_voltages}')"
            ((skip++)) || true
            continue
        fi

        echo "  [RUN]  extract_irdrop_labels.py (timeout ${TIMEOUT_S}s)"
        if timeout "$TIMEOUT_S" util/docker_shell openroad -python "$IRDROP_SCRIPT" \
               --odb "$cont_odb" --spef "$cont_spef" --liberty $cont_libs \
               --net "$NET" --voltage "$voltage" --out "$irdrop_out" </dev/null; then
            echo "  [OK]   ${label}_irdrop_labels.npz"
            ((pass++)) || true
        else
            ec=$?
            if [ $ec -eq 124 ]; then
                echo "  [TIMEOUT] IR-drop extraction exceeded ${TIMEOUT_S}s — skipping"
            else
                echo "  [FAIL]   IR-drop extraction failed (exit $ec)"
            fi
            ((fail++)) || true
        fi
    fi

done < <(find results -name "6_final.odb" | sort)

echo ""
echo "========================================="
echo "Done.  passed=${pass}  failed=${fail}  skipped=${skip}"
