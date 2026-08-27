#!/usr/bin/env bash
# Batch IR-drop (and placement feature) extraction from existing ORFS result dirs.
#
# Iterates every 6_final.odb already on disk (routed, RCX-extracted stage)
# with a matching 6_final.spef and runs:
#   1. extract_features.py         -> ml/congestion/data/<label>_features.npz
#   2. extract_irdrop_labels.py    -> ml/congestion/data/<label>_irdrop_labels.npz
#
# Unlike extract_thermal_batch.sh, IR-drop extraction uses OpenROAD's native
# PDNSim (analyze_power_grid), so the *stock* openroad/orfs:latest image is
# enough — no HotSpot / -ml suffix image is needed.
#
# Usage (from flow/):
#   export OR_IMAGE=openroad/orfs:latest
#   bash ml/congestion/data_collection/extract_irdrop_batch.sh [--timeout 600] [--force] \
#       [--net VDD] [--voltage 1.1]
#
# --force  : re-extract IR-drop labels even if they already exist (use after
#            changing extract_irdrop_labels.py)
#
# Already-extracted files are skipped (idempotent).
# Each design is given TIMEOUT_S seconds per extractor (default 3600).
# Liberty is resolved per-design from the platform's config.mk (LIB_FILES);
# designs whose platform config can't be found are skipped with a warning.

set -euo pipefail
cd "$(dirname "$0")/../../.."   # -> flow/

TIMEOUT_S=3600
FORCE=0
NET="VDD"
VOLTAGE="1.1"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout) TIMEOUT_S="$2"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        --net)     NET="$2"; shift 2 ;;
        --voltage) VOLTAGE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

DATA_DIR="/work/ml/congestion/data"
FEAT_SCRIPT="/work/ml/congestion/data_collection/extract_features.py"
IRDROP_SCRIPT="/work/ml/congestion/data_collection/extract_irdrop_labels.py"

pass=0; fail=0; skip=0

while IFS= read -r HOST_ODB; do
    # HOST_ODB is a relative path from flow/: results/<platform>/<design>/<tag>/6_final.odb
    rel="${HOST_ODB#results/}"                 # platform/design/tag/6_final.odb
    rel="${rel%/6_final.odb}"                  # platform/design/tag
    label="$(echo "$rel" | tr '/' '_')"        # platform_design_tag
    platform="$(echo "$rel" | cut -d/ -f1)"

    host_spef="results/${rel}/6_final.spef"
    if [ ! -f "$host_spef" ]; then
        echo "Design: ${label}  [SKIP] no 6_final.spef (RCX not enabled for this platform?)"
        ((skip++)) || true
        continue
    fi

    platform_cfg="platforms/${platform}/config.mk"
    if [ ! -f "$platform_cfg" ]; then
        echo "Design: ${label}  [SKIP] no platform config at ${platform_cfg}"
        ((skip++)) || true
        continue
    fi
    # LIB_FILES may be a multi-line "export LIB_FILES = a \\\n  b \\\n  c" block;
    # collapse continuations and strip the "export LIB_FILES =" prefix.
    lib_files="$(awk '/^export LIB_FILES/{flag=1} flag{print; if ($0 !~ /\\$/) flag=0}' "$platform_cfg" \
        | tr -d '\\' | sed 's/^export LIB_FILES *= *//' | tr '\n' ' ')"
    if [ -z "$lib_files" ]; then
        echo "Design: ${label}  [SKIP] LIB_FILES not found in ${platform_cfg}"
        ((skip++)) || true
        continue
    fi
    # Container paths: platform config paths are relative to flow/ already
    # (e.g. $(PLATFORM_DIR)/lib/...) but may retain the make variable — fall
    # back to a glob under platforms/<platform>/lib if substitution left junk.
    cont_libs=""
    for lib in $lib_files; do
        lib="${lib/\$(PLATFORM_DIR)/platforms\/${platform}}"
        cont_libs="${cont_libs} /work/${lib}"
    done

    cont_odb="/work/results/${rel}/6_final.odb"
    cont_spef="/work/results/${rel}/6_final.spef"

    feat_out="${DATA_DIR}/${label}_features.npz"
    irdrop_out="${DATA_DIR}/${label}_irdrop_labels.npz"

    feat_host="ml/congestion/data/${label}_features.npz"
    irdrop_host="ml/congestion/data/${label}_irdrop_labels.npz"

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
        echo "  [RUN]  extract_irdrop_labels.py (timeout ${TIMEOUT_S}s)"
        if timeout "$TIMEOUT_S" util/docker_shell openroad -python "$IRDROP_SCRIPT" \
               --odb "$cont_odb" --spef "$cont_spef" --liberty $cont_libs \
               --net "$NET" --voltage "$VOLTAGE" --out "$irdrop_out" </dev/null; then
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
