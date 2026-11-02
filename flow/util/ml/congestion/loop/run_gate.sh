#!/usr/bin/env bash
# Run the Stage B go/no-go gate end to end: place the density variants,
# extract features and HotSpot labels, train the three leave-design-family-out
# checkpoints, and rank the variants with gate_rank.py.
#
# The pass criteria are pre-registered in DESIGN_RUNS.md (2026-09-25).
#
# Usage (from anywhere; the script anchors itself to flow/):
#   bash util/ml/congestion/loop/run_gate.sh [--designs a,b,c,d] \
#       [--addons 0.05,0.15,0.30,0.45,0.60,0.75] [--skip-place] [--skip-extract] \
#       [--skip-train] [--epochs 200] [--dry-run]
#
# A failing placement or extraction for one variant does not abort the others;
# the summary lists them, and the last line of the run gives the number of
# skipped variants (gate.json skipped_variants) and failed stages, so a PASS
# with skipped variants is visible.
#
# Exit status: 0 PASS, 1 FAIL, 2 INCONCLUSIVE (gate_rank.py's status, and only
# when gate.json exists, parses and carries the matching verdict); 3 ERROR
# with no verdict: usage error, a failed train or verify stage (a failed
# training is never scored against an older checkpoint), gate_rank.py exit 3,
# a missing or mismatching gate.json, or a changed base/data fingerprint. On
# every exit 3 after the run began, any gate.json and gate.md are removed.
# --dry-run prints every command and runs none of them (Docker is not started,
# nothing is written).
#
# Checkpoints are trained with train_lodo.py --reuse-if-valid: an existing
# checkpoint is reused only if its sidecar hash, family, epochs, seed, batch
# size, learning rate, Laplacian weight and data-key list match this run,
# otherwise it is retrained. Each checkpoint is then checked with
# train_lodo.py --verify-only, and gate_rank.py gets --expect-epochs and
# --expect-seed, so a checkpoint from another run is refused. --epochs below 100
# is a smoke test: gate_rank.py records smoke_test in gate.json and the verdict
# is INCONCLUSIVE.
#
# Protection of existing data: the only writers are knob_variants.sh
# (results/<pdk>/<design>/dn_*), extract_variant_labels.sh
# (experiments/thermal_loop/data/), train_lodo.py (checkpoints/) and
# gate_rank.py (experiments/thermal_loop/). Every tag this script passes is
# checked against ^dn_[0-9]{3}$, so `base` can never be named, and before and
# after a real run a fingerprint of every results/*/*/base file and of
# util/ml/congestion/data/ is compared; a change aborts with an error.

set -uo pipefail
cd "$(dirname "$0")/../../../.."   # -> flow/
if [ ! -x util/docker_shell ] || [ ! -f util/ml/congestion/loop/gate_rank.py ]; then
    echo "Not anchored at flow/ (pwd=$(pwd))" >&2
    exit 3
fi

DESIGNS="sky130hd/riscv32i,gf180/riscv32i,sky130hs/aes,sky130hd/ibex"
ADDONS="0.05,0.15,0.30,0.45,0.60,0.75"
EPOCHS=200
SKIP_PLACE=0; SKIP_EXTRACT=0; SKIP_TRAIN=0; DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --designs)      DESIGNS="$2"; shift 2 ;;
        --addons)       ADDONS="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --skip-place)   SKIP_PLACE=1; shift ;;
        --skip-extract) SKIP_EXTRACT=1; shift ;;
        --skip-train)   SKIP_TRAIN=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 3 ;;
    esac
done

if [[ ! "$EPOCHS" =~ ^[0-9]+$ ]] || [ "$EPOCHS" -lt 1 ]; then
    echo "Invalid --epochs '${EPOCHS}'" >&2; exit 3
fi
MIN_GATE_EPOCHS=100

LOOP=util/ml/congestion/loop
DATA_DIR=util/ml/congestion/data
CKPT_DIR=util/ml/congestion/checkpoints
EXP_DIR=util/ml/congestion/experiments/thermal_loop
LOG="${EXP_DIR}/gate_run.log"

IFS=',' read -ra design_list <<< "$DESIGNS"
IFS=',' read -ra addon_list <<< "$ADDONS"

tags=()
for addon in "${addon_list[@]}"; do
    if [[ ! "$addon" =~ ^([0-9]+)(\.([0-9]{1,2}))?$ ]]; then
        echo "Invalid add-on '${addon}'" >&2; exit 3
    fi
    frac="${BASH_REMATCH[3]}00"
    printf -v tag 'dn_%03d' $(( 10#${BASH_REMATCH[1]} * 100 + 10#${frac:0:2} ))
    if [[ ! "$tag" =~ ^dn_[0-9]{3}$ ]]; then
        echo "Refusing tag '${tag}': only dn_NNN variants may be written" >&2; exit 3
    fi
    tags+=("$tag")
done
tag_csv=$(IFS=','; echo "${tags[*]}")

family_of() {
    local name="${1#*/}"
    echo "${name%_lvt}"
}

families=()
for d in "${design_list[@]}"; do
    fam=$(family_of "$d")
    [[ " ${families[*]} " == *" ${fam} "* ]] || families+=("$fam")
done

fingerprint() {
    { find results -path '*/base/*' -type f -printf '%p %T@ %s\n' 2>/dev/null | sort
      find "$DATA_DIR" -type f -printf '%p %T@ %s\n' | sort; } | md5sum
}

log() {
    if [ "$DRY_RUN" -eq 0 ]; then
        echo "$(date -Is) $*" >> "$LOG"
    fi
    echo "$(date -Is) $*"
}

stage_start() { STAGE="$1"; STAGE_T0=$(date +%s); log "START ${STAGE}"; }
stage_end()   { log "END   ${STAGE} status=$1 elapsed=$(( $(date +%s) - STAGE_T0 ))s"; }

failures=()
stage_failures=0

clear_verdict() { rm -f "${EXP_DIR}/gate.json" "${EXP_DIR}/gate.md"; }

# error_exit <message>: exit 3 with no verdict and no stale gate files.
error_exit() {
    clear_verdict
    echo "ERROR, no verdict: $1" >&2
    log "ERROR $1"
    exit 3
}

# run <label> <cmd...>: prints the command in a dry run, else runs it,
# logs its elapsed time and records a failure without aborting.
run() {
    local label="$1"; shift
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "WOULD RUN [${label}]: $*"
        return 0
    fi
    stage_start "$label"
    "$@"
    local ec=$?
    stage_end "$ec"
    if [ "$ec" -ne 0 ]; then
        failures+=("${label} (exit ${ec})")
        stage_failures=$((stage_failures + 1))
    fi
    return 0
}

echo "flow dir: $(pwd)"
echo "designs: ${DESIGNS}"
echo "add-ons: ${ADDONS}  tags: ${tag_csv}"
echo "families: ${families[*]}"
echo "epochs: ${EPOCHS}  dry-run: ${DRY_RUN}"
if [ "$EPOCHS" -lt "$MIN_GATE_EPOCHS" ]; then
    echo "!!! WARNING: --epochs ${EPOCHS} is below ${MIN_GATE_EPOCHS}: this is a SMOKE TEST." >&2
    echo "!!! The result must NOT be used as the gate verdict; gate_rank.py will report INCONCLUSIVE." >&2
fi

if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$EXP_DIR"
    clear_verdict
    before=$(fingerprint)
    log "gate run begins (designs=${DESIGNS} addons=${ADDONS})"
fi

for d in "${design_list[@]}"; do
    if [ "$SKIP_PLACE" -eq 0 ]; then
        run "place ${d}" bash "${LOOP}/knob_variants.sh" --design "$d" --addons "$ADDONS"
    fi
    if [ "$SKIP_EXTRACT" -eq 0 ]; then
        present=()
        for t in "${tags[@]}"; do
            if [ "$DRY_RUN" -eq 1 ] || [ -f "results/${d}/${t}/3_place.odb" ]; then
                present+=("$t")
            else
                failures+=("extract ${d}/${t} (no 3_place.odb; placement failed or infeasible)")
            fi
        done
        if [ "${#present[@]}" -gt 0 ]; then
            present_csv=$(IFS=','; echo "${present[*]}")
            run "extract ${d}" env OR_IMAGE=openroad/orfs-ml:latest bash "${LOOP}/extract_variant_labels.sh" --design "$d" --tags "$present_csv"
        fi
    fi
done

ckpt_map=""
train_failed=()
for fam in "${families[@]}"; do
    ckpt="${CKPT_DIR}/thermal_lodo_${fam}.pt"
    ckpt_map+="${ckpt_map:+,}${fam}=${ckpt}"
    failures_before=$stage_failures
    if [ "$SKIP_TRAIN" -eq 0 ]; then
        run "train ${fam}" python3 "${LOOP}/train_lodo.py" --data-dir "$DATA_DIR" --holdout-design "$fam" --out "$ckpt" --epochs "$EPOCHS" --seed 0 --reuse-if-valid
    fi
    run "verify ${fam}" python3 "${LOOP}/train_lodo.py" --data-dir "$DATA_DIR" --holdout-design "$fam" --out "$ckpt" --epochs "$EPOCHS" --seed 0 --verify-only
    if [ "$stage_failures" -gt "$failures_before" ]; then
        train_failed+=("$fam")
    fi
done

gate_cmd=(python3 "${LOOP}/gate_rank.py" --variant-dir "${EXP_DIR}/data" --designs "$DESIGNS"
          --checkpoint-map "$ckpt_map" --out "${EXP_DIR}/gate.json" --markdown "${EXP_DIR}/gate.md"
          --expect-epochs "$EPOCHS" --expect-seed 0)

if [ "$DRY_RUN" -eq 1 ]; then
    echo "WOULD RUN [gate_rank]: ${gate_cmd[*]}"
    echo "would log stage times to ${LOG}"
    exit 0
fi

if [ "${#train_failed[@]}" -gt 0 ]; then
    for f in "${failures[@]}"; do
        echo "  FAILED: ${f}" >&2
    done
    error_exit "train or verify failed for: ${train_failed[*]}; gate_rank.py was not run and no older checkpoint was scored"
fi

clear_verdict
stage_start "gate_rank"
"${gate_cmd[@]}"
gate_ec=$?
stage_end "$gate_ec"

after=$(fingerprint)
if [ "$before" != "$after" ]; then
    error_exit "results/*/*/base or ${DATA_DIR} changed during the run"
fi

case "$gate_ec" in
    0) want=PASS ;;
    1) want=FAIL ;;
    2) want=INCONCLUSIVE ;;
    *) error_exit "gate_rank.py exited ${gate_ec}; see its output above" ;;
esac
got=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"]["overall"])' "${EXP_DIR}/gate.json" 2>/dev/null) || got=""
if [ "$got" != "$want" ]; then
    error_exit "gate_rank.py exited ${gate_ec} (${want}) but ${EXP_DIR}/gate.json is missing, unreadable or says '${got}'"
fi

echo ""
echo "Summary: ${#failures[@]} stage/variant failure(s)"
for f in "${failures[@]}"; do
    echo "  FAILED: ${f}"
done
log "gate run ends, gate_rank exit ${gate_ec}"
case "$gate_ec" in
    0) echo "Verdict: PASS (exit 0); see ${EXP_DIR}/gate.json" ;;
    1) echo "Verdict: FAIL (exit 1); a genuine scored result, see ${EXP_DIR}/gate.json" ;;
    2) echo "Verdict: INCONCLUSIVE (exit 2); see ${EXP_DIR}/gate.json" ;;
esac
skipped=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sum(len(e["skipped_variants"]) for e in d["designs"].values()))' "${EXP_DIR}/gate.json")
echo "!!! ${skipped} skipped variant(s) in gate.json, ${stage_failures} failed stage(s), ${#failures[@]} failure line(s) above"
exit "$gate_ec"
