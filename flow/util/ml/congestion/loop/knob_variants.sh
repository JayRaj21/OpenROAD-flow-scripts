#!/usr/bin/env bash
# Generate global-placement density variants of one already-routed design by
# re-running only the placement steps (3_1 .. 3_place) from the base
# variant's synthesis and floorplan results.
#
# The knob is PLACE_DENSITY_LB_ADDON, not PLACE_DENSITY: scripts/util.tcl
# ignores PLACE_DENSITY entirely whenever PLACE_DENSITY_LB_ADDON is set, and
# most designs set it. It is normalised per design (it interpolates between
# the design's own minimum feasible density and 1.0), so the same value is
# comparable across designs. The floorplan is left untouched on purpose:
# changing utilization or aspect ratio would change die area, and with it the
# total power and package model used for the thermal labels.
#
# Usage (from flow/):
#   bash util/ml/congestion/loop/knob_variants.sh \
#       --design sky130hd/riscv32i \
#       --addons 0.05,0.15,0.30,0.45,0.60,0.75 \
#       [--pads 0] [--dry-run]
#
# --addons   comma-separated PLACE_DENSITY_LB_ADDON values, each a number in
#            [0, 1] written with at most 2 decimals (0.1, 0.10, 1). Anything
#            else is rejected; nothing is rounded. Values close to 1 are
#            infeasible (FLW-0024: lower bound + (1 - lower bound) * addon +
#            0.01 must stay <= 1.0); such a variant is reported as failed.
# --pads     comma-separated CELL_PAD_IN_SITES_GLOBAL_PLACEMENT values,
#            non-negative integers (default 0; the _p<pad> tag suffix is
#            omitted when pad is 0)
# --dry-run  Print the exact commands (also for variants that already exist,
#            marked "would skip") without staging files or running make.
#
# Variant tag: dn_<addon*100, 3 digits>[_p<pad>], e.g. dn_030, dn_030_p1.
# Outputs: results/<pdk>/<design>/<tag>/3_place.odb, logs/<pdk>/<design>/<tag>/
# Existing 3_place.odb files are skipped (idempotent). A variant directory
# created by a run whose placement or staging failed is removed again (logs/
# is kept for diagnosis); a directory that existed before the run is never
# deleted. Exit status is non-zero if any variant failed.
#
# Platform files: every variant is placed with PLATFORM_HOME=/work/platforms
# (this worktree), never the image's baked-in copy. Placement is timing-driven
# and sources platforms/<pdk>/setRC.tcl, so a placement is only comparable
# with one made using the same platform files. The 2026-09-14 base of the 12
# first designs used the image copy; see DESIGN_RUNS.md, 2026-09-20.

set -euo pipefail
cd "$(dirname "$0")/../../../.."   # → flow/

DESIGN=""
PADS="0"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --design)  DESIGN="$2"; shift 2 ;;
        --addons)  ADDONS="$2"; shift 2 ;;
        --pads)    PADS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$DESIGN" ] || [ -z "${ADDONS+x}" ]; then
    echo "Usage: $0 --design <pdk>/<design> --addons a,b,c [--pads 0] [--dry-run]" >&2
    exit 1
fi

# The trailing comma makes read keep a trailing empty entry, so "0.1," is
# rejected instead of silently dropped.
IFS=',' read -ra addon_list <<< "${ADDONS},"
IFS=',' read -ra pad_list <<< "${PADS},"

addon_pcts=()
for addon in "${addon_list[@]}"; do
    if [[ ! "$addon" =~ ^([0-9]+)(\.([0-9]{1,2}))?$ ]] || [ "${#BASH_REMATCH[1]}" -gt 1 ]; then
        echo "Invalid add-on '${addon}': expected a number in [0, 1] with at most 2 decimals" >&2
        exit 1
    fi
    frac="${BASH_REMATCH[3]}00"
    pct=$(( 10#${BASH_REMATCH[1]} * 100 + 10#${frac:0:2} ))
    if [ "$pct" -gt 100 ]; then
        echo "Invalid add-on '${addon}': out of range [0, 1]" >&2
        exit 1
    fi
    addon_pcts+=("$pct")
done

pad_values=()
for pad in "${pad_list[@]}"; do
    if [[ ! "$pad" =~ ^[0-9]{1,6}$ ]]; then
        echo "Invalid pad '${pad}': expected a non-negative integer" >&2
        exit 1
    fi
    pad_values+=("$(( 10#$pad ))")
done

config="designs/${DESIGN}/config.mk"
base_dir="results/${DESIGN}/base"
if [ ! -f "$config" ]; then
    echo "$config not found" >&2
    exit 1
fi
if [ ! -f "${base_dir}/2_floorplan.odb" ]; then
    echo "${base_dir}/2_floorplan.odb not found — route the base variant first" >&2
    exit 1
fi

mem_files=()
for f in "${base_dir}"/mem*.json; do
    [ -f "$f" ] && mem_files+=("$f")
done

# clock_period.txt is a prerequisite of the yosys canonicalize step and
# mem*.json of the memories chain; without them make re-runs synthesis and
# floorplan even when every 1_*/2_* file is present. One touch invocation
# gives every staged file the same timestamp; make rebuilds only on a
# strictly newer prerequisite, so per-file cp ordering would otherwise
# re-trigger the synthesis chain. Each step is a plain command with an
# explicit || return so a missing input is a [FAIL], not a set -e abort.
stage_cmds() {
    echo "mkdir -p ${variant_dir}"
    echo "cp ${base_dir}/1_* ${base_dir}/2_* ${base_dir}/clock_period.txt ${variant_dir}/"
    if [ "${#mem_files[@]}" -gt 0 ]; then
        echo "cp ${mem_files[*]} ${variant_dir}/"
    fi
    echo "find ${variant_dir} -type f -exec touch {} +"
}

stage_variant() {
    mkdir -p "$variant_dir" || return 1
    cp "${base_dir}"/1_* "${base_dir}"/2_* "${base_dir}"/clock_period.txt "$variant_dir"/ || return 1
    if [ "${#mem_files[@]}" -gt 0 ]; then
        cp "${mem_files[@]}" "$variant_dir"/ || return 1
    fi
    find "$variant_dir" -type f -exec touch {} + || return 1
}

pass=0; fail=0; skip=0

for i in "${!addon_list[@]}"; do
    addon="${addon_list[$i]}"
    for pad in "${pad_values[@]}"; do
        printf -v tag 'dn_%03d' "${addon_pcts[$i]}"
        [ "$pad" != "0" ] && tag+="_p${pad}"

        variant_dir="results/${DESIGN}/${tag}"
        result_odb="${variant_dir}/3_place.odb"

        echo ""
        echo ">>> ${DESIGN}/${tag}  (PLACE_DENSITY_LB_ADDON=${addon}, pad=${pad})"

        # /work overrides are mandatory: the openroad/orfs image's baked-in
        # flow/ copy is stale relative to this worktree.
        inner="make DESIGN_CONFIG=/work/${config} DESIGN_HOME=/work/designs PLATFORM_HOME=/work/platforms"
        inner+=" FLOW_VARIANT=${tag} PLACE_DENSITY_LB_ADDON=${addon}"
        inner+=" CELL_PAD_IN_SITES_GLOBAL_PLACEMENT=${pad} place"
        # </dev/null prevents docker -i from consuming any surrounding stdin.
        run_cmd="util/docker_shell -- \"${inner}\" </dev/null"

        exists=0
        [ -f "$result_odb" ] && exists=1

        if [ "$DRY_RUN" -eq 1 ]; then
            if [ "$exists" -eq 1 ]; then
                echo "  would skip: already exists (${result_odb}); the commands below would not run"
                ((skip++)) || true
            fi
            stage_cmds | sed 's/^/    /'
            echo "    ${run_cmd}"
            continue
        fi

        if [ "$exists" -eq 1 ]; then
            echo "  [SKIP] $result_odb already exists"
            ((skip++)) || true
            continue
        fi

        created=0
        [ -d "$variant_dir" ] || created=1

        if stage_variant && util/docker_shell -- "$inner" </dev/null; then
            ((pass++)) || true
        else
            echo "  [FAIL] ${DESIGN}/${tag} — continuing"
            if [ "$created" -eq 1 ]; then
                rm -rf "$variant_dir"
                echo "  removed ${variant_dir} (created by this run)"
            fi
            ((fail++)) || true
        fi
    done
done

echo ""
echo "Done.  passed=${pass}  failed=${fail}  skipped=${skip}  (dry-run=${DRY_RUN})"
[ "$fail" -eq 0 ] || exit 1
