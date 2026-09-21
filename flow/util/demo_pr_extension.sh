#!/usr/bin/env bash
# demo_pr_extension.sh
#
# Walks through the pr-extension tooling on one built design, step by step:
#   1. unit tests            (no Docker, no API key)
#   2. pr_metrics            per-stage timing/quality table
#   3. cts_diagnostic        clock-tree health + CTS->GRT timing cliff check
#   4. benchmark_dashboard   record two runs, then report/regression check + HTML
#   5. eco_fix               one targeted incremental repair, verified before/after
#   6. triage_agent          LLM diagnosis (only with --llm and ANTHROPIC_API_KEY)
#
# The demo is non-destructive: the ECO step backs up and restores the design
# database, and the benchmark history file is put back exactly as it was.
#
# Every run saves a copy of what it printed (report.txt) and the benchmark HTML
# chart under util/demo_runs/<platform>-<design>-<tag>/<date-time>/. Git ignores
# that folder.
#
# Usage (from anywhere; it runs inside flow/):
#   util/demo_pr_extension.sh
#   util/demo_pr_extension.sh --platform nangate45 --design gcd --pause
#   util/demo_pr_extension.sh --build          # build the design first if missing
#   util/demo_pr_extension.sh --llm            # also run the LLM triage agent
#
# Steps 4 and 5 run flow commands inside the Docker container via util/docker_shell.
# The multi-corner dashboard is not part of this demo: it needs a design with two
# or more timing corners and the report hooks enabled during the build.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLOW_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$FLOW_DIR"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PLATFORM=nangate45
DESIGN=gcd
TAG=base
BUILD=0
LLM=0
PAUSE=0
ECO_STAGE=grt
ECO_MAX_TRIES=10
BENCH_STAGE="Global route"   # "Finish" (the tool's default) is empty unless the build ran the whole flow

usage() {
    cat <<EOF
Usage: util/demo_pr_extension.sh [options]

Options:
  --platform <name>   Platform (default: $PLATFORM)
  --design <name>     Design (default: $DESIGN)
  --tag <name>        Variant tag (default: $TAG)
  --build             Build the design with Docker if its results are missing
  --llm               Also run the LLM triage agent (needs ANTHROPIC_API_KEY)
  --pause             Wait for Enter between steps
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --design)   DESIGN="$2";   shift 2 ;;
        --tag)      TAG="$2";      shift 2 ;;
        --build)    BUILD=1;       shift ;;
        --llm)      LLM=1;         shift ;;
        --pause)    PAUSE=1;       shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'; else BOLD=""; DIM=""; RESET=""; fi

STEP_NUM=0
step() {
    STEP_NUM=$((STEP_NUM + 1))
    echo
    echo "${BOLD}=== Step $STEP_NUM: $1 ===${RESET}"
    echo "${DIM}$2${RESET}"
    if [[ $PAUSE -eq 1 ]]; then read -r -p "Press Enter to run... " _; fi
    echo
}

show_cmd() { echo "${DIM}\$ $*${RESET}"; }

# Everything below runs inside main() so the driver at the bottom of the file
# can record what it prints into a report.
main() {
ODB="results/$PLATFORM/$DESIGN/$TAG/5_1_grt.odb"
[[ "$ECO_STAGE" == "cts" ]] && ODB="results/$PLATFORM/$DESIGN/$TAG/4_cts.odb"
HISTORY="util/benchmark_history/${PLATFORM}__${DESIGN}__${TAG}.jsonl"

TMP_DIR="$(mktemp -d)"
ODB_BACKUP="$TMP_DIR/odb.backup"
HISTORY_BACKUP="$TMP_DIR/history.backup"
HAD_HISTORY=0
RESTORE_ODB=0

# Put everything back the way it was, even if the demo fails part-way.
cleanup() {
    if [[ $RESTORE_ODB -eq 1 && -f "$ODB_BACKUP" ]]; then
        cp -p "$ODB_BACKUP" "$ODB"
    fi
    if [[ $HAD_HISTORY -eq 1 ]]; then
        cp -p "$HISTORY_BACKUP" "$HISTORY"
    else
        rm -f "$HISTORY"
    fi
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
python3 -c "import pytest" 2>/dev/null || { echo "pytest is required (pip install pytest)" >&2; exit 1; }

HAVE_DOCKER=1
command -v docker >/dev/null || HAVE_DOCKER=0

if [[ ! -f "$ODB" ]]; then
    if [[ $BUILD -eq 1 ]]; then
        [[ $HAVE_DOCKER -eq 1 ]] || { echo "Docker is required to build the design" >&2; exit 1; }
        echo "Building $PLATFORM/$DESIGN (this takes a few minutes)..."
        show_cmd util/docker_shell make DESIGN_CONFIG=designs/$PLATFORM/$DESIGN/config.mk
        util/docker_shell make "DESIGN_CONFIG=designs/$PLATFORM/$DESIGN/config.mk"
    else
        echo "No built design found at flow/$ODB" >&2
        echo "Build it first with:" >&2
        echo "  util/docker_shell make DESIGN_CONFIG=designs/$PLATFORM/$DESIGN/config.mk" >&2
        echo "or re-run this demo with --build." >&2
        exit 1
    fi
fi

echo "${BOLD}pr-extension demo${RESET}  design: $PLATFORM/$DESIGN ($TAG)"
echo "run: $STAMP   commit: $(git -C "$FLOW_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# ---------------------------------------------------------------------------
# 1. Unit tests
# ---------------------------------------------------------------------------
step "Unit tests" "The four test suites cover the loop agent (incl. eco_fix) and the three report tools. No Docker or API key needed."
show_cmd python3 -m pytest util/test_loop_agent.py util/test_cts_diagnostic.py util/test_multicorner_dashboard.py util/test_benchmark_dashboard.py -q
python3 -m pytest util/test_loop_agent.py util/test_cts_diagnostic.py \
    util/test_multicorner_dashboard.py util/test_benchmark_dashboard.py -q

# ---------------------------------------------------------------------------
# 2. pr_metrics
# ---------------------------------------------------------------------------
step "Stage-by-stage metrics" "pr_metrics.py collects WNS/TNS/Fmax/overflow from every flow stage. All the other tools build on it."
show_cmd python3 util/pr_metrics.py --platform $PLATFORM --design $DESIGN --tag $TAG
python3 util/pr_metrics.py --platform "$PLATFORM" --design "$DESIGN" --tag "$TAG"

# ---------------------------------------------------------------------------
# 3. cts_diagnostic
# ---------------------------------------------------------------------------
step "Clock-tree diagnostic" "Checks clock buffer/sink/skew health and whether timing falls off a cliff between CTS and global route."
show_cmd python3 util/cts_diagnostic.py --platform $PLATFORM --design $DESIGN --tag $TAG
rc=0
python3 util/cts_diagnostic.py --platform "$PLATFORM" --design "$DESIGN" --tag "$TAG" || rc=$?
case $rc in
    0) echo "${DIM}exit 0: no problems found${RESET}" ;;
    1) echo "${DIM}exit 1: the tool found something worth a look (see the verdict above; this is a finding, not a crash)${RESET}" ;;
    *) echo "cts_diagnostic failed with exit code $rc (2 = bad input, 3 = internal error)" >&2; exit "$rc" ;;
esac

# ---------------------------------------------------------------------------
# 4. benchmark_dashboard
# ---------------------------------------------------------------------------
step "Regression dashboard" "Records this run's metrics to an append-only history, twice, then reports deltas and checks for regressions."
if [[ -f "$HISTORY" ]]; then
    cp -p "$HISTORY" "$HISTORY_BACKUP"
    HAD_HISTORY=1
fi
HTML_OUT="$RUN_DIR/benchmark_dashboard.html"
show_cmd python3 util/benchmark_dashboard.py record --platform $PLATFORM --design $DESIGN --tag $TAG
python3 util/benchmark_dashboard.py record --platform "$PLATFORM" --design "$DESIGN" --tag "$TAG"
python3 util/benchmark_dashboard.py record --platform "$PLATFORM" --design "$DESIGN" --tag "$TAG"
echo
show_cmd python3 util/benchmark_dashboard.py report --platform $PLATFORM --design $DESIGN --tag $TAG --stage \"$BENCH_STAGE\" --html "$HTML_OUT"
rc=0
python3 util/benchmark_dashboard.py report --platform "$PLATFORM" --design "$DESIGN" --tag "$TAG" \
    --stage "$BENCH_STAGE" --html "$HTML_OUT" || rc=$?
if [[ $rc -eq 0 ]]; then
    echo "${DIM}exit 0: no regression (both records are the same run, so nothing changed)${RESET}"
else
    echo "${DIM}exit $rc: the report flagged a regression or a history problem${RESET}"
fi
echo "HTML trend chart: saved as benchmark_dashboard.html in this run's folder (path printed at the end)"

# ---------------------------------------------------------------------------
# 5. eco_fix
# ---------------------------------------------------------------------------
step "Targeted ECO repair (eco_fix)" "Upsizes one cell on a worst-timing path, measures timing before and after in one OpenROAD session, and only keeps the change if it helped. The design database is restored afterwards."
if [[ $HAVE_DOCKER -eq 0 ]]; then
    echo "Docker not found; skipping (eco_fix runs OpenROAD inside the Docker container)."
else
    cp -p "$ODB" "$ODB_BACKUP"
    RESTORE_ODB=1
    ECO_STAGE="$ECO_STAGE" ECO_MAX_TRIES="$ECO_MAX_TRIES" PLATFORM="$PLATFORM" DESIGN="$DESIGN" TAG="$TAG" \
    python3 - <<'PY'
import contextlib
import io
import itertools
import os
import re
import sys

sys.path.insert(0, "util")
from loop_agent import impl_eco_fix

stage = os.environ["ECO_STAGE"]
platform, design, tag = os.environ["PLATFORM"], os.environ["DESIGN"], os.environ["TAG"]
max_tries = int(os.environ["ECO_MAX_TRIES"])
counter = itertools.count(1)


def run(fix_type, target):
    # impl_eco_fix echoes the Docker command it runs; keep the demo output readable.
    with contextlib.redirect_stdout(io.StringIO()):
        return impl_eco_fix(fix_type, target, stage, "", platform, design, tag, ".", [], counter)


def without_targets(text):
    # The full result also lists every worst-path instance; the demo shows that list separately.
    return text.split("targets (worst setup paths):")[0].rstrip()


print("Asking eco_fix for the instances on the worst setup paths (a deliberate bad target)...")
probe = run("resize_up", "__demo_probe__")
candidates = []
for inst, cell in re.findall(r"^\s{2}(\S+)\s+\((\S+)\)\s+pin=", probe, re.M):
    if inst not in [c[0] for c in candidates]:
        candidates.append((inst, cell))
if not candidates:
    print("Could not find any candidate instances; the design may have no timing paths.")
    sys.exit(0)

print(f"Found {len(candidates)} instances on the worst paths, for example:")
for inst, cell in candidates[:5]:
    print(f"  {inst} ({cell})")

print()
print(f"Trying resize_up on up to {max_tries} of them until one is kept...")
first_rejected = None
kept = None
for inst, cell in candidates[:max_tries]:
    out = run("resize_up", inst)
    head = out.splitlines()[0]
    reason = re.search(r"verdict: \w+ — (.*)\)\s*$", head)
    reason = reason.group(1) if reason else ""
    if out.startswith("status: error"):
        msg = next((l for l in out.splitlines() if l.startswith("msg:")), "msg: unknown")
        print(f"  {inst} ({cell}): skipped — {msg[4:].strip()}")
        continue
    if out.startswith("status: applied"):
        print(f"  {inst} ({cell}): KEPT")
        kept = out
        break
    print(f"  {inst} ({cell}): rejected — {reason}")
    if first_rejected is None:
        first_rejected = out

print()
if kept:
    print("A change was kept. Full before/after result:")
    print(without_targets(kept))
    print()
    print("odb_written: True means the change was saved to the database (the demo restores it afterwards).")
elif first_rejected:
    print("None of those changes helped enough to be kept, so the database was never modified.")
    print("That is the safety check working. Result for the first one tried:")
    print(without_targets(first_rejected))
else:
    print("No candidate could be resized; try another design or stage.")
PY
    cp -p "$ODB_BACKUP" "$ODB"
    RESTORE_ODB=0
    echo "${DIM}Design database restored to its original state.${RESET}"
fi

# ---------------------------------------------------------------------------
# 6. triage_agent (optional)
# ---------------------------------------------------------------------------
if [[ $LLM -eq 1 ]]; then
    step "LLM triage" "An LLM reads the stage-by-stage metrics and explains why quality degrades. This calls the Anthropic API and costs money."
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        echo "ANTHROPIC_API_KEY is not set; skipping."
    else
        show_cmd python3 util/triage_agent.py --platform $PLATFORM --design $DESIGN --tag $TAG
        python3 util/triage_agent.py --platform "$PLATFORM" --design "$DESIGN" --tag "$TAG"
    fi
fi

echo
echo "${BOLD}Demo complete.${RESET}"
if [[ $LLM -eq 0 ]]; then
    echo "Not shown: the LLM tools. Re-run with --llm for the triage agent; loop_agent.py is the autonomous fixer and edits config.mk, so run it yourself:"
    echo "  python3 util/loop_agent.py --platform $PLATFORM --design $DESIGN"
fi
}

# ---------------------------------------------------------------------------
# Run the demo, keeping a copy of everything it prints
# ---------------------------------------------------------------------------
STAMP="$(date +%Y%m%d-%H%M%S)"
RUNS_DIR="$FLOW_DIR/util/demo_runs"
RUN_DIR="$RUNS_DIR/$PLATFORM-$DESIGN-$TAG/$STAMP"
mkdir -p "$RUN_DIR"
# A folder-level .gitignore that ignores everything (itself included) keeps the
# saved runs out of git without touching the repository's own .gitignore.
[[ -f "$RUNS_DIR/.gitignore" ]] || echo '*' > "$RUNS_DIR/.gitignore"

export PYTHONUNBUFFERED=1   # so output shows up live instead of in one lump at the end of each step
RAW_LOG="$RUN_DIR/.report.raw"

set +e
main 2>&1 | tee "$RAW_LOG"
rc=${PIPESTATUS[0]}
set -e

# The report is the same text with terminal colour codes removed.
sed 's/\x1b\[[0-9;]*m//g' "$RAW_LOG" > "$RUN_DIR/report.txt"
rm -f "$RAW_LOG"

echo
echo "${BOLD}Saved this run to:${RESET} $RUN_DIR"
for f in "$RUN_DIR"/*; do echo "  $(basename "$f")"; done
exit "$rc"
