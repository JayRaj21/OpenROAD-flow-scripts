#!/usr/bin/env python3
"""
P&R Triage Agent

Reads stage-by-stage metrics from a completed ORFS run and uses Claude
to diagnose timing/congestion failure patterns and recommend targeted
interventions — the reasoning a senior physical design engineer would
apply when handed a failing run.

Builds on pr_metrics.py: collect() gathers the quality trajectory, the
agent formats it into a structured prompt, and Claude produces a diagnosis
with concrete ORFS parameters or hooks to try.

Usage:
    python3 flow/util/triage_agent.py --platform nangate45 --design ibex --tag base
    python3 flow/util/triage_agent.py --reports-dir flow/reports/nangate45/ibex/base \\
                                       --logs-dir    flow/logs/nangate45/ibex/base

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=<your key>   # or use `ant auth login`
"""

import argparse
import os
import sys

try:
    import anthropic
except ImportError:
    print(
        "ERROR: anthropic package not installed. Run: pip install anthropic",
        file=sys.stderr,
    )
    sys.exit(1)

# pr_metrics.py lives alongside this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr_metrics import collect, STAGES  # noqa: E402

# ---------------------------------------------------------------------------
# Domain knowledge injected into every prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior physical design engineer with deep expertise in digital \
ASIC P&R flows using OpenROAD-flow-scripts (ORFS).

You will be given a stage-by-stage quality trajectory for a completed design \
run — WNS (worst negative slack), TNS (total negative slack), Fmax, HPWL, \
and GRT routing overflow at each stage from global placement through finish.

Your job is to:
1. Identify exactly where quality degrades and why, based on known P&R \
failure patterns.
2. Classify the root cause (e.g. parasitic underestimation, placement \
density, clock tree quality, routing congestion, unconstrained paths).
3. Recommend specific, actionable ORFS parameters, make variables, or \
hook scripts to apply — not generic advice.

Key context about the flow:
- Stages in order: global place → resizer → detail place → CTS → \
global route → finish (detail route + sign-off).
- At CTS, parasitics are estimated from placement (optimistic). Real \
wire RC is only known after global route.
- ORFS already runs repair_timing and repair_design at global route \
before writing the GRT report — so violations visible in the GRT row \
survived built-in repair.
- The post-CTS hook (post_cts_timing_repair.tcl) runs iterative cell \
upsizing using placement parasitics. It is most effective when violations \
are visible at CTS.
- The post-GRT hook (post_grt_timing_repair.tcl) runs the same algorithm \
after global route, but ORFS's built-in repair_timing already runs first, \
so it typically finds few candidates.

Relevant ORFS parameters:
- CORE_UTILIZATION: die area utilisation %. Lower = more routing room.
- PLACE_DENSITY_LB_ADDON: extra placement density margin (0.0–0.5).
- TNS_END_PERCENT: % of violating endpoints to fix during placement \
optimisation. 100 = fix all.
- SETUP_SLACK_MARGIN: extra setup margin added during repair (ns).
- OPT_POST_GRT_WNS: 1 = run an extra VT-swap repair pass after GRT.
- POST_CTS_TCL: path to a Tcl script sourced after CTS \
(post_cts_timing_repair.tcl available on this branch).

Be specific and concise. Format your response as:
- **Root cause**: one sentence.
- **Evidence**: the specific stage transitions that support your diagnosis.
- **Recommended actions**: numbered list of concrete steps, each with the \
exact parameter name or command.
- **Expected outcome**: what the metrics should look like after the fix.
"""


# ---------------------------------------------------------------------------
# Trajectory formatting
# ---------------------------------------------------------------------------


def format_trajectory(rows):
    """Format collect() output as a readable table for the prompt."""
    lines = []
    lines.append(
        f"{'Stage':<16} {'WNS (ns)':>10} {'TNS (ns)':>10}"
        f" {'Fmax (MHz)':>11} {'HPWL (um)':>14} {'GRT overflow':>13}"
    )
    lines.append("-" * 76)
    for name, m in rows:
        wns = f"{m['wns']:+.3f}" if "wns" in m else "—"
        tns = f"{m['tns']:+.3f}" if "tns" in m else "—"
        fmax = f"{m['fmax_mhz']:.1f}" if "fmax_mhz" in m else "—"
        hpwl = f"{m['hpwl']:,}" if "hpwl" in m else "—"
        overflow = (
            f"{m.get('gp_overflow', m.get('grt_overflow')):.4f}"
            if "gp_overflow" in m or "grt_overflow" in m
            else "—"
        )
        lines.append(
            f"{name:<16} {wns:>10} {tns:>10} {fmax:>11} {hpwl:>14} {overflow:>13}"
        )
    return "\n".join(lines)


def compute_deltas(rows):
    """Return a list of notable stage-to-stage WNS changes."""
    deltas = []
    prev_name, prev_wns = None, None
    for name, m in rows:
        wns = m.get("wns")
        if wns is not None and prev_wns is not None:
            delta = wns - prev_wns
            if abs(delta) >= 0.005:
                deltas.append(
                    f"  {prev_name} → {name}: WNS {prev_wns:+.3f} → {wns:+.3f}"
                    f" (Δ {delta:+.3f} ns)"
                )
        if wns is not None:
            prev_wns = wns
            prev_name = name
    return deltas


def build_user_prompt(rows, label):
    """Assemble the full user message."""
    trajectory = format_trajectory(rows)
    deltas = compute_deltas(rows)

    final_wns = None
    final_tns = None
    power = None
    for _, m in reversed(rows):
        if final_wns is None and "wns" in m:
            final_wns = m["wns"]
        if final_tns is None and "tns" in m:
            final_tns = m["tns"]
        if power is None and "total_power_w" in m:
            power = m["total_power_w"]

    parts = [
        f"Design run: {label}\n",
        "Stage-by-stage quality trajectory:",
        trajectory,
    ]

    if deltas:
        parts.append("\nNotable WNS changes between stages:")
        parts.extend(deltas)

    parts.append(f"\nFinal WNS: {final_wns:+.3f} ns" if final_wns is not None else "")
    parts.append(f"Final TNS: {final_tns:+.3f} ns" if final_tns is not None else "")
    if power is not None:
        parts.append(f"Total power: {power:.4e} W")

    parts.append(
        "\nDiagnose the root cause of any quality issues and recommend "
        "specific interventions."
    )

    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def run_triage(rows, label):
    client = anthropic.Anthropic()

    user_prompt = build_user_prompt(rows, label)

    print(f"\nTriaging run: {label}")
    print("=" * 70)
    print(format_trajectory(rows))
    print("=" * 70)
    print("\nQuerying Claude for diagnosis...\n")

    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    for block in response.content:
        if block.type == "text":
            print(block.text)

    print(
        f"\n[tokens: {response.usage.input_tokens} in,"
        f" {response.usage.output_tokens} out]"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="P&R triage agent")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--platform", help="Platform name (e.g. nangate45)")
    group.add_argument("--reports-dir", help="Direct path to reports directory")

    parser.add_argument("--design", help="Design name (required with --platform)")
    parser.add_argument("--tag", default="base", help="Tag / variant (default: base)")
    parser.add_argument("--logs-dir", help="Direct path to logs directory")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    flow_dir = os.path.dirname(script_dir)
    parser.add_argument(
        "--flow-dir",
        default=flow_dir,
        help=f"Path to flow/ directory (default: {flow_dir})",
    )

    args = parser.parse_args()

    if args.platform:
        if not args.design:
            parser.error("--design is required when using --platform")
        reports_dir = os.path.join(
            args.flow_dir, "reports", args.platform, args.design, args.tag
        )
        logs_dir = os.path.join(
            args.flow_dir, "logs", args.platform, args.design, args.tag
        )
        label = f"{args.platform}/{args.design}/{args.tag}"
    else:
        reports_dir = args.reports_dir
        logs_dir = args.logs_dir or reports_dir.replace("/reports/", "/logs/")
        label = reports_dir

    if not os.path.isdir(reports_dir):
        print(f"ERROR: reports directory not found: {reports_dir}", file=sys.stderr)
        sys.exit(1)

    rows = collect(reports_dir, logs_dir)
    run_triage(rows, label)


if __name__ == "__main__":
    main()
