#!/usr/bin/env python3
"""
P&R Closed-Loop Optimization Agent

Observes stage-by-stage metrics, diagnoses timing failures, applies targeted
ORFS parameter changes, re-runs the affected flow stages, and verifies
improvement — without human intervention.

Builds on pr_metrics.py and the triage-agent system prompt. Tools give the
model direct access to read metrics, queue parameter changes, and trigger
Docker-based make runs.

Usage:
    python3 flow/util/loop_agent.py --platform nangate45 --design aes --tag base

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=<your key>   # or `ant auth login`
    Docker available with openroad/orfs:latest image
"""

import argparse
import json
import os
import subprocess
import sys

try:
    import anthropic
except ImportError:
    print(
        "ERROR: anthropic package not installed. Run: pip install anthropic",
        file=sys.stderr,
    )
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr_metrics import collect  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TOOL_TURNS = 20  # hard cap on total API round-trips

# Parameters the agent is allowed to change
PARAM_ALLOWLIST = {
    "SETUP_SLACK_MARGIN",
    "TNS_END_PERCENT",
    "OPT_POST_GRT_WNS",
    "POST_CTS_TCL",
    "POST_GLOBAL_ROUTE_TCL",
}

# Hook scripts available inside the Docker container (workspace = /work)
HOOK_PATHS = {
    "POST_CTS_TCL": "/work/scripts/post_cts_timing_repair.tcl",
    "POST_GLOBAL_ROUTE_TCL": "/work/scripts/post_grt_timing_repair.tcl",
}

# Stale ODB files to delete when forcing a stage re-run
STAGE_STALE_FILES = {
    "cts": ["results/{p}/{d}/{t}/4_1_cts.odb", "results/{p}/{d}/{t}/4_cts.odb"],
    "grt": ["results/{p}/{d}/{t}/5_1_grt.odb", "results/{p}/{d}/{t}/5_1_grt.sdc"],
    "finish": [
        "results/{p}/{d}/{t}/5_2_route.odb",
        "results/{p}/{d}/{t}/5_route.odb",
    ],
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a closed-loop P&R optimization agent for OpenROAD-flow-scripts (ORFS). \
You have tools to read metrics, queue parameter changes, re-run flow stages, \
and declare completion.

## Workflow

1. Call get_metrics to read the current stage-by-stage quality trajectory.
2. Diagnose the root cause of any timing failures.
3. Call set_config_param for each targeted fix.
4. Call run_stage for each affected stage, starting from the earliest changed \
stage (cts, then grt if needed, then finish).
5. Call get_metrics again to verify improvement.
6. Repeat up to 3 iterations. When WNS ≥ 0 ns and TNS ≥ -0.05 ns at finish, \
or when your budget is exhausted, call finish.

## Flow context

- Stage order: global place → resizer → detail place → CTS → global route \
→ finish (detail route + sign-off).
- At CTS, parasitics are estimated from placement (optimistic, underestimates \
real wire RC). Real RC is only known after global route.
- A CTS→GRT cliff (CTS shows WNS +0.000, GRT shows WNS < -0.010) is \
parasitic underestimation. Fix: set SETUP_SLACK_MARGIN = 0.03.
- ORFS runs repair_timing at global route before writing the GRT report, so \
violations visible in the GRT row survived built-in repair.
- The post-CTS hook does iterative cell upsizing using placement parasitics. \
Set POST_CTS_TCL = "enabled" when violations will be visible at CTS after \
applying SETUP_SLACK_MARGIN.

## Allowlisted parameters

- SETUP_SLACK_MARGIN (float, 0.0–0.10 ns): extra setup margin during repair. \
Typical: 0.03. Forces CTS to see endpoints that only violate after real RC.
- TNS_END_PERCENT (int, 0–100): % of endpoints repaired. Set to 100 if any \
TNS exists.
- OPT_POST_GRT_WNS (0 or 1): enables a VT-swap pass after GRT.
- POST_CTS_TCL: set to "enabled" to arm the post-CTS upsizing hook.
- POST_GLOBAL_ROUTE_TCL: set to "enabled" to arm the post-GRT hook (rarely \
needed — built-in repair already runs before the GRT report).

## Stage re-run rules

- Changed SETUP_SLACK_MARGIN, TNS_END_PERCENT, or POST_CTS_TCL → run cts, \
then finish.
- Changed OPT_POST_GRT_WNS or POST_GLOBAL_ROUTE_TCL → run grt, then finish.

Budget: max 3 iterations. Each iteration may change up to 3 parameters and \
re-run 2 stages. Call finish when done regardless of outcome.
"""

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_metrics",
        "description": (
            "Read the current stage-by-stage quality trajectory (WNS, TNS, Fmax, "
            "GRT overflow). Call at the start and after each run_stage to see "
            "the updated results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "set_config_param",
        "description": (
            "Queue a parameter change that will be passed to the next run_stage "
            "call as a make variable. Only allowlisted parameters are accepted. "
            "For POST_CTS_TCL and POST_GLOBAL_ROUTE_TCL, pass value='enabled'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "ORFS make parameter name (must be allowlisted).",
                },
                "value": {
                    "type": "string",
                    "description": (
                        "Value to set. For hook paths use 'enabled'. "
                        "Numeric values as strings, e.g. '0.03' or '100'."
                    ),
                },
            },
            "required": ["param", "value"],
        },
    },
    {
        "name": "run_stage",
        "description": (
            "Re-run a flow stage inside the Docker container using all queued "
            "parameter changes. Stale output files are deleted first to force "
            "re-execution. Returns the tail of the make output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stage": {
                    "type": "string",
                    "enum": ["cts", "grt", "finish"],
                    "description": "Flow stage to re-run.",
                },
            },
            "required": ["stage"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Terminate the optimization loop. Call when timing has closed "
            "(WNS ≥ 0, TNS ≥ -0.05 at finish) or when the iteration budget "
            "is exhausted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "What was diagnosed, what parameters were changed, "
                        "and what the final metrics show."
                    ),
                },
                "success": {
                    "type": "boolean",
                    "description": "True if timing closed, False if budget exhausted.",
                },
            },
            "required": ["summary", "success"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def impl_get_metrics(platform, design, tag, flow_dir):
    reports_dir = os.path.join(flow_dir, "reports", platform, design, tag)
    logs_dir = os.path.join(flow_dir, "logs", platform, design, tag)
    if not os.path.isdir(reports_dir):
        return f"ERROR: reports directory not found: {reports_dir}"
    rows = collect(reports_dir, logs_dir)
    header = (
        f"{'Stage':<16} {'WNS (ns)':>10} {'TNS (ns)':>10}"
        f" {'Fmax (MHz)':>11} {'GRT overflow':>13}"
    )
    lines = [header, "-" * 62]
    for name, m in rows:
        wns = f"{m['wns']:+.3f}" if "wns" in m else "—"
        tns = f"{m['tns']:+.3f}" if "tns" in m else "—"
        fmax = f"{m['fmax_mhz']:.1f}" if "fmax_mhz" in m else "—"
        overflow = (
            f"{m.get('gp_overflow', m.get('grt_overflow')):.4f}"
            if "gp_overflow" in m or "grt_overflow" in m
            else "—"
        )
        lines.append(f"{name:<16} {wns:>10} {tns:>10} {fmax:>11} {overflow:>13}")
    for _, m in reversed(rows):
        if "wns" in m:
            lines.append(f"\nFinal WNS: {m['wns']:+.3f} ns")
            break
    for _, m in reversed(rows):
        if "tns" in m:
            lines.append(f"Final TNS: {m['tns']:+.3f} ns")
            break
    return "\n".join(lines)


def impl_set_config_param(param, value, pending_params, change_log):
    if param not in PARAM_ALLOWLIST:
        return (
            f"ERROR: '{param}' is not allowlisted. "
            f"Allowed: {sorted(PARAM_ALLOWLIST)}"
        )
    if param in HOOK_PATHS and value.lower() == "enabled":
        value = HOOK_PATHS[param]
    pending_params[param] = value
    change_log.append({"action": "set_param", "param": param, "value": value})
    return f"OK: {param} = {value}"


def impl_run_stage(stage, platform, design, tag, pending_params, flow_dir, change_log):
    # Delete stale output files to force make to re-run
    for pattern in STAGE_STALE_FILES.get(stage, []):
        path = os.path.join(flow_dir, pattern.format(p=platform, d=design, t=tag))
        if os.path.exists(path):
            os.remove(path)

    cmd = (
        [
            "util/docker_shell",
            "make",
            f"DESIGN_CONFIG=designs/{platform}/{design}/config.mk",
        ]
        + [f"{k}={v}" for k, v in pending_params.items()]
        + [stage]
    )
    change_log.append(
        {"action": "run_stage", "stage": stage, "params": dict(pending_params)}
    )

    print(f"\n[loop-agent] $ {' '.join(cmd)}", flush=True)
    try:
        result = subprocess.run(
            cmd, cwd=flow_dir, capture_output=True, text=True, timeout=1800
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "ERROR: make timed out after 30 minutes"

    tail = output[-3000:] if len(output) > 3000 else output
    return tail


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_loop(platform, design, tag, flow_dir):
    client = anthropic.Anthropic()
    pending_params = {}
    change_log = []
    label = f"{platform}/{design}/{tag}"

    print(f"\nLoop agent — {label}")
    print(f"Max tool turns: {MAX_TOOL_TURNS}")
    print("=" * 70)

    initial_metrics = impl_get_metrics(platform, design, tag, flow_dir)
    print(initial_metrics)
    print("=" * 70)

    messages = [
        {
            "role": "user",
            "content": (
                f"Design run: {label}\n\n"
                f"Current quality trajectory:\n{initial_metrics}\n\n"
                "Diagnose any timing issues and close them. "
                "You have at most 3 iterations."
            ),
        }
    ]

    turn = 0
    finished = False

    while not finished and turn < MAX_TOOL_TURNS:
        turn += 1

        response = client.messages.create(
            model="claude-opus-5",
            max_tokens=8000,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        # Print any visible text from the agent
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[agent] {block.text}")

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason != "tool_use":
            print(f"[loop-agent] Unexpected stop_reason: {response.stop_reason}")
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            inp = block.input
            print(f"\n[tool:{block.name}] {json.dumps(inp)}", flush=True)

            if block.name == "get_metrics":
                result = impl_get_metrics(platform, design, tag, flow_dir)

            elif block.name == "set_config_param":
                result = impl_set_config_param(
                    inp["param"], inp["value"], pending_params, change_log
                )

            elif block.name == "run_stage":
                result = impl_run_stage(
                    inp["stage"],
                    platform,
                    design,
                    tag,
                    pending_params,
                    flow_dir,
                    change_log,
                )

            elif block.name == "finish":
                status = "SUCCESS" if inp.get("success") else "BUDGET EXHAUSTED"
                print(f"\n{'='*70}")
                print(f"[loop-agent] {status}")
                print(inp.get("summary", ""))
                print(f"{'='*70}")
                change_log.append({"action": "finish", **inp})
                finished = True
                result = "Loop terminated."

            else:
                result = f"ERROR: unknown tool '{block.name}'"

            short = result[:300] + ("..." if len(result) > 300 else "")
            print(f"[result] {short}")

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    # Persist change log
    log_dir = os.path.join(flow_dir, "logs", platform, design, tag)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "loop_agent_changes.json")
    with open(log_path, "w") as f:
        json.dump(change_log, f, indent=2)
    print(f"\n[loop-agent] Change log → {log_path}")

    return change_log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="P&R closed-loop optimization agent")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--platform", help="Platform name (e.g. nangate45)")
    group.add_argument("--reports-dir", help="Direct path to reports directory")

    parser.add_argument("--design", help="Design name (required with --platform)")
    parser.add_argument("--tag", default="base", help="Tag / variant (default: base)")

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
        platform, design, tag = args.platform, args.design, args.tag
    else:
        parts = args.reports_dir.rstrip("/").split("/")
        tag, design, platform = parts[-1], parts[-2], parts[-3]
        flow_dir = args.flow_dir

    run_loop(platform, design, tag, flow_dir)


if __name__ == "__main__":
    main()
