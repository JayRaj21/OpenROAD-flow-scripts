#!/usr/bin/env python3
"""
CTS quality diagnostic.

Extracts clock-tree structural metrics (buffer count, sink count, skew) from
a completed CTS stage run and quantifies the "CTS->GRT parasitic
underestimation cliff" pattern documented in triage_agent.py: at CTS,
parasitics are estimated from placement (optimistic); real parasitics after
global route are worse, so WNS/TNS commonly degrade between the CTS and
Global route stages. This tool pulls both stages' timing from
pr_metrics.collect() and flags the transition quantitatively instead of
relying on prose.

Usage:
    python3 flow/util/cts_diagnostic.py --platform nangate45 --design ibex --tag base
    python3 flow/util/cts_diagnostic.py --reports-dir flow/reports/nangate45/ibex/base \
                                         --logs-dir    flow/logs/nangate45/ibex/base
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pr_metrics

# ---------------------------------------------------------------------------
# Grounding notes (see PR_EXTENSION_DEV_LOG.md for how these were verified
# against real checked-in flow/logs/.../4_1_cts.log and 4_1_cts.json files):
#
# - flow/scripts/cts.tcl runs clock_tree_synthesis (TritonCTS) as stage
#   "4_1_cts" (see flow/Makefile do-step(4_1_cts, ...)); flow.sh writes its
#   log to $LOG_DIR/4_1_cts.log and its metrics snapshot to $LOG_DIR/4_1_cts.json.
# - TritonCTS emits one "Created N clock buffers." line per clock net (the
#   final, cumulative buffer count for that net's H-tree) and one
#   "Sinks N" summary line per net (post dummy-load-balancing sink count).
#   Latency-balancing buffers are reported separately as
#   "Total number of delay buffers: N".
# - report_metrics.tcl (gated by REPORT_CLOCK_SKEW, default-on) calls
#   report_clock_skew_metric / report_clock_skew_metric -hold, which land in
#   the stage .json as keys ending "clock__skew__setup" / "clock__skew__hold"
#   (e.g. "cts__clock__skew__setup"). The .rpt text form (report_clock_skew,
#   no -hold flag from cts.tcl's call site) only carries setup skew, as
#   "<value> setup skew" — used as a fallback when the json is absent.
# ---------------------------------------------------------------------------

CTS_LOG_NAME = "4_1_cts.log"
CTS_JSON_NAME = "4_1_cts.json"
CTS_RPT_NAME = "4_cts_final.rpt"
CTS_STAGE_NAME = "CTS"
GRT_STAGE_NAME = "Global route"

_BUFFER_RE = re.compile(r"Created (\d+) clock buffers\.")
_DELAY_BUFFER_RE = re.compile(r"Total number of delay buffers:\s*(\d+)")
_SINKS_RE = re.compile(r"\]\s*Sinks\s+(\d+)\s*$")
_LEAF_BUFFER_RE = re.compile(r"\]\s*Leaf buffers\s+(\d+)\s*$")
_RPT_SETUP_SKEW_RE = re.compile(r"([\d.]+)\s+setup skew")
_RPT_HOLD_SKEW_RE = re.compile(r"([\d.]+)\s+hold skew")

DEFAULT_CLIFF_THRESHOLD_NS = 0.05
DEFAULT_BUFFER_RATIO_THRESHOLD = 0.5


def parse_cts_log(log_path):
    """Extract TritonCTS-inserted buffer and sink counts from the CTS log."""
    metrics = {}
    if not os.path.isfile(log_path):
        return metrics

    buffer_total = 0
    leaf_total = 0
    sink_total = 0
    found_buffers = False
    found_sinks = False

    with open(log_path) as f:
        for line in f:
            m = _BUFFER_RE.search(line)
            if m:
                buffer_total += int(m.group(1))
                found_buffers = True
                continue

            m = _LEAF_BUFFER_RE.search(line)
            if m:
                leaf_total += int(m.group(1))
                continue

            m = _SINKS_RE.search(line)
            if m:
                sink_total += int(m.group(1))
                found_sinks = True
                continue

            m = _DELAY_BUFFER_RE.search(line)
            if m:
                buffer_total += int(m.group(1))

    if found_buffers:
        metrics["buffer_count"] = buffer_total
    if leaf_total:
        metrics["leaf_buffer_count"] = leaf_total
    if found_sinks:
        metrics["sink_count"] = sink_total

    return metrics


def parse_cts_skew_json(json_path):
    """Extract setup/hold clock skew from the CTS stage metrics json."""
    metrics = {}
    if not os.path.isfile(json_path):
        return metrics
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return metrics

    for key, val in data.items():
        if key.endswith("clock__skew__setup"):
            metrics["setup_skew"] = val
        elif key.endswith("clock__skew__hold"):
            metrics["hold_skew"] = val
    return metrics


def parse_cts_skew_rpt(rpt_path):
    """Fallback: extract setup/hold clock skew from the CTS stage .rpt text."""
    metrics = {}
    if not os.path.isfile(rpt_path):
        return metrics
    with open(rpt_path) as f:
        content = f.read()

    m = _RPT_SETUP_SKEW_RE.search(content)
    if m:
        metrics["setup_skew"] = float(m.group(1))
    m = _RPT_HOLD_SKEW_RE.search(content)
    if m:
        metrics["hold_skew"] = float(m.group(1))
    return metrics


def gather(reports_dir, logs_dir):
    """Collect P&R stage rows plus CTS structural metrics."""
    rows = pr_metrics.collect(reports_dir, logs_dir)
    stage_map = dict(rows)

    structural = parse_cts_log(os.path.join(logs_dir, CTS_LOG_NAME))

    skew = parse_cts_skew_json(os.path.join(logs_dir, CTS_JSON_NAME))
    if not skew:
        skew = parse_cts_skew_rpt(os.path.join(reports_dir, CTS_RPT_NAME))
    structural.update(skew)

    return rows, stage_map, structural


def buffer_per_sink(structural):
    buffers = structural.get("buffer_count")
    sinks = structural.get("sink_count")
    if not buffers or not sinks:
        return None
    return buffers / sinks


def check_cliff(stage_map, threshold):
    """Compare CTS-stage vs. Global-route-stage WNS and flag a cliff.

    WNS is negative-is-worse; a "cliff" is the WNS getting more negative
    (worse) by more than `threshold` ns between CTS and Global route.
    """
    cts = stage_map.get(CTS_STAGE_NAME, {})
    grt = stage_map.get(GRT_STAGE_NAME, {})
    cts_wns = cts.get("wns")
    grt_wns = grt.get("wns")
    if cts_wns is None or grt_wns is None:
        return None

    drop = cts_wns - grt_wns
    return {
        "cts_wns": cts_wns,
        "grt_wns": grt_wns,
        "drop": drop,
        "detected": drop > threshold,
    }


def print_report(structural, cliff, buffer_ratio_threshold, label):
    print(f"\nCTS Quality Diagnostic — {label}")
    print("=" * 70)

    buffers = structural.get("buffer_count")
    sinks = structural.get("sink_count")
    ratio = buffer_per_sink(structural)

    print(
        f"Clock buffers/inverters inserted: {buffers if buffers is not None else '—'}"
    )
    print(f"Clock sinks:                      {sinks if sinks is not None else '—'}")
    if ratio is not None:
        print(f"Buffers per sink:                 {ratio:.3f}")
    else:
        print("Buffers per sink:                 —")

    setup_skew = structural.get("setup_skew")
    hold_skew = structural.get("hold_skew")
    print(
        f"Setup skew (ns):                  "
        f"{setup_skew if setup_skew is not None else '—'}"
    )
    print(
        f"Hold skew (ns):                    "
        f"{hold_skew if hold_skew is not None else '—'}"
    )

    print("-" * 70)

    over_buffered = ratio is not None and ratio > buffer_ratio_threshold
    if over_buffered:
        print(
            f"OVER-BUFFERING WARNING: buffers/sink {ratio:.3f} exceeds "
            f"threshold {buffer_ratio_threshold:.3f}"
        )

    if cliff is None:
        print("CTS->GRT cliff check: insufficient data (need CTS and GRT wns).")
    else:
        print(
            f"CTS WNS: {cliff['cts_wns']:+.3f} ns   "
            f"GRT WNS: {cliff['grt_wns']:+.3f} ns   "
            f"drop: {cliff['drop']:+.3f} ns"
        )
        if cliff["detected"]:
            print(
                "CLIFF DETECTED: WNS degraded by more than threshold between "
                "CTS and Global route — parasitics from placement estimate "
                "were optimistic relative to routed parasitics. Consider "
                "POST_CTS_TCL=post_cts_timing_repair.tcl."
            )
        else:
            print("No CTS->GRT cliff detected.")

    print()
    return over_buffered, (cliff is not None and cliff["detected"])


def main():
    parser = argparse.ArgumentParser(description="CTS quality diagnostic")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--platform", help="Platform name (e.g. nangate45)")
    group.add_argument("--reports-dir", help="Direct path to reports directory")

    parser.add_argument("--design", help="Design name (required with --platform)")
    parser.add_argument("--tag", help="Tag / variant (default: base)", default="base")
    parser.add_argument("--logs-dir", help="Direct path to logs directory")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    flow_dir = os.path.dirname(script_dir)
    parser.add_argument(
        "--flow-dir",
        default=flow_dir,
        help=f"Path to flow/ directory (default: {flow_dir})",
    )

    parser.add_argument(
        "--cliff-threshold",
        type=float,
        default=DEFAULT_CLIFF_THRESHOLD_NS,
        help=f"WNS degradation (ns) between CTS and GRT that counts as a "
        f"cliff (default: {DEFAULT_CLIFF_THRESHOLD_NS})",
    )
    parser.add_argument(
        "--buffer-ratio-threshold",
        type=float,
        default=DEFAULT_BUFFER_RATIO_THRESHOLD,
        help=f"Buffers-per-sink ratio above which the design is flagged as "
        f"over-buffered (default: {DEFAULT_BUFFER_RATIO_THRESHOLD})",
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

    _, stage_map, structural = gather(reports_dir, logs_dir)
    cliff = check_cliff(stage_map, args.cliff_threshold)

    over_buffered, cliff_detected = print_report(
        structural, cliff, args.buffer_ratio_threshold, label
    )

    sys.exit(1 if (over_buffered or cliff_detected) else 0)


if __name__ == "__main__":
    main()
