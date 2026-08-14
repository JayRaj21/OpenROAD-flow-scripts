#!/usr/bin/env python3
"""
P&R quality metric aggregator.

Reads existing ORFS report and log files for a completed design run and
prints a stage-by-stage table showing how timing, wirelength, and routing
congestion evolve across the P&R flow.

Usage:
    python3 flow/util/pr_metrics.py --platform nangate45 --design ibex --tag base
    python3 flow/util/pr_metrics.py --reports-dir flow/reports/nangate45/ibex/base \
                                     --logs-dir    flow/logs/nangate45/ibex/base
"""

import argparse
import os
import re
import sys


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _grep(path, pattern):
    """Return first match of pattern in file, or None."""
    if not os.path.isfile(path):
        return None
    rx = re.compile(pattern)
    with open(path) as f:
        for line in f:
            m = rx.search(line)
            if m:
                return m
    return None


def _grep_last(path, pattern):
    """Return last match of pattern in file, or None."""
    if not os.path.isfile(path):
        return None
    rx = re.compile(pattern)
    last = None
    with open(path) as f:
        for line in f:
            m = rx.search(line)
            if m:
                last = m
    return last


def _float(m, group=1):
    try:
        return float(m.group(group))
    except (AttributeError, IndexError, ValueError):
        return None


def parse_rpt(rpt_path):
    """Extract WNS, TNS, worst_slack, fmax, total_power from a stage .rpt file."""
    metrics = {}
    if not os.path.isfile(rpt_path):
        return metrics

    with open(rpt_path) as f:
        content = f.read()

    # tns / wns: the first occurrence of "tns max <val>" etc.
    m = re.search(r"tns\s+max\s+([-\d.]+)", content)
    if m:
        metrics["tns"] = float(m.group(1))

    m = re.search(r"wns\s+max\s+([-\d.]+)", content)
    if m:
        metrics["wns"] = float(m.group(1))

    m = re.search(r"worst slack\s+max\s+([-\d.]+)", content)
    if m:
        metrics["worst_slack"] = float(m.group(1))

    m = re.search(r"fmax\s*=\s*([\d.]+)", content)
    if m:
        metrics["fmax_mhz"] = float(m.group(1))

    # Total power (last occurrence — power section is at the end)
    for m in re.finditer(r"^Total\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)",
                         content, re.MULTILINE):
        metrics["total_power_w"] = float(m.group(4))

    return metrics


def parse_gp_log(log_path):
    """Extract final HPWL and final routing overflow from a global-place log."""
    metrics = {}
    if not os.path.isfile(log_path):
        return metrics

    # Last HPWL line wins (final iteration)
    m = _grep_last(log_path, r"HPWL:\s*([\d]+)")
    if m:
        metrics["hpwl"] = int(m.group(1))

    # Last overflow line wins
    m = _grep_last(log_path, r"Total routing overflow:\s*([\d.]+)")
    if m:
        metrics["gp_overflow"] = float(m.group(1))

    return metrics


def parse_grt_log(log_path):
    """Extract final routing overflow from a global-route log."""
    metrics = {}
    if not os.path.isfile(log_path):
        return metrics

    m = _grep_last(log_path, r"Total routing overflow:\s*([\d.]+)")
    if m:
        metrics["grt_overflow"] = float(m.group(1))

    m = _grep_last(log_path, r"Number of overflowed tiles:\s*(\d+)")
    if m:
        metrics["grt_overflow_tiles"] = int(m.group(1))

    return metrics


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGES = [
    {
        "name": "Global place",
        "rpt":  "3_global_place.rpt",
        "log":  "3_3_place_gp.log",
        "log_parser": "gp",
    },
    {
        "name": "Resizer",
        "rpt":  "3_resizer.rpt",
    },
    {
        "name": "Detail place",
        "rpt":  "3_detailed_place.rpt",
    },
    {
        "name": "CTS",
        "rpt":  "4_cts_final.rpt",
    },
    {
        "name": "Global route",
        "rpt":  "5_global_route.rpt",
        "log":  "5_1_grt.log",
        "log_parser": "grt",
    },
    {
        "name": "Finish",
        "rpt":  "6_finish.rpt",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect(reports_dir, logs_dir):
    rows = []
    for stage in STAGES:
        rpt_path = os.path.join(reports_dir, stage["rpt"])
        m = parse_rpt(rpt_path)

        if stage.get("log"):
            log_path = os.path.join(logs_dir, stage["log"])
            if stage.get("log_parser") == "gp":
                m.update(parse_gp_log(log_path))
            elif stage.get("log_parser") == "grt":
                m.update(parse_grt_log(log_path))

        rows.append((stage["name"], m))
    return rows


def fmt(val, fmt_str, missing="—"):
    if val is None:
        return missing
    return fmt_str.format(val)


def print_table(rows, design_label):
    print(f"\nP&R Quality Trajectory — {design_label}")
    print("=" * 90)

    header = (
        f"{'Stage':<16} {'WNS (ns)':>10} {'TNS (ns)':>10} {'Worst slack':>12} "
        f"{'Fmax (MHz)':>11} {'HPWL (um)':>12} {'GRT overflow':>13}"
    )
    print(header)
    print("-" * 90)

    for name, m in rows:
        wns        = fmt(m.get("wns"),           "{:+.3f}")
        tns        = fmt(m.get("tns"),           "{:+.3f}")
        ws         = fmt(m.get("worst_slack"),   "{:+.3f}")
        fmax       = fmt(m.get("fmax_mhz"),      "{:.1f}")
        hpwl       = fmt(m.get("hpwl"),          "{:,.0f}")
        overflow   = fmt(m.get("gp_overflow") if "gp_overflow" in m
                        else m.get("grt_overflow"), "{:.4f}")

        print(f"{name:<16} {wns:>10} {tns:>10} {ws:>12} {fmax:>11} {hpwl:>12} {overflow:>13}")

    print("-" * 90)

    # Power summary from the last stage that has it
    power_w = None
    for _, m in reversed(rows):
        if m.get("total_power_w") is not None:
            power_w = m["total_power_w"]
            break
    if power_w is not None:
        print(f"\nTotal power (post-route):  {power_w:.4e} W")

    print()


def main():
    parser = argparse.ArgumentParser(description="P&R stage-by-stage quality report")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--platform", help="Platform name (e.g. nangate45)")
    group.add_argument("--reports-dir", help="Direct path to reports directory")

    parser.add_argument("--design",  help="Design name (required with --platform)")
    parser.add_argument("--tag",     help="Tag / variant (default: base)", default="base")
    parser.add_argument("--logs-dir", help="Direct path to logs directory")

    # Root of the ORFS repo; defaults to the directory two levels above this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    flow_dir = os.path.dirname(script_dir)
    parser.add_argument("--flow-dir", default=flow_dir,
                        help=f"Path to flow/ directory (default: {flow_dir})")

    args = parser.parse_args()

    if args.platform:
        if not args.design:
            parser.error("--design is required when using --platform")
        reports_dir = os.path.join(args.flow_dir, "reports", args.platform, args.design, args.tag)
        logs_dir    = os.path.join(args.flow_dir, "logs",    args.platform, args.design, args.tag)
        label = f"{args.platform}/{args.design}/{args.tag}"
    else:
        reports_dir = args.reports_dir
        logs_dir    = args.logs_dir or reports_dir.replace("/reports/", "/logs/")
        label = reports_dir

    if not os.path.isdir(reports_dir):
        print(f"ERROR: reports directory not found: {reports_dir}", file=sys.stderr)
        sys.exit(1)

    rows = collect(reports_dir, logs_dir)
    print_table(rows, label)


if __name__ == "__main__":
    main()
