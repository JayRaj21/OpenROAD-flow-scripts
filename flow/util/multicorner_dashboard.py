#!/usr/bin/env python3
"""
Multi-corner / multi-mode timing dashboard.

Builds on flow/util/pr_metrics.py and ORFS's multi-corner STA support
(flow/scripts/read_liberty.tcl, flow/scripts/report_multicorner_timing.tcl).
report_multicorner_timing.tcl writes one <stage>_<when>_multicorner_<corner>.rpt
file per corner when REPORT_MULTICORNER_TIMING is set and CORNERS has more
than one entry; this script finds those files for a given stage, parses
each with pr_metrics.parse_rpt(), and prints a per-corner comparison table
marking the worst corner for each metric.

Usage:
    python3 flow/util/multicorner_dashboard.py --platform nangate45 --design aes --tag base
    python3 flow/util/multicorner_dashboard.py --platform nangate45 --design aes --stage 6_finish
    python3 flow/util/multicorner_dashboard.py --reports-dir flow/reports/nangate45/aes/base
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr_metrics import parse_rpt

MULTICORNER_RE = re.compile(r"^(?P<stage>.+)_multicorner_(?P<corner>[^_.]+)\.rpt$")

# Higher-is-worse for these; worst_slack/wns/clock skew are more negative = worse.
WORSE_IS_LOWER = {"tns", "wns", "worst_slack", "clock_skew"}

METRIC_ROWS = [
    ("wns", "WNS (ns)", "{:+.3f}"),
    ("tns", "TNS (ns)", "{:+.3f}"),
    ("worst_slack", "Worst slack (ns)", "{:+.3f}"),
    ("clock_skew", "Clock skew (ns)", "{:+.3f}"),
]


def parse_clock_skew(rpt_path):
    """Extract worst clock skew from a per-corner .rpt file, if present."""
    if not os.path.isfile(rpt_path):
        return None
    with open(rpt_path) as f:
        content = f.read()
    # report_clock_skew's summary line looks like "Worst skew -0.123"
    m = re.search(r"[Ww]orst\s+skew\s+([-\d.]+)", content)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def find_multicorner_reports(reports_dir, stage):
    """Return {corner: rpt_path} for the given stage prefix."""
    pattern = os.path.join(reports_dir, "*multicorner_*.rpt")
    results = {}
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        m = MULTICORNER_RE.match(base)
        if not m:
            continue
        file_stage = m.group("stage")
        if stage is not None and file_stage != stage:
            continue
        results[m.group("corner")] = path
    return results


def default_stage(reports_dir):
    """Pick the stage with the highest numeric prefix among available multicorner files."""
    pattern = os.path.join(reports_dir, "*multicorner_*.rpt")
    stages = set()
    for path in glob.glob(pattern):
        m = MULTICORNER_RE.match(os.path.basename(path))
        if m:
            stages.add(m.group("stage"))
    if not stages:
        return None

    def sort_key(s):
        m = re.match(r"(\d+)", s)
        return int(m.group(1)) if m else -1

    return sorted(stages, key=sort_key)[-1]


def collect_per_corner(reports_dir, stage):
    """Return {corner: metrics_dict} for all corners found for the given stage."""
    rpt_files = find_multicorner_reports(reports_dir, stage)
    data = {}
    for corner, path in rpt_files.items():
        metrics = parse_rpt(path)
        skew = parse_clock_skew(path)
        if skew is not None:
            metrics["clock_skew"] = skew
        data[corner] = metrics
    return data


def worst_corner(data, metric):
    """Return the corner name with the worst value for metric, or None."""
    candidates = [(c, m[metric]) for c, m in data.items() if metric in m]
    if not candidates:
        return None
    if metric in WORSE_IS_LOWER:
        return min(candidates, key=lambda cv: cv[1])[0]
    return max(candidates, key=lambda cv: cv[1])[0]


def build_table(data, stage_label):
    """Return the dashboard table as a single string."""
    corners = sorted(data.keys())
    lines = []
    lines.append(f"\nMulti-corner timing dashboard — stage: {stage_label}")
    lines.append("=" * (24 + 18 * len(corners)))

    header = f"{'Metric':<22}" + "".join(f"{c:>18}" for c in corners)
    lines.append(header)
    lines.append("-" * len(header))

    for key, label, fmt_str in METRIC_ROWS:
        if not any(key in data[c] for c in corners):
            continue
        worst = worst_corner(data, key)
        cells = []
        for c in corners:
            if key not in data[c]:
                cells.append("—")
                continue
            val_str = fmt_str.format(data[c][key])
            if c == worst:
                val_str += " (worst)"
            cells.append(val_str)
        row = f"{label:<22}" + "".join(f"{cell:>18}" for cell in cells)
        lines.append(row)

    lines.append("-" * len(header))
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-corner / multi-mode timing dashboard"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--platform", help="Platform name (e.g. nangate45)")
    group.add_argument("--reports-dir", help="Direct path to reports directory")

    parser.add_argument("--design", help="Design name (required with --platform)")
    parser.add_argument("--tag", help="Tag / variant (default: base)", default="base")
    parser.add_argument(
        "--stage",
        help="Stage prefix to read, e.g. '6_finish' (default: latest stage found)",
        default=None,
    )

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
        label = f"{args.platform}/{args.design}/{args.tag}"
    else:
        reports_dir = args.reports_dir
        label = reports_dir

    if not os.path.isdir(reports_dir):
        print(f"ERROR: reports directory not found: {reports_dir}", file=sys.stderr)
        sys.exit(1)

    stage = args.stage or default_stage(reports_dir)
    if stage is None:
        print(
            "No multi-corner reports found — was REPORT_MULTICORNER_TIMING set "
            "and CORNERS multi-valued?",
            file=sys.stderr,
        )
        sys.exit(1)

    data = collect_per_corner(reports_dir, stage)
    if not data:
        print(
            f"No multi-corner reports found for stage '{stage}' — was "
            "REPORT_MULTICORNER_TIMING set and CORNERS multi-valued?",
            file=sys.stderr,
        )
        sys.exit(1)

    print(build_table(data, f"{label} / {stage}"))


if __name__ == "__main__":
    main()
