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
# against a real, locally-generated flow/logs/.../4_1_cts.log and
# 4_1_cts.json from an actual `clock_tree_synthesis` run. flow/logs and
# flow/reports are gitignored build output, not committed to this repo, so
# these exact files are not reproducible from a clean checkout without
# running the flow yourself):
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

# Exit codes (see --help epilog / main() for the full scheme):
EXIT_CLEAN = 0
EXIT_FINDING = 1
EXIT_USAGE_ERROR = 2
EXIT_INTERNAL_ERROR = 3

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
# TNS scales with design size (total over all violating endpoints), so unlike
# WNS a pure absolute-ns threshold isn't meaningful across designs; a
# relative (percentage) degradation vs. the CTS-stage TNS is used as the
# primary signal. But on near-zero-baseline designs (e.g. a CTS TNS of
# -0.001ns) that percentage blows up to hundreds/thousands of percent (or
# infinite, when CTS TNS is exactly 0) for a numerically negligible
# picosecond-scale change, so a cliff additionally requires the absolute
# drop to clear a small ns floor. Calibrated against real ORFS runs under
# flow/reports: nangate45/dynamic_node/base (+0.06ns, +8.6%) and
# nangate45/jpeg/base (+5.34ns, +13.3%) are real cliffs that must clear both
# bars; nangate45/aes/base (+0.01ns, "+inf%") is timing-clean noise that
# must clear neither.
DEFAULT_TNS_CLIFF_THRESHOLD_PCT = 5.0
DEFAULT_TNS_CLIFF_THRESHOLD_ABS_NS = 0.03


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

    if not isinstance(data, dict):
        print(
            f"WARNING: {json_path} does not contain a JSON object at the top "
            "level (got "
            f"{type(data).__name__}); skipping CTS skew extraction from it.",
            file=sys.stderr,
        )
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


def derive_logs_dir(reports_dir):
    """Best-effort sibling logs/ dir for a given reports_dir.

    Replaces the "reports" path component with "logs" (matching ORFS'
    flow/reports/<platform>/<design>/<tag> <-> flow/logs/<platform>/<design>/<tag>
    layout) rather than doing a naive substring replace, which silently
    no-ops when reports_dir doesn't contain the literal "/reports/" (e.g. a
    relative path given from within the flow/ directory itself). Falls back
    to a "logs" directory next to reports_dir if no "reports" component is
    found at all.
    """
    abs_reports_dir = os.path.abspath(reports_dir)
    parts = abs_reports_dir.split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "reports":
            return os.sep.join(parts[:i] + ["logs"] + parts[i + 1 :])
    return os.path.join(os.path.dirname(abs_reports_dir), "logs")


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


def check_cliff(
    stage_map,
    threshold,
    tns_threshold_pct=DEFAULT_TNS_CLIFF_THRESHOLD_PCT,
    tns_threshold_abs=DEFAULT_TNS_CLIFF_THRESHOLD_ABS_NS,
):
    """Compare CTS-stage vs. Global-route-stage WNS and TNS and flag a cliff.

    WNS and TNS are both negative-is-worse. A "cliff" is flagged if EITHER:
      - WNS gets more negative (worse) by more than `threshold` ns, or
      - TNS gets more negative (worse) by more than `tns_threshold_pct`
        percent (relative to the CTS-stage TNS magnitude) AND by more than
        `tns_threshold_abs` ns
    between CTS and Global route. TNS uses a relative threshold because TNS
    magnitude scales with design size, but the percentage alone false-flags
    near-zero-baseline designs (a CTS TNS of e.g. -0.001ns turns a
    picosecond-scale, timing-clean wobble into a huge or infinite percent
    swing), so an absolute-ns floor is required in addition to the
    percentage bar.
    """
    cts = stage_map.get(CTS_STAGE_NAME, {})
    grt = stage_map.get(GRT_STAGE_NAME, {})
    cts_wns = cts.get("wns")
    grt_wns = grt.get("wns")
    if cts_wns is None or grt_wns is None:
        return None

    drop = cts_wns - grt_wns
    wns_detected = drop > threshold

    cts_tns = cts.get("tns")
    grt_tns = grt.get("tns")
    tns_drop = None
    tns_drop_pct = None
    tns_detected = False
    if cts_tns is not None and grt_tns is not None:
        tns_drop = cts_tns - grt_tns
        if cts_tns != 0:
            tns_drop_pct = (tns_drop / abs(cts_tns)) * 100.0
        else:
            tns_drop_pct = float("inf") if tns_drop > 0 else 0.0
        tns_detected = tns_drop_pct > tns_threshold_pct and tns_drop > tns_threshold_abs

    return {
        "cts_wns": cts_wns,
        "grt_wns": grt_wns,
        "drop": drop,
        "wns_detected": wns_detected,
        "cts_tns": cts_tns,
        "grt_tns": grt_tns,
        "tns_drop": tns_drop,
        "tns_drop_pct": tns_drop_pct,
        "tns_detected": tns_detected,
        "detected": wns_detected or tns_detected,
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
        if cliff["cts_tns"] is not None and cliff["grt_tns"] is not None:
            print(
                f"CTS TNS: {cliff['cts_tns']:+.3f} ns   "
                f"GRT TNS: {cliff['grt_tns']:+.3f} ns   "
                f"drop: {cliff['tns_drop']:+.3f} ns ({cliff['tns_drop_pct']:+.1f}%)"
            )
        else:
            print("CTS TNS: —   GRT TNS: —   drop: — (need CTS and GRT tns)")

        if cliff["detected"]:
            triggers = []
            if cliff["wns_detected"]:
                triggers.append("WNS")
            if cliff["tns_detected"]:
                triggers.append("TNS")
            print(
                f"CLIFF DETECTED ({'/'.join(triggers)} degraded by more than "
                "threshold) between CTS and Global route — parasitics from "
                "placement estimate were optimistic relative to routed "
                "parasitics. Consider POST_CTS_TCL=post_cts_timing_repair.tcl."
            )
        else:
            print("No CTS->GRT cliff detected.")

    print()
    return over_buffered, (cliff is not None and cliff["detected"])


def _main():
    parser = argparse.ArgumentParser(
        description="CTS quality diagnostic",
        epilog="Exit codes: 0 = clean, 1 = finding detected (cliff and/or "
        "over-buffering), 2 = usage/input error (bad args, missing "
        "reports dir), 3 = internal error (unexpected exception/crash).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    parser.add_argument(
        "--tns-cliff-threshold",
        type=float,
        default=DEFAULT_TNS_CLIFF_THRESHOLD_PCT,
        help=f"TNS degradation (percent, relative to CTS-stage TNS) between "
        f"CTS and GRT that counts as a cliff; must be exceeded together "
        f"with --tns-cliff-threshold-abs (default: "
        f"{DEFAULT_TNS_CLIFF_THRESHOLD_PCT})",
    )
    parser.add_argument(
        "--tns-cliff-threshold-abs",
        type=float,
        default=DEFAULT_TNS_CLIFF_THRESHOLD_ABS_NS,
        help=f"Minimum absolute TNS degradation (ns) between CTS and GRT "
        f"required for a TNS cliff, in addition to --tns-cliff-threshold; "
        f"guards against near-zero-baseline designs where a tiny ns change "
        f"is a huge or infinite percentage (default: "
        f"{DEFAULT_TNS_CLIFF_THRESHOLD_ABS_NS})",
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
        logs_dir = args.logs_dir or derive_logs_dir(reports_dir)
        label = reports_dir

    if not os.path.isdir(reports_dir):
        print(f"ERROR: reports directory not found: {reports_dir}", file=sys.stderr)
        sys.exit(EXIT_USAGE_ERROR)

    if not os.path.isdir(logs_dir):
        print(
            f"WARNING: logs directory not found: {logs_dir} — structural CTS "
            "metrics (buffer/sink counts, skew fallback) and log-based P&R "
            "metrics will be unavailable; pass --logs-dir explicitly if this "
            "is unexpected.",
            file=sys.stderr,
        )

    _, stage_map, structural = gather(reports_dir, logs_dir)
    cliff = check_cliff(
        stage_map,
        args.cliff_threshold,
        args.tns_cliff_threshold,
        args.tns_cliff_threshold_abs,
    )

    over_buffered, cliff_detected = print_report(
        structural, cliff, args.buffer_ratio_threshold, label
    )

    sys.exit(EXIT_FINDING if (over_buffered or cliff_detected) else EXIT_CLEAN)


def main():
    try:
        _main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"INTERNAL ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(EXIT_INTERNAL_ERROR)


if __name__ == "__main__":
    main()
