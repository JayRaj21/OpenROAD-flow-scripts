#!/usr/bin/env python3
"""
Regression / benchmark dashboard for P&R quality history.

Builds a history/regression layer on top of `pr_metrics.collect()` — it does
not re-parse ORFS reports or logs itself. Each `record` invocation appends one
JSON line to a per-design/tag history file; `report` reads that history back
and flags regressions against thresholds so it can gate a CI pipeline.

Usage:
    python3 flow/util/benchmark_dashboard.py record --platform nangate45 --design ibex --tag base
    python3 flow/util/benchmark_dashboard.py report --platform nangate45 --design ibex --tag base
    python3 flow/util/benchmark_dashboard.py report --platform nangate45 --design ibex --tag base \
        --stage "Global route" --last 10 --html out.html
"""

import argparse
import fcntl
import html
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from pr_metrics import collect

HISTORY_DIRNAME = "benchmark_history"

DEFAULT_WNS_THRESHOLD_NS = 0.01
DEFAULT_FMAX_THRESHOLD_PCT = 1.0
DEFAULT_OVERFLOW_THRESHOLD = 0.001


def history_dir(flow_util_dir):
    return os.path.join(flow_util_dir, HISTORY_DIRNAME)


def history_path(flow_util_dir, platform, design, tag):
    fname = f"{platform}__{design}__{tag}.jsonl"
    return os.path.join(history_dir(flow_util_dir), fname)


def git_sha(repo_dir):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError, FileNotFoundError):
        return None


def rows_to_stage_dict(rows):
    return {name: metrics for name, metrics in rows}


def append_record(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(record) + "\n"
    # flock rather than relying on OS atomic-append: a full multi-stage record
    # can exceed PIPE_BUF (4096 bytes), so plain O_APPEND no longer guarantees
    # writes from concurrent `record` invocations won't interleave.
    with open(path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_records(path):
    records = []
    if not os.path.isfile(path):
        return records
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"WARNING: skipping corrupt history line {lineno} in {path}: {e}",
                    file=sys.stderr,
                )
    return records


def overflow_of(stage_metrics):
    if "gp_overflow" in stage_metrics:
        return stage_metrics["gp_overflow"]
    return stage_metrics.get("grt_overflow")


def compute_delta(prev_metrics, cur_metrics, key):
    prev = prev_metrics.get(key) if prev_metrics else None
    cur = cur_metrics.get(key)
    if prev is None or cur is None:
        return None
    return cur - prev


def detect_regressions(
    prev_metrics, cur_metrics, wns_threshold, fmax_threshold_pct, overflow_threshold
):
    regressions = []

    wns_delta = compute_delta(prev_metrics, cur_metrics, "wns")
    if wns_delta is not None and wns_delta < -wns_threshold:
        regressions.append(
            f"REGRESSION: WNS worsened by {wns_delta:.4f} ns "
            f"(threshold {wns_threshold} ns)"
        )

    prev_fmax = prev_metrics.get("fmax_mhz") if prev_metrics else None
    cur_fmax = cur_metrics.get("fmax_mhz")
    if prev_fmax is not None and cur_fmax is not None and prev_fmax > 0:
        pct_drop = (prev_fmax - cur_fmax) / prev_fmax * 100.0
        if pct_drop > fmax_threshold_pct:
            regressions.append(
                f"REGRESSION: Fmax dropped {pct_drop:.2f}% "
                f"(threshold {fmax_threshold_pct}%)"
            )

    prev_overflow = overflow_of(prev_metrics) if prev_metrics else None
    cur_overflow = overflow_of(cur_metrics)
    if prev_overflow is not None and cur_overflow is not None:
        overflow_delta = cur_overflow - prev_overflow
        if overflow_delta > overflow_threshold:
            regressions.append(
                f"REGRESSION: routing overflow increased by {overflow_delta:.5f} "
                f"(threshold {overflow_threshold})"
            )

    return regressions


def best_ever(records, stage, key, better="lower"):
    values = []
    for r in records:
        v = r.get("stages", {}).get(stage, {}).get(key)
        if v is not None:
            values.append(v)
    if not values:
        return None
    return min(values) if better == "lower" else max(values)


def fmt(val, fmt_str, missing="—"):
    if val is None:
        return missing
    return fmt_str.format(val)


def fmt_delta(val, fmt_str, missing="—"):
    if val is None:
        return missing
    return fmt_str.format(val)


def build_report_rows(
    records, stage, wns_threshold, fmax_threshold_pct, overflow_threshold
):
    """Return (table_rows, regressions_against_latest)."""
    table_rows = []
    prev_metrics = None

    best_wns = best_ever(records, stage, "wns", "higher")
    best_fmax = best_ever(records, stage, "fmax_mhz", "higher")
    best_hpwl = best_ever(records, stage, "hpwl", "lower")

    for idx, rec in enumerate(records):
        metrics = rec.get("stages", {}).get(stage, {})

        wns_delta = compute_delta(prev_metrics, metrics, "wns")
        tns_delta = compute_delta(prev_metrics, metrics, "tns")
        fmax_delta = compute_delta(prev_metrics, metrics, "fmax_mhz")
        hpwl_delta = compute_delta(prev_metrics, metrics, "hpwl")

        flags = []
        if (
            best_wns is not None
            and metrics.get("wns") is not None
            and metrics["wns"] < best_wns
        ):
            flags.append("worse-than-best-WNS")
        if (
            best_fmax is not None
            and metrics.get("fmax_mhz") is not None
            and metrics["fmax_mhz"] < best_fmax
        ):
            flags.append("worse-than-best-Fmax")
        if (
            best_hpwl is not None
            and metrics.get("hpwl") is not None
            and metrics["hpwl"] > best_hpwl
        ):
            flags.append("worse-than-best-HPWL")

        regressions = []
        if idx > 0:
            regressions = detect_regressions(
                prev_metrics,
                metrics,
                wns_threshold,
                fmax_threshold_pct,
                overflow_threshold,
            )

        table_rows.append(
            {
                "record": rec,
                "metrics": metrics,
                "wns_delta": wns_delta,
                "tns_delta": tns_delta,
                "fmax_delta": fmax_delta,
                "hpwl_delta": hpwl_delta,
                "flags": flags,
                "regressions": regressions,
            }
        )
        prev_metrics = metrics

    latest_regressions = table_rows[-1]["regressions"] if table_rows else []
    return table_rows, latest_regressions


def print_report(records, stage, table_rows, label):
    print(f"\nBenchmark history — {label} — stage: {stage}")
    print("=" * 110)
    header = (
        f"{'Timestamp':<21} {'SHA':<9} {'WNS (ns)':>10} {'dWNS':>8} "
        f"{'Fmax(MHz)':>10} {'dFmax':>8} {'HPWL':>12} {'dHPWL':>10} {'Flags':<24}"
    )
    print(header)
    print("-" * 110)

    for row in table_rows:
        rec = row["record"]
        m = row["metrics"]
        ts = rec.get("timestamp", "—")[:19]
        sha = (rec.get("git_sha") or "—")[:8]
        wns = fmt(m.get("wns"), "{:+.3f}")
        dwns = fmt_delta(row["wns_delta"], "{:+.3f}")
        fmax = fmt(m.get("fmax_mhz"), "{:.1f}")
        dfmax = fmt_delta(row["fmax_delta"], "{:+.1f}")
        hpwl = fmt(m.get("hpwl"), "{:,.0f}")
        dhpwl = fmt_delta(row["hpwl_delta"], "{:+,.0f}")
        flags = ",".join(row["flags"]) if row["flags"] else ""

        print(
            f"{ts:<21} {sha:<9} {wns:>10} {dwns:>8} {fmax:>10} {dfmax:>8} "
            f"{hpwl:>12} {dhpwl:>10} {flags:<24}"
        )

    print("-" * 110)

    latest_regressions = table_rows[-1]["regressions"] if table_rows else []
    if latest_regressions:
        print()
        for r in latest_regressions:
            print(r)
    elif len(table_rows) >= 2:
        print("\nNo regressions detected against previous record.")
    else:
        print("\nOnly one record present — nothing to compare against yet.")
    print()


def render_html(records, stage, table_rows, label, out_path):
    width, height = 760, 260
    pad_left, pad_right, pad_top, pad_bottom = 60, 20, 20, 30

    def series(key):
        return [
            (i, row["metrics"].get(key))
            for i, row in enumerate(table_rows)
            if row["metrics"].get(key) is not None
        ]

    def svg_for(key, title, color):
        pts = series(key)
        if len(pts) < 2:
            return f"<p>Not enough data to chart {title}.</p>"
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if y_min == y_max:
            y_min -= 1
            y_max += 1
        x_span = max(x_max - x_min, 1)
        y_span = y_max - y_min

        def sx(x):
            return pad_left + (x - x_min) / x_span * (width - pad_left - pad_right)

        def sy(y):
            return (
                height
                - pad_bottom
                - (y - y_min) / y_span * (height - pad_top - pad_bottom)
            )

        poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        circles = "".join(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}"/>'
            for x, y in pts
        )
        return f"""
        <div class="chart">
          <h3>{html.escape(title)}</h3>
          <svg viewBox="0 0 {width} {height}" width="100%" height="{height}">
            <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" stroke="#888"/>
            <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#888"/>
            <polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2"/>
            {circles}
            <text x="{pad_left}" y="{pad_top - 5}" font-size="11" fill="#333">max {y_max:.4g}</text>
            <text x="{pad_left}" y="{height - pad_bottom + 15}" font-size="11" fill="#333">min {y_min:.4g}</text>
          </svg>
        </div>
        """

    charts = (
        svg_for("wns", "WNS (ns)", "#c0392b")
        + svg_for("fmax_mhz", "Fmax (MHz)", "#2471a3")
        + svg_for("hpwl", "HPWL", "#27ae60")
    )

    rows_html = ""
    for row in table_rows:
        rec = row["record"]
        m = row["metrics"]
        flags = ", ".join(row["flags"]) if row["flags"] else ""
        regressions = "<br>".join(html.escape(r) for r in row["regressions"])
        rows_html += (
            "<tr>"
            f"<td>{html.escape(str(rec.get('timestamp', ''))[:19])}</td>"
            f"<td>{html.escape(str(rec.get('git_sha') or '')[:8])}</td>"
            f"<td>{fmt(m.get('wns'), '{:+.3f}')}</td>"
            f"<td>{fmt(m.get('fmax_mhz'), '{:.1f}')}</td>"
            f"<td>{fmt(m.get('hpwl'), '{:,.0f}')}</td>"
            f"<td>{html.escape(flags)}</td>"
            f"<td>{regressions}</td>"
            "</tr>\n"
        )

    safe_label = html.escape(label)
    safe_stage = html.escape(stage)

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Benchmark dashboard — {safe_label}</title>
<style>
  body {{ font-family: sans-serif; margin: 2em; color: #222; background: #fff; }}
  h1 {{ font-size: 1.3em; }}
  .chart {{ margin-bottom: 1.5em; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; font-size: 0.85em; text-align: left; }}
  th {{ background: #f0f0f0; }}
</style>
</head>
<body>
<h1>Benchmark dashboard — {safe_label} — stage: {safe_stage}</h1>
{charts}
<table>
<thead><tr><th>Timestamp</th><th>SHA</th><th>WNS</th><th>Fmax</th><th>HPWL</th><th>Flags</th><th>Regressions</th></tr></thead>
<tbody>
{rows_html}
</tbody>
</table>
</body>
</html>
"""
    with open(out_path, "w") as f:
        f.write(doc)


def cmd_record(args, flow_dir, flow_util_dir, reports_dir, logs_dir, label):
    if not os.path.isdir(reports_dir):
        print(f"ERROR: reports directory not found: {reports_dir}", file=sys.stderr)
        sys.exit(1)

    rows = collect(reports_dir, logs_dir)
    repo_dir = os.path.dirname(flow_dir)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(repo_dir),
        "platform": args.platform,
        "design": args.design,
        "tag": args.tag,
        "stages": rows_to_stage_dict(rows),
    }

    path = history_path(flow_util_dir, args.platform, args.design, args.tag)
    append_record(path, record)
    print(f"Recorded benchmark for {label} -> {path}")


def cmd_report(args, flow_dir, flow_util_dir, reports_dir, logs_dir, label):
    path = history_path(flow_util_dir, args.platform, args.design, args.tag)
    records = load_records(path)

    if not records:
        print(f"No history found at {path}")
        sys.exit(0)

    stage = args.stage
    # best-ever and deltas are computed over the FULL history so that --last
    # only narrows what's displayed, never what "worse than all-time best"
    # or the regression check against the immediately-previous record means.
    table_rows, latest_regressions = build_report_rows(
        records,
        stage,
        args.wns_threshold,
        args.fmax_threshold_pct,
        args.overflow_threshold,
    )

    display_rows = table_rows[-args.last :] if args.last else table_rows

    print_report(records, stage, display_rows, label)

    if args.html:
        render_html(records, stage, display_rows, label, args.html)
        print(f"Wrote HTML dashboard: {args.html}")

    sys.exit(1 if latest_regressions else 0)


def add_common_args(parser, flow_dir_default):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--platform", help="Platform name (e.g. nangate45)")
    group.add_argument("--reports-dir", help="Direct path to reports directory")

    parser.add_argument("--design", help="Design name (required with --platform)")
    parser.add_argument("--tag", help="Tag / variant (default: base)", default="base")
    parser.add_argument("--logs-dir", help="Direct path to logs directory")
    parser.add_argument(
        "--flow-dir",
        default=flow_dir_default,
        help=f"Path to flow/ directory (default: {flow_dir_default})",
    )


def resolve_dirs(args):
    if args.platform:
        if not args.design:
            raise SystemExit("--design is required when using --platform")
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
        if not args.platform or not args.design:
            parts = os.path.normpath(reports_dir).split(os.sep)
            if len(parts) >= 3:
                args.platform = args.platform or parts[-3]
                args.design = args.design or parts[-2]
                args.tag = args.tag or parts[-1]

        if not args.platform or not args.design:
            raise SystemExit(
                "error: could not determine --platform/--design from "
                f"--reports-dir {reports_dir!r} (need at least "
                "<platform>/<design>/<tag> path components); pass "
                "--platform and --design explicitly"
            )
    return reports_dir, logs_dir, label


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    flow_dir_default = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(description="Regression / benchmark dashboard")
    sub = parser.add_subparsers(dest="command", required=True)

    p_record = sub.add_parser("record", help="Record one run's metrics into history")
    add_common_args(p_record, flow_dir_default)

    p_report = sub.add_parser(
        "report", help="Print history report and detect regressions"
    )
    add_common_args(p_report, flow_dir_default)
    p_report.add_argument(
        "--stage", default="Finish", help="Stage name to report on (default: Finish)"
    )
    p_report.add_argument(
        "--last", type=int, default=None, help="Limit to N most recent records"
    )
    p_report.add_argument(
        "--html", help="Write a self-contained HTML dashboard to this path"
    )
    p_report.add_argument(
        "--wns-threshold",
        type=float,
        default=DEFAULT_WNS_THRESHOLD_NS,
        help=f"WNS regression threshold in ns (default: {DEFAULT_WNS_THRESHOLD_NS})",
    )
    p_report.add_argument(
        "--fmax-threshold-pct",
        type=float,
        default=DEFAULT_FMAX_THRESHOLD_PCT,
        help=f"Fmax regression threshold in %% (default: {DEFAULT_FMAX_THRESHOLD_PCT})",
    )
    p_report.add_argument(
        "--overflow-threshold",
        type=float,
        default=DEFAULT_OVERFLOW_THRESHOLD,
        help=f"Routing overflow regression threshold (default: {DEFAULT_OVERFLOW_THRESHOLD})",
    )

    args = parser.parse_args()
    flow_dir = args.flow_dir
    flow_util_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir, logs_dir, label = resolve_dirs(args)

    if args.command == "record":
        cmd_record(args, flow_dir, flow_util_dir, reports_dir, logs_dir, label)
    elif args.command == "report":
        cmd_report(args, flow_dir, flow_util_dir, reports_dir, logs_dir, label)


if __name__ == "__main__":
    main()
