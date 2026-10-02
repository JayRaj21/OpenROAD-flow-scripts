"""
Renders arch_sweep.py's `--analyze` output (the `*_report_data.json` files
it writes alongside each `*_summary.md`) as a single local HTML page with
charts, so the bake-off result can be read at a glance instead of parsed
out of a markdown table.

This is a pure renderer: every number it draws comes straight from the
`_verdict()` stats already computed and printed by `--analyze`. It does
not recompute any statistic, so the chart can never show a number that
disagrees with the text summary.

Usage (from flow/):
  python3 util/ml/congestion/training/arch_sweep.py --track thermal \\
      --out util/ml/congestion/experiments/arch_sweep.json --analyze
  python3 util/ml/congestion/training/arch_sweep.py --track irdrop \\
      --out util/ml/congestion/experiments/irdrop_arch_sweep.json --analyze

  python3 util/ml/congestion/training/view_report.py \\
      util/ml/congestion/experiments/arch_sweep_report_data.json \\
      util/ml/congestion/experiments/irdrop_arch_sweep_report_data.json \\
      --out util/ml/congestion/experiments/bakeoff_report.html

Then open the file in any browser. One or two report_data.json files may
be given (one renders a single track; two renders both as sections on one
page); each must already exist (produced by a prior `--analyze` run).
"""

import argparse
import html
import json
import os

# Status colors (fixed role, never used for anything else): matches this
# codebase's chart convention (DESIGN_RUNS.md plots use the same pair).
VERDICT_COLOR = {
    "Better": "#0ca30c",
    "Worse": "#d03b3b",
    "Indistinguishable": "#8a8a86",
}
# Diverging pair for the delta bars: blue = better-than-baseline (negative
# delta, lower error), red = worse-than-baseline (positive delta).
BLUE = "#2a78d6"
RED = "#e34948"
GRAY_MID = "#c9c8c2"


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    try:
        if x != x:  # NaN
            return "n/a"
    except TypeError:
        return str(x)
    return f"{x:.{nd}f}"


def _bar_row(label, candidate_label, delta, scale, verdict, note=""):
    """One diverging bar: a horizontal track centered on 0, a filled bar
    reaching from 0 to `delta` (clamped to +-scale), colored by sign, plus
    the verdict badge and the raw number as direct labels."""
    half = 140.0
    clamped = max(-scale, min(scale, delta))
    bar_w = abs(clamped) / scale * half
    color = BLUE if delta < 0 else RED if delta > 0 else GRAY_MID
    bar_x = half - bar_w if delta < 0 else half
    vcolor = VERDICT_COLOR.get(verdict, "#8a8a86")
    return f"""
    <div class="bar-row" title="{html.escape(candidate_label)}: delta={delta:.6f} ({verdict}){html.escape(note)}">
      <div class="bar-label">{html.escape(label)}</div>
      <div class="bar-track">
        <svg viewBox="0 0 280 22" preserveAspectRatio="none" class="bar-svg">
          <line x1="140" y1="2" x2="140" y2="20" class="zero-line" />
          <rect x="{bar_x:.2f}" y="4" width="{max(bar_w, 0):.2f}" height="14" rx="4" fill="{color}"></rect>
        </svg>
      </div>
      <div class="bar-value">{delta:+.4f}</div>
      <div class="verdict-badge" style="--vcolor:{vcolor}">{html.escape(verdict)}</div>
    </div>"""


def _rho_tick_row(label, rho, threshold):
    """A small secondary strip showing median_delta_rho against the +-0.02
    gate, so a reader can see at a glance why a big MSE win did or didn't
    clear the full 'Better' bar."""
    half = 140.0
    scale = max(threshold * 4, 0.15)
    x = half + max(-scale, min(scale, rho)) / scale * half
    ok = -threshold <= rho  # the "Better" side of the gate (>= -0.02)
    color = "#0ca30c" if ok else "#d03b3b"
    gate_lo = half + (-threshold) / scale * half
    return f"""
    <div class="rho-row" title="{html.escape(label)}: median delta-rho={rho:.4f} (gate: >= -{threshold:g})">
      <div class="bar-label rho-label">{html.escape(label)}</div>
      <div class="bar-track">
        <svg viewBox="0 0 280 16" preserveAspectRatio="none" class="bar-svg">
          <line x1="140" y1="2" x2="140" y2="14" class="zero-line" />
          <line x1="{gate_lo:.2f}" y1="2" x2="{gate_lo:.2f}" y2="14" class="gate-line" />
          <circle cx="{x:.2f}" cy="8" r="4.5" fill="{color}"></circle>
        </svg>
      </div>
      <div class="bar-value">{rho:+.4f}</div>
    </div>"""


def _track_section(data, track_label):
    track = data["track"]
    n = data["n_designs"]
    vs32 = {r["candidate"]: r for r in data["vs_unet32"]}
    vsblur = {r["candidate"]: r for r in data["vs_blur"]}
    cand_order = [c for c in ["unet16", "unet8", "fno", "xgb"] if c in vs32]

    scale32 = max(0.001, max(abs(r["median_delta"]) for r in vs32.values()) * 1.3) if vs32 else 0.02
    rows32 = "".join(
        _bar_row(
            vs32[c]["candidate"],
            f"{c} vs unet32",
            vs32[c]["median_delta"],
            scale32,
            vs32[c]["verdict"],
            f" | wins/losses={vs32[c]['wins']}/{vs32[c]['losses']} adj_p={_fmt(vs32[c]['adjusted_p'], 5)}",
        )
        for c in cand_order
    )

    blur_order = [a for a in data["archs_present"] if a != "blur" and a in vsblur]
    scale_blur = (
        max(0.001, max(abs(r["median_delta"]) for r in vsblur.values()) * 1.3) if vsblur else 0.02
    )
    rows_blur = "".join(
        _bar_row(
            vsblur[a]["candidate"],
            f"{a} vs blur",
            vsblur[a]["median_delta"],
            scale_blur,
            vsblur[a]["verdict"],
            f" | wins/losses={vsblur[a]['wins']}/{vsblur[a]['losses']} adj_p={_fmt(vsblur[a]['adjusted_p'], 5)}",
        )
        for a in blur_order
    )
    rows_rho = "".join(
        _rho_tick_row(a, vsblur[a]["median_delta_rho"], 0.02) for a in blur_order
    )

    def table(rows, baseline):
        head = (
            "<tr><th>Candidate</th><th>median &Delta;</th><th>mean &Delta;</th>"
            "<th>wins/losses</th><th>adj. p</th><th>T (noise floor)</th>"
            "<th>families &minus;/+</th><th>median &Delta;&rho;</th><th>Verdict</th></tr>"
        )
        body = "".join(
            f"<tr><td>{html.escape(r['candidate'])} vs {baseline}</td>"
            f"<td>{_fmt(r['median_delta'], 6)}</td><td>{_fmt(r['mean_delta'], 6)}</td>"
            f"<td>{r['wins']}/{r['losses']}</td><td>{_fmt(r['adjusted_p'], 5)}</td>"
            f"<td>{_fmt(r['T'], 6)}</td>"
            f"<td>{r['families_negative']}/{r['families_positive']} (of {r['n_families']})</td>"
            f"<td>{_fmt(r['median_delta_rho'], 4)}</td>"
            f"<td><span class='verdict-badge' style=\"--vcolor:{VERDICT_COLOR.get(r['verdict'], '#8a8a86')}\">"
            f"{html.escape(r['verdict'])}</span></td></tr>"
            for r in rows
        )
        return f"<table class='data-table'><thead>{head}</thead><tbody>{body}</tbody></table>"

    return f"""
  <section class="track-section">
    <h2>{html.escape(track_label)}</h2>
    <p class="track-meta">{n} held-out designs, 9 design families, 5 seeds per learned arch (Holm-corrected across unet16 / unet8 / fno / xgb).</p>

    <h3>vs. the production U-Net (unet32)</h3>
    <div class="legend"><span class="swatch" style="background:{BLUE}"></span>lower error (better)
      <span class="swatch" style="background:{RED}"></span>higher error (worse)</div>
    <div class="bars">{rows32}</div>
    {table(vs32.values(), "unet32")}

    <h3>vs. a free blur baseline &mdash; the calibration check</h3>
    <p class="track-meta">Every learned arch's median error delta against doing no learning at all.</p>
    <div class="bars">{rows_blur}</div>
    <h4>median &Delta;&rho; (spatial rank correlation) &mdash; the gate that blocks &ldquo;Better&rdquo;</h4>
    <p class="track-meta">An MSE win only counts as "Better" if the arch's spatial ranking of
      hotspots is at least as good as blur's (median &Delta;&rho; &ge; &minus;0.02, dashed line).
      A model can win decisively on error and still fail this gate.</p>
    <div class="bars rho-bars">{rows_rho}</div>
    {table(vsblur.values(), "blur")}
  </section>"""


def build_html(files, track_labels):
    sections = []
    for path, label in zip(files, track_labels):
        with open(path) as f:
            data = json.load(f)
        sections.append(_track_section(data, label))
    body = "\n".join(sections)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Architecture bake-off report</title>
<style>
:root {{
  --bg: #fcfcfb; --surface: #ffffff; --border: #e5e4df;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #85837c;
  --accent: #2a78d6;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #1a1a19; --surface: #242321; --border: #38372f;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8d82;
    --accent: #3987e5;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--text-primary);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}}
.page {{ max-width: 920px; margin: 0 auto; }}
h1 {{ font-size: 1.6rem; margin: 0 0 4px; text-wrap: balance; }}
.subtitle {{ color: var(--text-secondary); margin: 0 0 32px; }}
h2 {{ font-size: 1.25rem; border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-top: 40px; }}
h3 {{ font-size: 1.05rem; margin: 28px 0 4px; }}
h4 {{ font-size: 0.95rem; margin: 20px 0 4px; color: var(--text-secondary); }}
.track-meta {{ color: var(--text-secondary); margin: 4px 0 12px; font-size: 0.92rem; }}
.legend {{ font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 10px; }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 5px 0 12px; vertical-align: -1px; }}
.swatch:first-child {{ margin-left: 0; }}
.bars {{ display: flex; flex-direction: column; gap: 8px; margin-bottom: 18px; }}
.bar-row, .rho-row {{
  display: grid; grid-template-columns: 90px 1fr 70px 110px; align-items: center; gap: 10px;
}}
.rho-row {{ grid-template-columns: 90px 1fr 70px; }}
.bar-label {{ font-size: 0.85rem; font-weight: 600; min-width: 0; }}
.rho-label {{ font-weight: 400; color: var(--text-secondary); }}
.bar-track {{ min-width: 0; }}
.bar-svg {{ display: block; width: 100%; height: 22px; }}
.rho-bars .bar-svg {{ height: 16px; }}
.zero-line {{ stroke: var(--border); stroke-width: 2; }}
.gate-line {{ stroke: var(--text-muted); stroke-width: 1.5; stroke-dasharray: 3 3; }}
.bar-value {{ font-variant-numeric: tabular-nums; font-size: 0.82rem; color: var(--text-secondary); text-align: right; }}
.verdict-badge {{
  font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em;
  color: var(--vcolor); border: 1px solid var(--vcolor); border-radius: 4px;
  padding: 2px 8px; text-align: center; white-space: nowrap;
}}
table.data-table {{ width: 100%; border-collapse: collapse; margin: 10px 0 8px; font-size: 0.82rem; }}
table.data-table {{ display: block; overflow-x: auto; }}
table.data-table thead, table.data-table tbody {{ display: table; width: 100%; table-layout: fixed; }}
table.data-table th, table.data-table td {{
  border-bottom: 1px solid var(--border); padding: 6px 8px; text-align: left; font-variant-numeric: tabular-nums;
}}
table.data-table th {{ color: var(--text-secondary); font-weight: 600; }}
footer {{ margin-top: 40px; color: var(--text-muted); font-size: 0.8rem; }}
</style>
</head>
<body>
<div class="page">
  <h1>Architecture bake-off report</h1>
  <p class="subtitle">Production U-Net (unet32) vs. two smaller U-Nets, a Fourier Neural Operator, and a free blur baseline &mdash; held-out-design-family comparison. Generated from <code>arch_sweep.py --analyze</code>'s own output; no numbers here are recomputed.</p>
{body}
  <footer>Rendered by <code>view_report.py</code> from <code>*_report_data.json</code>. Regenerate after any new sweep run.</footer>
</div>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report_data", nargs="+", help="One or two *_report_data.json files from --analyze")
    ap.add_argument("--out", default=None, help="Output HTML path (default: alongside the first input)")
    args = ap.parse_args()

    labels = []
    for path in args.report_data:
        with open(path) as f:
            track = json.load(f)["track"]
        labels.append({"thermal": "Thermal", "irdrop": "IR-drop"}.get(track, track))

    out_path = args.out or os.path.join(os.path.dirname(args.report_data[0]) or ".", "bakeoff_report.html")
    html_out = build_html(args.report_data, labels)
    with open(out_path, "w") as f:
        f.write(html_out)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
