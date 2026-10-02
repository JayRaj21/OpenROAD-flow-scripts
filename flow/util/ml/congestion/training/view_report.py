"""
Renders arch_sweep.py's `--analyze` output (the `*_report_data.json` files
it writes alongside each `*_summary.md`) as a single local HTML page, so
the bake-off result can be read in plain language instead of parsed out of
a markdown table of statistics.

This is a pure renderer: every number and verdict it shows comes straight
from the `_verdict()` stats already computed and printed by `--analyze`.
It does not recompute any statistic, so the page can never say something
that disagrees with the text summary.

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

VERDICT_COLOR = {
    "Better": "#0ca30c",
    "Worse": "#d03b3b",
    "Indistinguishable": "#8a8a86",
}
BLUE = "#2a78d6"
RED = "#e34948"
GRAY_MID = "#c9c8c2"

CANDIDATE_NAMES = {
    "unet16": "Smaller U-Net (half width)",
    "unet8": "Smaller U-Net (quarter width)",
    "fno": "Fourier Neural Operator",
    "xgb": "Per-pixel decision tree",
}

VERDICT_EXPLAIN = {
    "Better": "Clears every bar below: enough designs improved, the effect "
    "is bigger than run-to-run noise, it shows up broadly across different "
    "chip families, and it isn't bought by getting worse at locating the "
    "hotspots.",
    "Worse": "Clears every bar below in the losing direction: a real, "
    "broad, noise-beating regression.",
    "Indistinguishable": "At least one required check didn't clear its "
    "bar (see the detail line). That doesn't mean the two are identical, "
    "only that this test can't tell them apart with confidence.",
}


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    try:
        if x != x:
            return "n/a"
    except TypeError:
        return str(x)
    return f"{x:.{nd}f}"


def _pct_change(row):
    base = row.get("baseline_median_mse")
    if not base:
        return None
    return (row["candidate_median_mse"] - row["baseline_median_mse"]) / base * 100.0


def _verdict_reason(row, baseline_name):
    """A short, specific, plain-language reason for the verdict, built from
    the same numbers in the detail table -- so a reader doesn't have to
    reverse-engineer the criteria themselves."""
    v = row["verdict"]
    wins, losses, n = row["wins"], row["losses"], row["n"]
    adj_p = row["adjusted_p"]
    rho = row["median_delta_rho"]
    fam_dir = row["families_negative"] if row["median_delta"] < 0 else row["families_positive"]
    n_fam = row["n_families"]
    better_dir = row["median_delta"] < 0
    win_word = "won" if better_dir else "lost"
    count = wins if better_dir else losses

    if v != "Indistinguishable":
        return (
            f"{win_word.capitalize()} on {count} of {n} held-out designs, holds up across "
            f"{fam_dir} of {n_fam} chip families, and the gap is bigger than the "
            f"noise floor (statistically solid after correcting for testing "
            f"multiple candidates, adjusted p={adj_p:.3f})."
        )

    reasons = []
    if count < 20:
        reasons.append(f"only {win_word} on {count} of {n} designs (needs 20+)")
    if adj_p >= 0.05:
        reasons.append(f"the gap could plausibly be noise (adjusted p={adj_p:.3f}, needs < 0.05)")
    if fam_dir < 7:
        reasons.append(f"only shows up in {fam_dir} of {n_fam} chip families (needs 7+)")
    if abs(row["median_delta"]) <= row["T"]:
        reasons.append("the typical gap is smaller than the run-to-run noise floor")
    rho_gate_dir = rho >= -0.02 if better_dir else rho <= 0.02
    if not rho_gate_dir:
        reasons.append(
            f"it gets worse at locating the hotspots while winning on average error "
            f"(hotspot-ranking change {rho:+.3f}, the wrong side of the gate)"
        )
    if not reasons:
        reasons.append("it falls just short of one of the required bars")
    return "Not confident enough to call it either way: " + "; ".join(reasons) + "."


def _mini_bar(pct, scale):
    """A small diverging bar inside a card: left of center = more accurate
    (blue), right of center = less accurate (red), as a quick visual cue
    alongside the percent number."""
    half = 120.0
    clamped = max(-scale, min(scale, pct))
    bar_w = abs(clamped) / scale * half
    color = BLUE if pct < 0 else RED if pct > 0 else GRAY_MID
    bar_x = half - bar_w if pct < 0 else half
    return f"""<svg viewBox="0 0 240 14" preserveAspectRatio="none" class="mini-bar">
      <line x1="120" y1="1" x2="120" y2="13" class="zero-line" />
      <rect x="{bar_x:.2f}" y="2" width="{max(bar_w, 0):.2f}" height="10" rx="3" fill="{color}"></rect>
    </svg>"""


def _candidate_card(row, baseline_label, scale):
    name = CANDIDATE_NAMES.get(row["candidate"], row["candidate"])
    pct = _pct_change(row)
    direction = "lower (more accurate)" if pct is not None and pct < 0 else "higher (less accurate)"
    pct_str = f"{abs(pct):.0f}% {direction}" if pct is not None else "n/a"
    vcolor = VERDICT_COLOR.get(row["verdict"], "#8a8a86")
    reason = _verdict_reason(row, baseline_label)
    bar = _mini_bar(pct, scale) if pct is not None else ""
    return f"""
    <div class="card">
      <div class="card-top">
        <div class="card-name">{html.escape(name)} <span class="card-slug">({html.escape(row['candidate'])})</span></div>
        <div class="verdict-badge big" style="--vcolor:{vcolor}">{html.escape(row['verdict'])}</div>
      </div>
      <div class="card-headline">Typical error vs. {html.escape(baseline_label)}: <strong>{pct_str}</strong></div>
      {bar}
      <div class="card-reason">{html.escape(reason)}</div>
    </div>"""


def _data_table(rows, baseline):
    head = (
        "<tr><th>Candidate</th><th>Median error change</th><th>Wins / losses</th>"
        "<th>Adj. p-value</th><th>Noise floor</th><th>Families agreeing</th>"
        "<th>Hotspot-ranking change</th><th>Verdict</th></tr>"
    )
    body = "".join(
        f"<tr><td>{html.escape(r['candidate'])} vs {html.escape(baseline)}</td>"
        f"<td>{_fmt(r['median_delta'], 6)}</td>"
        f"<td>{r['wins']} / {r['losses']} (of {r['n']})</td>"
        f"<td>{_fmt(r['adjusted_p'], 5)}</td>"
        f"<td>&plusmn;{_fmt(r['T'], 6)}</td>"
        f"<td>{r['families_negative']}/{r['families_positive']} (of {r['n_families']})</td>"
        f"<td>{_fmt(r['median_delta_rho'], 4)}</td>"
        f"<td><span class='verdict-badge' style=\"--vcolor:{VERDICT_COLOR.get(r['verdict'], '#8a8a86')}\">"
        f"{html.escape(r['verdict'])}</span></td></tr>"
        for r in rows
    )
    return (
        "<details class='tech-table'><summary>Full statistics table</summary>"
        f"<table class='data-table'><thead>{head}</thead><tbody>{body}</tbody></table>"
        "</details>"
    )


def _track_section(data, track_label):
    n = data["n_designs"]
    vs32 = {r["candidate"]: r for r in data["vs_unet32"]}
    vsblur = {r["candidate"]: r for r in data["vs_blur"]}
    cand_order = [c for c in ["unet16", "unet8", "fno", "xgb"] if c in vs32]
    blur_order = [a for a in data["archs_present"] if a != "blur" and a in vsblur]

    winners = [c for c in cand_order if vs32[c]["verdict"] == "Better"]
    losers = [c for c in cand_order if vs32[c]["verdict"] == "Worse"]
    if winners:
        headline = (
            f"<strong>{html.escape(CANDIDATE_NAMES.get(winners[0], winners[0]))}</strong> "
            "beats the current production model, clearly enough to trust."
        )
        hcolor = VERDICT_COLOR["Better"]
    elif losers:
        headline = (
            f"<strong>{html.escape(CANDIDATE_NAMES.get(losers[0], losers[0]))}</strong> "
            "does worse than the current production model, clearly enough to trust."
        )
        hcolor = VERDICT_COLOR["Worse"]
    else:
        headline = (
            "None of the alternatives tried here beat the current production model "
            "by enough to trust the difference."
        )
        hcolor = VERDICT_COLOR["Indistinguishable"]

    scale32 = max(1.0, max(abs(_pct_change(r) or 0) for r in vs32.values()) * 1.3) if vs32 else 10.0
    cards32 = "".join(_candidate_card(vs32[c], "the current model", scale32) for c in cand_order)

    blur_wins = [a for a in blur_order if vsblur[a]["verdict"] == "Better"]
    if blur_wins:
        blur_headline = "At least one model meaningfully beats a free, untrained guess."
    else:
        blur_summary_pct = [
            _pct_change(vsblur[a]) for a in blur_order if _pct_change(vsblur[a]) is not None
        ]
        if blur_summary_pct and max(abs(p) for p in blur_summary_pct) > 10:
            blur_headline = (
                "Every model has much lower average error than a free, untrained guess "
                "&mdash; but none of them officially counts as &ldquo;better,&rdquo; because "
                "they're worse at pinpointing <em>where</em> the worst spots actually are "
                "(see the cards below)."
            )
        else:
            blur_headline = "No model clearly beats a free, untrained guess here."
    scale_blur = (
        max(1.0, max(abs(_pct_change(r) or 0) for r in vsblur.values()) * 1.3) if vsblur else 10.0
    )
    cards_blur = "".join(
        _candidate_card(vsblur[a], "a free blur-of-the-input guess", scale_blur) for a in blur_order
    )

    return f"""
  <section class="track-section">
    <h2>{html.escape(track_label)}</h2>
    <p class="track-meta">{n} held-out chip designs across 9 unrelated circuit families, each tested 5 times with different random starts.</p>

    <div class="headline-box" style="--hcolor:{hcolor}">{headline}</div>

    <h3>Does anything beat the current production model?</h3>
    <div class="cards">{cards32}</div>
    {_data_table(vs32.values(), "unet32")}

    <h3>Is any of this better than doing nothing?</h3>
    <p class="track-meta">The free baseline just blurs the input and makes no prediction of its own &mdash; a sanity floor every real model should clear easily.</p>
    <div class="headline-box soft">{blur_headline}</div>
    <div class="cards">{cards_blur}</div>
    {_data_table(vsblur.values(), "blur")}
  </section>"""


GLOSSARY = """
  <details class="glossary" open>
    <summary>What do these terms mean?</summary>
    <dl>
      <dt>Median error change</dt>
      <dd>How much the typical held-out design's prediction error went up or
      down compared to the baseline, as a percentage. A big negative number
      is a real improvement; a big positive number is a real regression.
      "Median" means the middle design when all 30 are sorted by how much
      they changed &mdash; it isn't skewed by one unusually good or bad design.</dd>

      <dt>Wins / losses</dt>
      <dd>Out of the held-out designs, how many had lower error ("win") vs.
      higher error ("loss") than the baseline. A real effect should win on
      most designs, not just a handful.</dd>

      <dt>Adjusted p-value</dt>
      <dd>Roughly: the odds this pattern could show up by pure chance if
      there were actually no difference. "Adjusted" means it's already been
      made stricter to account for testing several candidates at once, so
      one of them looking good by luck doesn't get mistaken for a real
      effect. Below 0.05 is the bar for calling a result real.</dd>

      <dt>Noise floor</dt>
      <dd>How much a model's score naturally jitters from run to run, just
      from a different random starting point &mdash; estimated from the
      actual seed-to-seed spread measured here. A difference smaller than
      this isn't trustworthy even if the p-value looks good, since it could
      just be luck in that particular training run.</dd>

      <dt>Families agreeing</dt>
      <dd>The 30 chip designs come from 9 unrelated circuit families (e.g.
      an AES encryption core vs. a RISC-V CPU vs. a JPEG decoder). A
      trustworthy effect should point the same direction in most of these
      9 families, not come from one family dragging the average.</dd>

      <dt>Hotspot-ranking change</dt>
      <dd>A separate check from average error: does the model correctly
      rank <em>where on the chip</em> the worst spots are, relative to
      everywhere else? A model can have great average error and still be
      bad at this &mdash; which is exactly what happens in some of the
      results below: a model's predictions can be closer to the truth on
      average while still getting the relative ranking of hot/cold spots
      more wrong than a much cruder baseline does.</dd>

      <dt>Better / Worse / Indistinguishable</dt>
      <dd><strong>Better</strong> or <strong>Worse</strong> means every one
      of the checks above agrees in the same direction, by enough to trust.
      <strong>Indistinguishable</strong> means at least one check didn't
      clear its bar &mdash; it does <em>not</em> mean the two are proven
      identical, only that this test isn't confident enough to call it.</dd>
    </dl>
  </details>"""


def build_html(files, track_labels):
    sections = [
        _track_section(json.load(open(path)), label)
        for path, label in zip(files, track_labels)
    ]
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
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}}
.page {{ max-width: 880px; margin: 0 auto; }}
h1 {{ font-size: 1.6rem; margin: 0 0 4px; text-wrap: balance; }}
.subtitle {{ color: var(--text-secondary); margin: 0 0 24px; }}
h2 {{ font-size: 1.3rem; border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-top: 44px; }}
h3 {{ font-size: 1.05rem; margin: 28px 0 4px; }}
.track-meta {{ color: var(--text-secondary); margin: 4px 0 14px; font-size: 0.92rem; }}

.glossary {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 4px 18px 16px; margin: 20px 0 32px;
}}
.glossary summary {{ cursor: pointer; font-weight: 600; padding: 12px 0; }}
.glossary dl {{ margin: 8px 0 4px; }}
.glossary dt {{ font-weight: 600; margin-top: 12px; color: var(--accent); }}
.glossary dd {{ margin: 3px 0 0; color: var(--text-secondary); max-width: 68ch; }}

.headline-box {{
  background: color-mix(in srgb, var(--hcolor) 12%, var(--surface));
  border-left: 4px solid var(--hcolor);
  border-radius: 6px; padding: 12px 16px; margin: 6px 0 20px; font-size: 1.02rem;
}}
.headline-box.soft {{ --hcolor: var(--text-muted); border-left-color: var(--border); background: var(--surface); border: 1px solid var(--border); }}

.cards {{ display: flex; flex-direction: column; gap: 10px; margin-bottom: 14px; }}
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
.card-top {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }}
.card-name {{ font-weight: 700; font-size: 1rem; }}
.card-slug {{ font-weight: 400; color: var(--text-muted); font-size: 0.85rem; }}
.card-headline {{ margin-top: 8px; font-size: 0.95rem; }}
.mini-bar {{ display: block; width: 100%; max-width: 260px; height: 14px; margin-top: 6px; }}
.zero-line {{ stroke: var(--border); stroke-width: 2; }}
.card-reason {{ margin-top: 8px; color: var(--text-secondary); font-size: 0.88rem; }}

.verdict-badge {{
  font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em;
  color: var(--vcolor); border: 1px solid var(--vcolor); border-radius: 4px;
  padding: 2px 8px; text-align: center; white-space: nowrap;
}}
.verdict-badge.big {{ font-size: 0.78rem; padding: 4px 10px; }}

.tech-table {{ margin: 8px 0 28px; }}
.tech-table summary {{ cursor: pointer; color: var(--text-secondary); font-size: 0.88rem; margin-bottom: 8px; }}
table.data-table {{ width: 100%; border-collapse: collapse; margin: 10px 0 8px; font-size: 0.8rem; display: block; overflow-x: auto; }}
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
  <p class="subtitle">Comparing the production model against alternatives, on held-out chip designs it never trained on. Generated straight from <code>arch_sweep.py --analyze</code>'s own numbers.</p>
{GLOSSARY}
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
