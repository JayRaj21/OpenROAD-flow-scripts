"""
Generate a self-contained HTML report comparing predicted vs ground-truth
IR-drop maps for every design in the data directory.

Usage (from flow/):
  python3 ml/congestion/inference/visualize_irdrop.py \\
      --data-dir   ml/congestion/data \\
      --checkpoint ml/congestion/checkpoints/irdrop_best.pt \\
      --out        irdrop_report.html

Open the output file in any browser — no server needed.
"""

import argparse
import base64
import glob
import io
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from unet import CongestionUNet


def _parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default="ml/congestion/data")
    ap.add_argument("--checkpoint", default="ml/congestion/checkpoints/irdrop_best.pt")
    ap.add_argument("--out", default="irdrop_report.html")
    ap.add_argument("--base-features", type=int, default=32)
    return ap.parse_args()


def _arr_to_b64(arr, cmap="hot", vmin=0, vmax=1, colorbar_label=None):
    """Render array to a base64 PNG, optionally with a labelled colorbar."""
    if colorbar_label is not None:
        fig, (ax, cax) = plt.subplots(
            1,
            2,
            figsize=(3.0, 2.5),
            gridspec_kw={"width_ratios": [10, 0.5], "wspace": 0.05},
            layout="constrained",
        )
    else:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        cax = None

    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.axis("off")

    if cax is not None:
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(colorbar_label, color="#9090b0", fontsize=7)
        cb.ax.yaxis.set_tick_params(color="#9090b0", labelsize=6, labelcolor="#9090b0")
        cax.set_facecolor("#0d0d0d")
        cb.outline.set_edgecolor("#2a2a38")
    else:
        fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _platform(label):
    if label.startswith("asap7"):
        return "asap7"
    if label.startswith("nangate45"):
        return "nangate45"
    return "sky130hd"


def collect(data_dir, model, device):
    irdrop_files = glob.glob(os.path.join(data_dir, "*_irdrop_labels.npz"))
    labels = sorted(
        os.path.basename(p).replace("_irdrop_labels.npz", "") for p in irdrop_files
    )

    rows = []
    for label in labels:
        feat_path = os.path.join(data_dir, f"{label}_features.npz")
        irdrop_path = os.path.join(data_dir, f"{label}_irdrop_labels.npz")
        if not os.path.exists(feat_path):
            print(f"  [skip] {label} — no features file")
            continue

        feat = np.load(feat_path)
        irdrop_npz = np.load(irdrop_path)
        x = np.stack(
            [
                feat["cell_density"],
                feat["macro_density"],
                feat["pin_density"],
                feat["fanout_density"],
                irdrop_npz["stripe_density"],
                irdrop_npz["via_density"],
            ]
        ).astype(np.float32)
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)

        d = irdrop_npz["irdrop_map"].astype(np.float32)
        d_lo, d_hi = float(d.min()), float(d.max())
        denom = (d_hi - d_lo) if d_hi > d_lo else 1.0
        d_norm = (d - d_lo) / denom

        with torch.no_grad():
            pred = model(x_t).heatmap[0, 0].cpu().numpy()

        mae = float(np.abs(pred - d_norm).mean())
        with np.errstate(invalid="ignore"):
            corr = float(np.corrcoef(pred.ravel(), d_norm.ravel())[0, 1])

        # Predicted drop in volts = rescale pred [0,1] back to V using GT range.
        pred_v = pred * (d_hi - d_lo) + d_lo

        rows.append(
            {
                "label": label,
                "corr": round(corr, 3),
                "mae": round(mae, 4),
                "d_min_mv": round(d_lo * 1e3, 2),
                "d_max_mv": round(d_hi * 1e3, 2),
                "delta_mv": round((d_hi - d_lo) * 1e3, 2),
                "stripe": _arr_to_b64(
                    irdrop_npz["stripe_density"],
                    "Blues",
                    0,
                    float(irdrop_npz["stripe_density"].max()) or 1,
                ),
                # Ground truth with mV colorbar so absolute drop is readable.
                "gt": _arr_to_b64(
                    d * 1e3, "hot", d_lo * 1e3, d_hi * 1e3, colorbar_label="mV"
                ),
                # Predicted in mV (rescaled from normalised output via GT range).
                "pred": _arr_to_b64(
                    pred_v * 1e3, "hot", d_lo * 1e3, d_hi * 1e3, colorbar_label="mV"
                ),
            }
        )
        print(
            f"  {label:45s}  corr={corr:+.3f}  mae={mae:.3f}  "
            f"{d_lo*1e3:.1f}–{d_hi*1e3:.1f}mV"
        )

    return rows


HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IR-Drop Prediction Report</title>
<style>
:root {{
  --bg:      #0e0e12;
  --surface: #16161d;
  --border:  #2a2a38;
  --text:    #d4d4e8;
  --muted:   #6b6b88;
  --accent:  #ff6b35;
  --good:    #4ade80;
  --warn:    #facc15;
  --bad:     #f87171;
  --mono: 'Courier New', Courier, monospace;
  --sans: system-ui, -apple-system, sans-serif;
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:13px;line-height:1.5;min-height:100vh}}
header{{padding:28px 32px 20px;border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}}
header h1{{font-size:18px;font-weight:600;letter-spacing:.04em}}
header .meta{{font-size:12px;color:var(--muted);font-family:var(--mono)}}
.controls{{padding:14px 32px;border-bottom:1px solid var(--border);display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
.controls label{{color:var(--muted);font-size:11px;letter-spacing:.05em;text-transform:uppercase}}
select,input[type=range]{{background:var(--surface);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:4px 8px;font-size:12px;font-family:var(--mono);cursor:pointer}}
select:focus,input:focus{{outline:1px solid var(--accent)}}
.sep{{width:1px;height:20px;background:var(--border)}}
#count{{font-size:11px;color:var(--muted);font-family:var(--mono);margin-left:auto}}
#grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:1px;background:var(--border)}}
.card{{background:var(--surface);padding:16px 18px 14px;display:flex;flex-direction:column;gap:10px}}
.card-header{{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}}
.card-label{{font-family:var(--mono);font-size:12px;font-weight:600;flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.card-label .plat{{color:var(--muted);font-weight:400}}
.card-stats{{display:flex;gap:14px;flex-shrink:0}}
.stat{{display:flex;flex-direction:column;align-items:flex-end}}
.stat-val{{font-family:var(--mono);font-size:13px;font-variant-numeric:tabular-nums;font-weight:600}}
.stat-lbl{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
.corr-bar{{height:2px;border-radius:1px;background:var(--border);overflow:hidden}}
.corr-fill{{height:100%;border-radius:1px}}
.panels{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px}}
.panel{{display:flex;flex-direction:column;gap:4px}}
.panel-lbl{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}}
.panel img{{width:100%;aspect-ratio:1;display:block;border-radius:2px;border:1px solid var(--border);image-rendering:pixelated}}
</style>
</head>
<body>
<header>
  <h1>IR-Drop Prediction Report</h1>
  <span class="meta">{checkpoint} &nbsp;·&nbsp; {n} designs</span>
</header>
<div class="controls">
  <label>Platform</label>
  <select id="fPlat">
    <option value="">All</option>
    <option value="asap7">asap7</option>
    <option value="nangate45">nangate45</option>
    <option value="sky130hd">sky130hd</option>
  </select>
  <div class="sep"></div>
  <label>Min corr</label>
  <input type="range" id="fCorr" min="-1" max="1" step="0.05" value="-1">
  <span id="fCorrVal" style="font-family:var(--mono);font-size:12px;width:40px">—</span>
  <div class="sep"></div>
  <label>Sort</label>
  <select id="fSort">
    <option value="label">Name</option>
    <option value="corr-desc">Corr ↓</option>
    <option value="corr-asc">Corr ↑</option>
    <option value="delta-desc">ΔV ↓</option>
  </select>
  <span id="count"></span>
</div>
<div id="grid"></div>
<script>
const DATA={data_js};
function corrColor(c){{return c>=.85?'var(--good)':c>=.6?'var(--warn)':'var(--bad)'}}
function platform(l){{return l.startsWith('asap7')?'asap7':l.startsWith('nangate45')?'nangate45':'sky130hd'}}
function fmtLabel(l){{
  const p=platform(l);
  const rest=l.slice(p.length+1);
  return`<span class="plat">${{p}} / </span>${{rest}}`;
}}
function render(){{
  const p=document.getElementById('fPlat').value;
  const mc=parseFloat(document.getElementById('fCorr').value);
  const s=document.getElementById('fSort').value;
  document.getElementById('fCorrVal').textContent=mc.toFixed(2);
  let rows=DATA.filter(r=>(!p||platform(r.label)===p)&&r.corr>=mc);
  if(s==='corr-desc') rows.sort((a,b)=>b.corr-a.corr);
  else if(s==='corr-asc') rows.sort((a,b)=>a.corr-b.corr);
  else if(s==='delta-desc') rows.sort((a,b)=>b.delta_mv-a.delta_mv);
  else rows.sort((a,b)=>a.label.localeCompare(b.label));
  document.getElementById('count').textContent=rows.length+' / '+DATA.length+' designs';
  document.getElementById('grid').innerHTML=rows.map(r=>{{
    const fc=corrColor(r.corr);
    return`<div class="card">
      <div class="card-header">
        <div class="card-label">${{fmtLabel(r.label)}}</div>
        <div class="card-stats">
          <div class="stat"><span class="stat-val" style="color:${{fc}}">${{r.corr.toFixed(3)}}</span><span class="stat-lbl">corr</span></div>
          <div class="stat"><span class="stat-val">${{r.mae.toFixed(3)}}</span><span class="stat-lbl">mae</span></div>
          <div class="stat"><span class="stat-val">${{r.d_min_mv.toFixed(1)}}–${{r.d_max_mv.toFixed(1)}}</span><span class="stat-lbl">mV drop</span></div>
          <div class="stat"><span class="stat-val">${{r.delta_mv.toFixed(1)}}mV</span><span class="stat-lbl">ΔV</span></div>
        </div>
      </div>
      <div class="corr-bar"><div class="corr-fill" style="width:${{Math.max(0,r.corr)*100}}%;background:${{fc}}"></div></div>
      <div class="panels">
        <div class="panel"><span class="panel-lbl">Stripe density</span><img src="data:image/png;base64,${{r.stripe}}" alt="stripe density"></div>
        <div class="panel"><span class="panel-lbl">Ground truth</span><img src="data:image/png;base64,${{r.gt}}" alt="ground truth"></div>
        <div class="panel"><span class="panel-lbl">Predicted</span><img src="data:image/png;base64,${{r.pred}}" alt="predicted"></div>
      </div>
    </div>`;
  }}).join('');
}}
['fPlat','fCorr','fSort'].forEach(id=>document.getElementById(id).addEventListener('change',render));
document.getElementById('fCorr').addEventListener('input',render);
render();
</script>
</body>
</html>
"""


def main():
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = CongestionUNet(
        in_channels=6, base_features=args.base_features, num_heatmap_layers=1
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    rows = collect(args.data_dir, model, device)
    if not rows:
        print("No paired designs found — check --data-dir.")
        return

    data_js = json.dumps(rows)
    html = HTML_TEMPLATE.format(
        checkpoint=os.path.basename(args.checkpoint),
        n=len(rows),
        data_js=data_js,
    )

    with open(args.out, "w") as f:
        f.write(html)
    print(f"\nReport → {args.out}  ({os.path.getsize(args.out)//1024} KB)")
    print(f"Open with:  xdg-open {args.out}")


if __name__ == "__main__":
    main()
