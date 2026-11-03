"""
Input-normalisation experiment for the thermal and IR-drop networks.

The networks in the symmetric re-evaluation were fed UNNORMALISED inputs
while the best free baseline (`ridge_mm`) sees per-design min-max-scaled
multi-scale features. This retrains `unet32` and `fno` on per-design
normalised inputs to separate "model class" from "input scaling".

Input variants (applied to the network input only; targets untouched):
  V0  identity (raw inputs, as in every earlier run).
  V1  per-design, per-channel min-max to [0, 1]. A constant channel maps to
      all zeros. Same channel count as V0.
  V2  V1's channels plus Gaussian blurs (sigma 1, 2, 4, 8) of the RAW density
      channels (thermal 0-3, IR-drop 0-5), each blurred map min-max scaled
      per design. Thermal 21 channels, IR-drop 30.

Transforms are computed once per design from the un-augmented input and the
flip augmentation is then applied to the transformed input and the target
together (same torch.rand draws as the base datasets). Per-channel min-max
and reflect-boundary blurs commute with flips up to float rounding.

Stages (see DESIGN_RUNS.md "Input-normalisation experiment, pre-registration"):
  train    V1/V2 neural runs, resumable; merges by (track, arch, variant, fold, seed).
  gate     G-V0: V0 through this wrapper must reproduce the stored P0 MSE.
  analyze  N1-N4, summary.md and an HTML report.

Usage (from flow/):
  python3 util/ml/congestion/training/input_norm_eval.py --stage train \\
      --variants V1,V2 --tracks thermal,irdrop --seeds 0-4 --budget-min 25
"""

import argparse
import contextlib
import html
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "models"))
sys.path.insert(0, os.path.join(_HERE, "..", "loop"))

import symmetric_eval as se  # noqa: E402
from arch_sweep import _make_family_folds  # noqa: E402
from laplacian_sweep import TRACKS  # noqa: E402

_CONGESTION = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_OUT = os.path.join(_CONGESTION, "experiments", "input_norm_eval.json")
DEFAULT_SUMMARY = os.path.join(_CONGESTION, "experiments", "input_norm_eval_summary.md")
DEFAULT_HTML = os.path.join(_CONGESTION, "experiments", "input_norm_eval_report.html")
SYMMETRIC = se.DEFAULT_OUT

VARIANTS = ("V0", "V1", "V2")
NEW_VARIANTS = ("V1", "V2")
BLUR_SIGMAS = (1.0, 2.0, 4.0, 8.0)
BLUR_CHANNELS = {"thermal": (0, 1, 2, 3), "irdrop": (0, 1, 2, 3, 4, 5)}
BASE_CHANNELS = {"thermal": 5, "irdrop": 6}
SEEDS = [0, 1, 2, 3, 4]
GATE_FOLDS = ("ibex", "gcd")
GATE_SEEDS = (0, 1)


def in_channels(track: str, variant: str) -> int:
    n = BASE_CHANNELS[track]
    return n + len(BLUR_CHANNELS[track]) * len(BLUR_SIGMAS) if variant == "V2" else n


# ---------------------------------------------------------------- transforms


def minmax_channels(x: np.ndarray) -> np.ndarray:
    """Per-channel min-max of (C,H,W) to [0,1]; a constant channel becomes zeros."""
    x64 = x.astype(np.float64)
    lo = x64.min(axis=(1, 2), keepdims=True)
    hi = x64.max(axis=(1, 2), keepdims=True)
    span = hi - lo
    safe = np.where(span > 0, span, 1.0)
    return np.where(span > 0, (x64 - lo) / safe, 0.0).astype(np.float32)


def blur_channels(x: np.ndarray, channels, sigmas=BLUR_SIGMAS) -> np.ndarray:
    """Per-design min-maxed Gaussian blurs of the raw channels, channel-major."""
    maps = [gaussian_filter(x[c].astype(np.float64), s) for c in channels for s in sigmas]
    return minmax_channels(np.stack(maps))


def transform_input(x: np.ndarray, track: str, variant: str) -> np.ndarray:
    """x (C,H,W) raw input of ONE design -> network input for `variant`.
    Reads nothing but x."""
    if variant == "V0":
        return x
    v1 = minmax_channels(x)
    if variant == "V1":
        return v1
    if variant == "V2":
        return np.concatenate([v1, blur_channels(x, BLUR_CHANNELS[track])], axis=0)
    raise KeyError(f"Unknown variant {variant!r}")


class NormDataset(Dataset):
    """Wraps a NON-augmenting ThermalDataset / IRDropDataset. Exposes `keys`,
    `__len__` and `__getitem__` (same dict, transformed `x`); flips x and the
    target together with the base datasets' torch.rand sequence."""

    def __init__(self, base, track: str, variant: str, augment: bool):
        if getattr(base, "augment", False):
            raise ValueError("NormDataset needs a non-augmenting base dataset")
        self.keys = list(base.keys)
        self.augment = augment
        self.target_key = TRACKS[track]["target_key"]
        self.items = []
        for i in range(len(base)):
            item = dict(base[i])
            item["x"] = torch.from_numpy(np.ascontiguousarray(transform_input(item["x"].numpy(), track, variant)))
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = dict(self.items[idx])
        if self.augment:
            x, t = item["x"], item[self.target_key]
            if torch.rand(1).item() > 0.5:
                x, t = torch.flip(x, dims=[2]), torch.flip(t, dims=[2])
            if torch.rand(1).item() > 0.5:
                x, t = torch.flip(x, dims=[1]), torch.flip(t, dims=[1])
            item["x"], item[self.target_key] = x, t
        return item


@contextlib.contextmanager
def channels_override(track: str, n: int):
    """Point symmetric_eval's own TRACKS binding at a copy with a different
    in_channels, restoring it afterwards. The shared TRACKS dict is untouched."""
    original = se.TRACKS
    se.TRACKS = {**original, track: {**original[track], "in_channels": n}}
    try:
        yield
    finally:
        se.TRACKS = original


# ---------------------------------------------------------------- persistence


def run_key(r) -> tuple:
    return (r["track"], r["arch"], r["variant"], r["fold"], r["seed"])


def merge_runs(existing: list, new: list) -> list:
    """Union by run_key; a later record replaces an earlier one."""
    merged = {run_key(r): r for r in existing}
    merged.update({run_key(r): r for r in new})
    return list(merged.values())


def load_out(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"runs": [], "gate_runs": [], "gates": {}, "timing": []}


def save_out(path, payload):
    se.save_out(path, payload)


def load_datasets(track, variant, data_dir):
    cls = TRACKS[track]["dataset_cls"]
    base = cls(data_dir, augment=False)
    return NormDataset(base, track, variant, True), NormDataset(base, track, variant, False)


def epochs_for(track, override=None):
    return override or se.EPOCHS[track]


def run_one(track, arch, variant, fold, seed, train_ds, eval_ds, epochs, device):
    with channels_override(track, in_channels(track, variant)):
        preds, wall_s = se.train_and_predict(
            arch, track, train_ds, eval_ds, list(fold["held_out"]), seed, epochs, se.BATCH_SIZE, se.LR, device
        )
    rec = se.neural_run_record(track, arch, fold["name"], seed, 1.0, epochs, preds, wall_s)
    rec["variant"] = variant
    return rec


# ---------------------------------------------------------------- stages


def stage_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = load_out(args.out)
    done = {run_key(r) for r in payload["runs"]}
    wanted = set(args.folds.split(",")) if args.folds else None
    t_start = time.time()
    n_new = 0
    stopped = False
    for variant in args.variants:
        for track in args.tracks:
            train_ds, eval_ds = load_datasets(track, variant, args.data_dir)
            folds = _make_family_folds(eval_ds.keys)
            if wanted and not wanted & {f["name"] for f in folds}:
                raise SystemExit(f"--folds {args.folds} matched no fold names")
            epochs = epochs_for(track, args.epochs)
            for seed in args.seeds:
                for arch in args.archs:
                    for fold in folds:
                        if wanted and fold["name"] not in wanted:
                            continue
                        if (track, arch, variant, fold["name"], seed) in done:
                            continue
                        if (time.time() - t_start) / 60.0 > args.budget_min:
                            stopped = True
                            break
                        rec = run_one(track, arch, variant, fold, seed, train_ds, eval_ds, epochs, device)
                        payload["runs"] = merge_runs(payload["runs"], [rec])
                        save_out(args.out, payload)
                        n_new += 1
                        if args.verbose or n_new % 20 == 0:
                            print(
                                f"{variant} {track} {arch} fold={fold['name']} seed={seed} "
                                f"wall_s={rec['wall_s']:.1f} runs_this_chunk={n_new}",
                                flush=True,
                            )
                    if stopped:
                        break
                if stopped:
                    break
            if stopped:
                break
        if stopped:
            break
    payload["timing"].append(
        {
            "stage": "train",
            "variants": list(args.variants),
            "tracks": list(args.tracks),
            "n_new_runs": n_new,
            "wall_s": time.time() - t_start,
            "budget_stopped": stopped,
            "at": time.strftime("%F %T"),
        }
    )
    save_out(args.out, payload)
    print(f"chunk done: {n_new} new runs in {time.time() - t_start:.1f}s (budget_stopped={stopped})")


def stage_gate(args):
    """G-V0: V0 through the wrapper and the in_channels plumbing vs the stored bake-off MSE."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = load_out(args.out)
    done = {run_key(r) for r in payload["gate_runs"]}
    for track in args.tracks:
        train_ds, eval_ds = load_datasets(track, "V0", args.data_dir)
        folds = [f for f in _make_family_folds(eval_ds.keys) if f["name"] in GATE_FOLDS]
        for seed in GATE_SEEDS:
            for arch in args.archs:
                for fold in folds:
                    if (track, arch, "V0", fold["name"], seed) in done:
                        continue
                    rec = run_one(track, arch, "V0", fold, seed, train_ds, eval_ds, epochs_for(track, args.epochs), device)
                    payload["gate_runs"] = merge_runs(payload["gate_runs"], [rec])
                    save_out(args.out, payload)
    payload["gates"] = check_v0_gate(payload["gate_runs"])
    save_out(args.out, payload)
    for k, v in payload["gates"].items():
        print(f"{k}: {'PASS' if v['pass'] else 'FAIL'} matched {v['matched']}/{v['checked']}")
    if not all(v["pass"] for v in payload["gates"].values()):
        raise SystemExit("G-V0 failed")


def check_v0_gate(gate_runs, stored_paths=None):
    stored_paths = stored_paths or se.STORED
    gates = {}
    for track in se.TRACK_NAMES:
        with open(stored_paths[track]) as f:
            stored = json.load(f)["runs"]
        ref = {(r["arch"], r["design"], r["seed"]): r["heldout_mse"] for r in stored if r["arch"] in se.NEURAL}
        n = ok = 0
        for r in gate_runs:
            if r["track"] != track:
                continue
            for d, v in r["designs"].items():
                n += 1
                ok += repr(v["P0"]["heldout_mse_torch"]) == repr(ref[(r["arch"], d, r["seed"])])
        expected = len(se.NEURAL) * len(GATE_FOLDS) * len(GATE_SEEDS)
        n_runs = sum(r["track"] == track for r in gate_runs)
        gates[f"G-V0 {track}"] = {"matched": ok, "checked": n, "pass": n_runs == expected and ok == n, "runs": n_runs, "expected_runs": expected}
    return gates


# ---------------------------------------------------------------- analysis


def series_for(runs, track, arch, variant, pp, seeds=SEEDS):
    """Per-design series for a (track, arch, variant, pp); V0 from the
    symmetric_eval runs, V1/V2 from this experiment's runs. Requires every
    design to have all `seeds`."""
    sel = [r for r in runs if r["track"] == track and r["arch"] == arch and r.get("variant", "V0") == variant]
    s = se.neural_series(sel, track, arch, pp, 1.0, seeds)
    if not s:
        raise SystemExit(f"No runs for {track} {arch} {variant}")
    short = {d: len(e["mse"]) for d, e in s.items() if len(e["mse"]) != len(seeds)}
    if short or len(s) != 30:
        raise SystemExit(f"Incomplete runs for {track} {arch} {variant} {pp}: {len(s)} designs, short={short}")
    return s


def n1_specs(get, track):
    return [
        (f"{a} {v} vs {a} V0 (P0)", get(track, a, v, "P0"), get(track, a, "V0", "P0"))
        for a in se.NEURAL
        for v in NEW_VARIANTS
    ]


def n1_label(verdict):
    return {"Better": "helps", "Worse": "hurts"}.get(verdict, "no demonstrated effect")


def n2_baselines(track):
    return ("ridge_mm|P1", "blur_sel|P1") if track == "thermal" else ("ridge_raw|P0", "mean_map|P0")


def n2_specs(get, free, track):
    pp = "P1" if track == "thermal" else "P0"
    return [
        (f"{a} {v} ({pp}) vs {b}", get(track, a, v, pp), free(track, b))
        for b in n2_baselines(track)
        for a in se.NEURAL
        for v in NEW_VARIANTS
    ]


def n2_label(verdict):
    return "still behind" if verdict == "Worse" else "matches or beats"


def n3_specs(get, track="thermal"):
    return [
        (f"fno vs unet32 {v} ({pp})", get(track, "fno", v, pp), get(track, "unet32", v, pp))
        for v in VARIANTS
        for pp in se.NEURAL_PPS
    ]


def analyze_all(get, free, emit):
    res = {"tables": {}, "N1": {}, "N2": {}, "N3": {}, "N4": {}}
    for track in se.TRACK_NAMES:
        fam = se.families_of(get(track, "fno", "V0", "P0"))
        emit(f"## N1 {track}: does normalisation help the network (P0, Holm m=4)")
        n1 = se.run_comparisons(n1_specs(get, track), fam, len(SEEDS), emit)
        for c in n1:
            c["effect"] = n1_label(c["verdict"])
        res["N1"][track] = n1
        emit(f"## N2 {track}: normalised networks vs free baselines (Holm m=8)")
        n2 = se.run_comparisons(n2_specs(get, free, track), fam, len(SEEDS), emit)
        for c in n2:
            c["effect"] = n2_label(c["verdict"])
        res["N2"][track] = n2
    fam = se.families_of(get("thermal", "fno", "V0", "P0"))
    emit("## N3 thermal: fno vs unet32 within each variant (Holm m=6)")
    res["N3"] = se.run_comparisons(n3_specs(get), fam, len(SEEDS), emit)
    n3_p0 = {c["label"]: c["verdict"] for c in res["N3"] if "(P0)" in c["label"]}
    res["N3_survives"] = {v: n3_p0[f"fno vs unet32 {v} (P0)"] for v in VARIANTS}
    emit(f"N3 P0 verdicts by variant: {res['N3_survives']}")
    emit("## N4 noise: median across designs of the seed sd of per-design MSE")
    for track in se.TRACK_NAMES:
        for arch in se.NEURAL:
            for v in VARIANTS:
                row = {"track": track, "arch": arch, "variant": v}
                for pp in se.NEURAL_PPS:
                    row[f"sd_median_{pp}"] = float(np.median(list(se.per_design_sd(get(track, arch, v, pp)).values())))
                res["N4"].setdefault(track, []).append(row)
                emit(f"{track} {arch} {v}: seed sd median P0 {row['sd_median_P0']:.5f} P1 {row['sd_median_P1']:.5f}")
    for track in se.TRACK_NAMES:
        res["tables"][track] = results_table(get, free, track)
        emit(f"## {track}: median MSE (median Spearman)")
        for name, cells in res["tables"][track]:
            emit(f"- {name}: " + ", ".join(f"{pp} {a:.5f} ({b:.3f})" for pp, (a, b) in cells.items()))
    return res


def results_table(get, free, track):
    rows = []
    for arch in se.NEURAL:
        for v in VARIANTS:
            rows.append((f"{arch} {v}", {pp: (se.median_of(get(track, arch, v, pp), "mse"), se.median_of(get(track, arch, v, pp), "rho")) for pp in se.NEURAL_PPS}))
    for name in ("const_mean", "mean_map", "blur_sel", "ridge_raw", "ridge_mm"):
        cells = {}
        for pp in se.NEURAL_PPS:
            key = f"{name}|{pp}"
            try:
                s = free(track, key)
            except KeyError:
                continue
            cells[pp] = (se.median_of(s, "mse"), se.median_of(s, "rho"))
        rows.append((name + " (free)", cells))
    return rows


# ---------------------------------------------------------------- report


def _short(c):
    return (
        f"{c['label']}: {c['verdict']}, median change {c['median_delta']:+.4f}, lower for the first model on "
        f"{c['wins']}/{c['n']} designs and in {c['families_negative']}/{c['n_families']} families, adjusted p {c['adjusted_p']:.3g}"
    )


def answers(res):
    name = {"thermal": "Thermal", "irdrop": "IR-drop"}
    out = []
    parts = []
    for track in se.TRACK_NAMES:
        parts.append(
            f"{name[track]}: "
            + "; ".join(f"{c['label'].replace(' (P0)', '')} {c['effect']} (median change {c['median_delta']:+.4f}, lower in {c['families_negative']}/{c['n_families']} families)" for c in res["N1"][track])
        )
    out.append("Does normalising the inputs help the networks? " + ". ".join(parts) + ".")
    parts = []
    for track in se.TRACK_NAMES:
        parts.append(
            f"{name[track]}: "
            + "; ".join(
                f"{c['label']}: {c['verdict']} (network median change {c['median_delta']:+.4f}, "
                f"network lower on {c['wins']}/{c['n']} designs and in {c['families_negative']}/{c['n_families']} families, "
                f"adjusted p {c['adjusted_p']:.3g})"
                for c in res["N2"][track]
            )
        )
    out.append(
        "Do the normalised networks reach the free baselines? Not shown to differ from them in any comparison, "
        "but that reflects limited statistical power (30 designs, 9 families), not a demonstrated match: "
        "a positive median change means the baseline is better. " + ". ".join(parts) + "."
    )
    out.append(
        "Does the FNO-vs-U-Net gap depend on normalisation (thermal)? "
        + "; ".join(f"{c['label']}: {c['verdict']} (median change {c['median_delta']:+.4f}, lower for fno in {c['families_negative']}/{c['n_families']} families)" for c in res["N3"])
        + ". A verdict that appears under only one of the two post-processing settings (unscaled vs rescaled) should be read as fragile."
    )
    return out


def _fmt(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def build_html(res):
    esc = html.escape
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Input normalisation</title><style>",
        ":root{color-scheme:light dark;--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ccc}",
        "@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--mut:#9a9a9a;--line:#444}}",
        "body{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem}",
        "table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid var(--line);padding:.25rem .6rem;text-align:right}",
        "th:first-child,td:first-child{text-align:left}.mut{color:var(--mut)}li{margin:.5rem 0}",
        "</style></head><body><h1>What does scaling the inputs change for the networks?</h1>",
        "<h2>Answers</h2><ol>",
    ]
    parts += [f"<li>{esc(a)}</li>" for a in answers(res)]
    parts.append("</ol>")
    for track in se.TRACK_NAMES:
        parts.append(f"<h2>{track}: median over 30 designs</h2>")
        parts.append("<p class='mut'>Each cell is median error (MSE, lower is better) with median rank agreement (Spearman, higher is better) in brackets.</p>")
        parts.append("<table><tr><th>predictor</th>" + "".join(f"<th>{pp}</th>" for pp in se.NEURAL_PPS) + "</tr>")
        for name, cells in res["tables"][track]:
            tds = []
            for pp in se.NEURAL_PPS:
                c = cells.get(pp)
                tds.append(f"<td>{_fmt(c[0])} ({_fmt(c[1])})</td>" if c else "<td class='mut'>-</td>")
            parts.append(f"<tr><td>{esc(name)}</td>{''.join(tds)}</tr>")
        parts.append("</table>")
    parts += [
        "<h2>Glossary</h2><ul>",
        "<li><b>V0</b>: the network is fed the raw input maps, as in every earlier run.</li>",
        "<li><b>V1</b>: each input map of a design is rescaled so its own lowest value is 0 and highest is 1 (a flat map becomes zeros).</li>",
        "<li><b>V2</b>: V1 plus blurred copies (four blur widths) of the raw density maps, each rescaled the same way. This hands the network the kind of multi-scale features the ridge baseline uses.</li>",
        "<li><b>P0</b>: the predicted map exactly as produced. <b>P1</b>: the prediction stretched to its own 0-1 range (uses nothing but the prediction).</li>",
        "<li><b>Median MSE</b>: mean squared error per design, then the middle value across the 30 designs. Lower is better.</li>",
        "<li><b>Spearman</b>: how well the predicted pixel ordering matches the truth; 1 is perfect, 0 is none.</li>",
        "<li><b>Better / Worse / Indistinguishable</b>: the project's seven-check rule (error difference above seed noise, at least 20 of 30 designs and 7 of 9 families on the same side, adjusted p below 0.05, rank agreement not moving the other way). Anything else is Indistinguishable.</li>",
        "<li><b>Design families</b>: the 30 designs fall in 9 families held out together, so they are not independent.</li>",
        "</ul>",
        "<h2>Detail</h2><ul>",
    ]
    for key in ("N1", "N2"):
        for track in se.TRACK_NAMES:
            parts += [f"<li>{key} {track}: {esc(_short(c))}</li>" for c in res[key][track]]
    parts += [f"<li>N3: {esc(_short(c))}</li>" for c in res["N3"]]
    parts.append("</ul></body></html>")
    return "".join(parts)


def stage_analyze(args):
    payload = load_out(args.out)
    with open(args.symmetric) as f:
        sym = json.load(f)
    runs = sym["runs"] + payload["runs"]
    lines: list[str] = []

    def emit(s=""):
        lines.append(s)
        print(s)

    def get(track, arch, variant, pp):
        return series_for(runs, track, arch, variant, pp)

    def free(track, name):
        return se.free_series(sym["baselines"], track, name)

    emit("## Parity gate G-V0")
    for k, v in payload["gates"].items():
        emit(f"{k}: {'PASS' if v['pass'] else 'FAIL'} matched {v['matched']}/{v['checked']}")
    res = analyze_all(get, free, emit)
    res["gates"] = payload["gates"]
    payload["analysis"] = res
    save_out(args.out, payload)
    se.write_summary(args.summary, lines)
    with open(args.html, "w") as f:
        f.write(build_html(res))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["train", "gate", "analyze"], required=True)
    ap.add_argument("--variants", default="V1,V2")
    ap.add_argument("--tracks", default=",".join(se.TRACK_NAMES))
    ap.add_argument("--archs", default="unet32,fno")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--folds", default=None, help="Comma-separated fold names")
    ap.add_argument("--epochs", type=int, default=None, help="Override; default is the stored bake-off epochs")
    ap.add_argument("--budget-min", type=float, default=float("inf"))
    ap.add_argument("--verbose", action="store_true", help="print one line per training run (default: one per 20)")
    ap.add_argument("--data-dir", default=se.DEFAULT_DATA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--symmetric", default=SYMMETRIC)
    ap.add_argument("--summary", default=DEFAULT_SUMMARY)
    ap.add_argument("--html", default=DEFAULT_HTML)
    args = ap.parse_args()
    args.seeds = se.parse_seeds(args.seeds)
    args.archs = args.archs.split(",")
    args.tracks = args.tracks.split(",")
    args.variants = args.variants.split(",")
    for a in args.archs:
        if a not in se.NEURAL:
            raise SystemExit(f"--archs supports {se.NEURAL}, got {a!r}")
    for v in args.variants:
        if v not in NEW_VARIANTS:
            raise SystemExit(f"--variants supports {NEW_VARIANTS}, got {v!r}")
    {"train": stage_train, "gate": stage_gate, "analyze": stage_analyze}[args.stage](args)


if __name__ == "__main__":
    main()
