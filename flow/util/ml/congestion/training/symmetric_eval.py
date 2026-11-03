"""
Symmetric re-evaluation of predictors on the thermal and IR-drop tracks.

Every predictor (free baselines, a ridge filter bank, the trained `unet32`
and `fno`) is scored under the same post-processing options, on the same 9
leave-one-design-family-out folds, so that "learned model vs free baseline"
and "fno vs unet32" no longer depend on which side got a calibration step.

Post-processing, applied to a predicted 64x64 map using only that map:
  P0  none, as produced.
  P1  per-design min-max rescale to [0, 1]; a constant map becomes all zeros
      (same convention as diagnose_thermal._minmax_normalise).
  A   global affine a*m+b fitted by least squares on the TRAINING designs'
      pixels, clipped to [0, 1]. Free/fitted non-network predictors only
      (two fitted parameters); reported as P1+A (after P1) and A (after P0).

Stages (see DESIGN_RUNS.md "Symmetric re-evaluation, pre-registration"):
  train      neural arms, resumable; merges by (track, arch, fold, seed, frac).
  baselines  non-neural predictors (no training loop, seconds).
  analyze    parity gates, S1-S4 (+ Stage B), summary.md and an HTML report.

Usage (from flow/):
  python3 util/ml/congestion/training/symmetric_eval.py --stage train \\
      --track thermal --archs unet32,fno --seeds 0-9 --budget-min 30
  python3 util/ml/congestion/training/symmetric_eval.py --stage baselines
  python3 util/ml/congestion/training/symmetric_eval.py --stage analyze
"""

import argparse
import html
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "models"))
sys.path.insert(0, os.path.join(_HERE, "..", "loop"))

import diagnose_thermal as dt  # noqa: E402
from arch_sweep import (  # noqa: E402
    ARCHS,
    _blur_pred,
    _holm,
    _make_family_folds,
    _score,
    _verdict,
)
from laplacian_sweep import TRACKS, _parse_key  # noqa: E402
from train_lodo import family_of  # noqa: E402

_CONGESTION = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_DATA = os.path.join(_CONGESTION, "data")
DEFAULT_OUT = os.path.join(_CONGESTION, "experiments", "symmetric_eval.json")
DEFAULT_SUMMARY = os.path.join(_CONGESTION, "experiments", "symmetric_eval_summary.md")
DEFAULT_HTML = os.path.join(_CONGESTION, "experiments", "symmetric_eval_report.html")
STORED = {
    "thermal": os.path.join(_CONGESTION, "experiments", "arch_sweep.json"),
    "irdrop": os.path.join(_CONGESTION, "experiments", "irdrop_arch_sweep.json"),
}
DIAG = os.path.join(_CONGESTION, "experiments", "thermal_diagnostics.json")

EPOCHS = {"thermal": 200, "irdrop": 120}
BATCH_SIZE, LR = 4, 1e-3
TRACK_NAMES = ("thermal", "irdrop")
NEURAL = ("unet32", "fno")
BLUR_SIGMAS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
RIDGE_SIGMAS = dt.RIDGE_SIGMAS
RIDGE_CHANNELS = {"thermal": (0, 1, 2, 3), "irdrop": (0, 1, 2, 3, 4, 5)}
RIDGE_ALPHA = 1.0
NEURAL_PPS = ("P0", "P1")
FREE_PPS = ("P0", "P1", "P1+A", "A")
ALPHA_SIG = 0.05


# ---------------------------------------------------------------- post-processing


def p1(m: np.ndarray) -> np.ndarray:
    """Per-design min-max; a constant map becomes zeros."""
    m = m.astype(np.float64)
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)


def fit_affine(maps: np.ndarray, targets: np.ndarray):
    return dt.fit_affine(maps, targets)


def apply_affine(m: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip(a * m.astype(np.float64) + b, 0.0, 1.0)


def _mean_mse(preds: np.ndarray, targets: np.ndarray) -> float:
    return dt._per_design_mse(preds, targets)


def _max_norm(m: np.ndarray) -> np.ndarray:
    return dt._max_normalise(m)


def variant_maps(p0_maps: np.ndarray, targets: np.ndarray, train_idx):
    """The four post-processed variants of one predictor's P0 maps (n,H,W).
    The affine fits use `train_idx` designs only."""
    p1_maps = np.stack([p1(m) for m in p0_maps])
    a1, b1 = fit_affine(p1_maps[train_idx], targets[train_idx])
    a0, b0 = fit_affine(p0_maps[train_idx], targets[train_idx])
    return {
        "P0": p0_maps,
        "P1": p1_maps,
        "P1+A": apply_affine(p1_maps, a1, b1),
        "A": apply_affine(p0_maps, a0, b0),
    }


def select_sigma(maps_by_sigma: dict, targets: np.ndarray, train_idx, variant: str) -> float:
    """Sigma with the lowest training-design MSE of `variant`; ties to the
    smaller sigma. Uses training designs only."""
    train_mse = {
        s: _mean_mse(variant_maps(m, targets, train_idx)[variant][train_idx], targets[train_idx])
        for s, m in maps_by_sigma.items()
    }
    return dt.choose_sigma(train_mse)


# ---------------------------------------------------------------- non-neural predictors


def ridge_design_features(x_i: np.ndarray, channels, per_design_minmax: bool) -> np.ndarray:
    """Multi-scale blur features of one design, (H*W, n_channels*n_sigmas).
    With per_design_minmax each blurred feature map is min-max scaled over
    this design alone (no target information, no other design)."""
    feats = []
    for c in channels:
        for f in dt._blur_features(x_i[c], RIDGE_SIGMAS):
            feats.append(p1(f) if per_design_minmax else f)
    return np.stack(feats, axis=-1).reshape(-1, len(feats))


def baseline_results(track: str, x: np.ndarray, targets: np.ndarray, keys, folds):
    """{predictor|pp: {design: score}} plus {predictor|pp: {fold: sigma}}."""
    out: dict[str, dict] = {}
    sigmas: dict[str, dict] = {}

    def put(name, pp, design, pred, target):
        out.setdefault(f"{name}|{pp}", {})[design] = _score(pred, target)

    maps = {s: np.stack([_max_norm(gaussian_filter(xi[0], s)) for xi in x]) for s in BLUR_SIGMAS}
    blur3 = np.stack([_blur_pred(track, xi) for xi in x])
    feats = {
        mode: [ridge_design_features(xi, RIDGE_CHANNELS[track], mode == "mm") for xi in x]
        for mode in ("raw", "mm")
    }

    for fold in folds:
        train_idx, test_idx = dt.fold_indices(keys, fold)
        const = float(targets[train_idx].mean())
        mean_map = targets[train_idx].mean(axis=0)
        for i in test_idx:
            for name, pred in (("const_mean", np.full_like(targets[i], const)), ("mean_map", mean_map)):
                put(name, "P0", keys[i], pred, targets[i])
                put(name, "P1", keys[i], p1(pred), targets[i])

        v3 = variant_maps(blur3, targets, train_idx)
        for pp in FREE_PPS:
            for i in test_idx:
                put("blur3", pp, keys[i], v3[pp][i], targets[i])
        for pp in FREE_PPS:
            s = select_sigma(maps, targets, train_idx, pp)
            sigmas.setdefault(f"blur_sel|{pp}", {})[fold["name"]] = s
            vs = variant_maps(maps[s], targets, train_idx)
            for i in test_idx:
                put("blur_sel", pp, keys[i], vs[pp][i], targets[i])

        y = np.concatenate([targets[i].astype(np.float64).ravel() for i in train_idx])
        for mode in ("raw", "mm"):
            model = dt.ridge_fit(np.concatenate([feats[mode][i] for i in train_idx]), y, RIDGE_ALPHA)
            for i in test_idx:
                pred = dt.ridge_predict(model, feats[mode][i]).reshape(targets[i].shape)
                put(f"ridge_{mode}", "P0", keys[i], pred, targets[i])
                put(f"ridge_{mode}", "P1", keys[i], p1(pred), targets[i])
    return out, sigmas


# ---------------------------------------------------------------- neural arms


def train_and_predict(arch_name, track, train_ds, eval_ds, held_out_keys, seed, epochs, batch_size, lr, device):
    """Mirror of arch_sweep._run_torch (same seeding, flags, optimiser,
    schedule, clipping, loss, loaders, per-epoch eval pass, final-epoch
    model) that returns the held-out prediction maps instead of scores."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    spec = TRACKS[track]
    loss_fn = spec["loss"]
    target_key = spec["target_key"]
    arch = ARCHS[arch_name]
    forward = arch["forward"]

    held_out_set = set(held_out_keys)
    train_idx = [i for i, k in enumerate(train_ds.keys) if k not in held_out_set]
    eval_idx = [i for i, k in enumerate(eval_ds.keys) if k in held_out_set]

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(Subset(eval_ds, eval_idx), batch_size=batch_size, shuffle=False, num_workers=0)

    model = arch["build"](spec["in_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch[target_key].to(device)
            pred = forward(model, x)
            loss = loss_fn(pred, target, 0.0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                x = batch["x"].to(device)
                target = batch[target_key].to(device)
                loss_fn(forward(model, x), target, 0.0)
    wall_s = time.time() - t0

    model.eval()
    preds = []
    with torch.no_grad():
        for idx in eval_idx:
            sample = eval_ds[idx]
            x = sample["x"].unsqueeze(0).to(device)
            target = sample[target_key].unsqueeze(0).to(device)
            pred = forward(model, x)
            preds.append(
                {
                    "design": eval_ds.keys[idx],
                    "pred": pred.squeeze(0).squeeze(0).cpu().numpy(),
                    "target": target.squeeze(0).squeeze(0).cpu().numpy(),
                    "mse_torch": float(loss_fn(pred, target, 0.0).item()),
                }
            )
    return preds, wall_s


def neural_run_record(track, arch, fold, seed, frac, epochs, preds, wall_s):
    designs = {}
    for p in preds:
        p0 = _score(p["pred"], p["target"])
        p0["heldout_mse_torch"] = p["mse_torch"]
        designs[p["design"]] = {"P0": p0, "P1": _score(p1(p["pred"]), p["target"])}
    return {
        "track": track,
        "arch": arch,
        "fold": fold,
        "seed": seed,
        "frac": frac,
        "epochs": epochs,
        "wall_s": wall_s,
        "designs": designs,
    }


def run_key(r) -> tuple:
    return (r["track"], r["arch"], r["fold"], r["seed"], r["frac"])


def merge_runs(existing: list, new: list) -> list:
    """Union by run_key; a later record replaces an earlier one."""
    merged = {run_key(r): r for r in existing}
    merged.update({run_key(r): r for r in new})
    return list(merged.values())


# ---------------------------------------------------------------- persistence


def load_out(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"runs": [], "baselines": {}, "baseline_sigmas": {}, "timing": []}


def save_out(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def load_track_arrays(track, data_dir):
    ds = TRACKS[track]["dataset_cls"](data_dir, augment=False)
    keys = list(ds.keys)
    x = np.stack([ds[i]["x"].numpy() for i in range(len(ds))])
    targets = np.stack([ds[i][TRACKS[track]["target_key"]].numpy()[0] for i in range(len(ds))])
    return keys, x, targets


def stage_baselines(args):
    payload = load_out(args.out)
    for track in args.tracks:
        keys, x, targets = load_track_arrays(track, args.data_dir)
        folds = _make_family_folds(keys)
        t0 = time.time()
        res, sig = baseline_results(track, x, targets, keys, folds)
        payload["baselines"][track] = res
        payload["baseline_sigmas"][track] = sig
        payload["timing"].append(
            {"stage": "baselines", "track": track, "wall_s": time.time() - t0, "at": time.strftime("%F %T")}
        )
        print(f"baselines {track}: {len(res)} predictor variants, {time.time() - t0:.1f}s")
    save_out(args.out, payload)


def stage_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    track = args.track
    epochs = args.epochs or EPOCHS[track]
    dataset_cls = TRACKS[track]["dataset_cls"]
    train_ds = dataset_cls(args.data_dir, augment=True)
    eval_ds = dataset_cls(args.data_dir, augment=False)
    folds = _make_family_folds(eval_ds.keys)
    wanted = set(args.folds.split(",")) if args.folds else None
    if wanted and not wanted & {f["name"] for f in folds}:
        raise SystemExit(f"--folds {args.folds} matched no fold names")

    payload = load_out(args.out)
    done = {run_key(r) for r in payload["runs"]}
    t_start = time.time()
    n_new = 0
    stopped = False
    for seed in args.seeds:
        for arch in args.archs:
            for fi, fold in enumerate(folds):
                if wanted and fold["name"] not in wanted:
                    continue
                key = (track, arch, fold["name"], seed, args.fraction)
                if key in done:
                    continue
                if (time.time() - t_start) / 60.0 > args.budget_min:
                    stopped = True
                    break
                held = list(fold["held_out"])
                if args.fraction == 1.0:
                    tr, ev = train_ds, eval_ds
                else:
                    train_keys = [k for k in train_ds.keys if k not in set(held)]
                    allowed = dt.nested_train_keys(train_keys, fi, seed, args.fraction) + held
                    tr, ev = dt.RestrictedDataset(train_ds, allowed), dt.RestrictedDataset(eval_ds, allowed)
                preds, wall_s = train_and_predict(
                    arch, track, tr, ev, held, seed, epochs, BATCH_SIZE, LR, device
                )
                rec = neural_run_record(track, arch, fold["name"], seed, args.fraction, epochs, preds, wall_s)
                payload["runs"] = merge_runs(payload["runs"], [rec])
                save_out(args.out, payload)
                n_new += 1
                if args.verbose or n_new % 20 == 0:
                    print(
                        f"{track} {arch} fold={fold['name']} seed={seed} frac={args.fraction} "
                        f"wall_s={wall_s:.1f} runs_this_chunk={n_new}",
                        flush=True,
                    )
            if stopped:
                break
        if stopped:
            break
    payload = load_out(args.out)
    payload["timing"].append(
        {
            "stage": "train",
            "track": track,
            "archs": list(args.archs),
            "frac": args.fraction,
            "n_new_runs": n_new,
            "wall_s": time.time() - t_start,
            "budget_stopped": stopped,
            "at": time.strftime("%F %T"),
        }
    )
    save_out(args.out, payload)
    print(f"chunk done: {n_new} new runs in {time.time() - t_start:.1f}s (budget_stopped={stopped})")


# ---------------------------------------------------------------- analysis


def neural_series(runs, track, arch, pp, frac=1.0, seeds=None):
    """design -> {"mse": [per seed], "rho": [per seed], "mae": [...], "top10": [...]}.
    P0 MSE is the torch-float32 value (the bake-off's); P1 is numpy float64."""
    mse_field = "heldout_mse_torch" if pp == "P0" else "heldout_mse"
    out: dict[str, dict] = {}
    for r in sorted(runs, key=lambda r: r["seed"]):
        if (r["track"], r["arch"], r["frac"]) != (track, arch, frac):
            continue
        if seeds is not None and r["seed"] not in seeds:
            continue
        for d, v in r["designs"].items():
            e = out.setdefault(d, {"mse": [], "rho": [], "mae": [], "top10": []})
            s = v[pp]
            e["mse"].append(s[mse_field])
            e["rho"].append(s["spatial_spearman"])
            e["mae"].append(s["heldout_mae"])
            e["top10"].append(s["top10_abs_err"])
    return out


def free_series(baselines, track, name):
    return {
        d: {
            "mse": [s["heldout_mse"]],
            "rho": [s["spatial_spearman"]],
            "mae": [s["heldout_mae"]],
            "top10": [s["top10_abs_err"]],
        }
        for d, s in baselines[track][name].items()
    }


def _nanmean(v):
    v = [x for x in v if not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def per_design(series, field):
    return {d: _nanmean(e[field]) if field == "rho" else float(np.mean(e[field])) for d, e in series.items()}


def per_design_sd(series):
    return {d: float(np.std(e["mse"])) if len(e["mse"]) > 1 else 0.0 for d, e in series.items()}


def median_of(series, field):
    vals = [v for v in per_design(series, field).values() if not np.isnan(v)]
    return float(np.median(vals)) if vals else float("nan")


def compare(cand, base, designs_by_family, n_seeds_c, adjusted_p, emit, label):
    """_verdict for cand vs base (delta = cand - base, negative = cand better)."""
    common = sorted(set(cand) & set(base))
    mc, mb = per_design(cand, "mse"), per_design(base, "mse")
    deltas = {k: mc[k] - mb[k] for k in common}
    rc, rb = per_design(cand, "rho"), per_design(base, "rho")
    delta_rhos = {k: rc[k] - rb[k] for k in common if not (np.isnan(rc[k]) or np.isnan(rb[k]))}
    args = (per_design_sd(cand), per_design_sd(base), designs_by_family, n_seeds_c, adjusted_p)
    verdict, st = _verdict(deltas, delta_rhos, *args, emit, label)
    if not delta_rhos:
        # A constant comparator has no rank correlation, so the NaN rank
        # criterion in _verdict makes both Better and Worse unreachable.
        waived, _ = _verdict(deltas, {k: 0.0 for k in deltas}, *args, lambda s: None, label)
        st["verdict_rank_waived"] = waived
        emit(f"  (rank criterion undefined for this comparator; verdict with it waived: {waived})")
    return verdict, st


def paired_p(deltas) -> float:
    """Two-sided Wilcoxon signed-rank p; 1.0 when every difference is zero."""
    if not np.any(np.asarray(deltas) != 0):
        return 1.0
    return float(stats.wilcoxon(deltas)[1])


def wilcoxon_p(cand, base):
    common = sorted(set(cand) & set(base))
    mc, mb = per_design(cand, "mse"), per_design(base, "mse")
    return paired_p([mc[k] - mb[k] for k in common])


def run_comparisons(specs, designs_by_family, n_seeds, emit):
    """specs: list of (label, cand_series, base_series). Holm over all specs."""
    ps = [wilcoxon_p(c, b) for _, c, b in specs]
    adj = _holm(ps)
    out = []
    for (label, c, b), p, a in zip(specs, ps, adj):
        verdict, st = compare(c, b, designs_by_family, n_seeds, a, emit, label)
        emit(f"  -> {verdict}")
        out.append({"label": label, "raw_p": p, "verdict": verdict, **st})
    return out


def families_of(designs):
    return {k: family_of(_parse_key(k)[1]) for k in designs}


def s1_label(r_p0, r_p1):
    if r_p1["verdict"] == "Better":
        return "survives equal post-processing"
    if r_p1["verdict"] == "Worse":
        return "reverses"
    if r_p0["verdict"] == "Better":
        return "fno's win over unet32 is mostly output calibration"
    return "no fno win under either post-processing"


def stage_b_label(med75, med100, p):
    change = (med75 - med100) / med75
    if change >= 0.10 and p < ALPHA_SIG:
        return "data-limited", change
    if abs(change) < 0.05 and p >= ALPHA_SIG:
        return "saturated", change
    return "unresolved", change


def check_gates(payload, emit, stored_paths=None, diag_path=DIAG):
    stored_paths = stored_paths or STORED
    gates = {}
    for track in TRACK_NAMES:
        with open(stored_paths[track]) as f:
            stored = json.load(f)["runs"]
        ref = {(r["arch"], r["design"], r["seed"]): r["heldout_mse"] for r in stored if r["arch"] in NEURAL}
        n = ok = 0
        bad = []
        for r in payload["runs"]:
            if r["track"] != track or r["frac"] != 1.0 or r["seed"] > 4:
                continue
            for d, v in r["designs"].items():
                n += 1
                match = repr(v["P0"]["heldout_mse_torch"]) == repr(ref[(r["arch"], d, r["seed"])])
                ok += match
                if not match:
                    bad.append((r["arch"], d, r["seed"]))
        expected = 2 * 30 * 5
        gates[f"G-P0 {track}"] = {"matched": ok, "checked": n, "expected": expected, "pass": ok == n == expected, "bad": bad[:5]}
        ref_blur = {r["design"]: r["heldout_mse"] for r in stored if r["arch"] == "blur"}
        mine = payload["baselines"][track]["blur3|P0"]
        nb = sum(repr(mine[d]["heldout_mse"]) == repr(ref_blur[d]) for d in ref_blur)
        gates[f"G-blur {track}"] = {"matched": nb, "checked": len(ref_blur), "pass": nb == len(ref_blur) == 30}
    with open(diag_path) as f:
        diag = json.load(f)
    bl = payload["baselines"]["thermal"]
    sel = diag["d1"]["cell_blur"]["selected"]["calibrated"]
    comps = {
        "blur_sel|P1+A vs D1 calibrated": ({d: v["heldout_mse"] for d, v in bl["blur_sel|P1+A"].items()}, {d: v["heldout_mse"] for d, v in sel.items()}),
        "ridge_raw|P0 vs D3 ridge_inputs@alpha=1": ({d: v["heldout_mse"] for d, v in bl["ridge_raw|P0"].items()}, {d: v["heldout_mse"] for d, v in diag["d3"]["ridge_inputs@alpha=1"].items()}),
        "const_mean|P0 vs D1": ({d: v["heldout_mse"] for d, v in bl["const_mean|P0"].items()}, {d: v["heldout_mse"] for d, v in diag["d1"]["const_mean"].items()}),
        "mean_map|P0 vs D1": ({d: v["heldout_mse"] for d, v in bl["mean_map|P0"].items()}, {d: v["heldout_mse"] for d, v in diag["d1"]["mean_map"].items()}),
    }
    # The diagnostics' sigma grid stops at 8 and ours adds 12, so blur_sel is
    # only comparable on folds where 12 was not chosen (there the argmin over
    # the larger grid is the argmin over the smaller one).
    chosen = payload["baseline_sigmas"]["thermal"]["blur_sel|P1+A"]
    comparable = {d for d in sel if chosen[family_of(_parse_key(d)[1])] <= 8.0}
    for name, (mine, ref) in comps.items():
        if name.startswith("blur_sel"):
            ref = {d: v for d, v in ref.items() if d in comparable}
            mine = {d: v for d, v in mine.items() if d in comparable}
        diffs = [abs(mine[d] - ref[d]) for d in ref]
        gates[f"G-diag {name}"] = {
            "max_abs_diff": float(max(diffs)),
            "median_mine": float(np.median([mine[d] for d in ref])),
            "median_diag": float(np.median(list(ref.values()))),
            "n": len(diffs),
            "pass": max(diffs) < 1e-6,
        }
    for k, v in gates.items():
        emit(f"{k}: {'PASS' if v['pass'] else 'FAIL'} {json.dumps({a: b for a, b in v.items() if a != 'pass'})}")
    return gates


PREDICTOR_ORDER = ["const_mean", "mean_map", "blur3", "blur_sel", "ridge_raw", "ridge_mm"]


def predictor_table(payload, track, runs, seeds):
    """rows: (predictor, {pp: (median_mse, median_rho)})"""
    rows = []
    for name in PREDICTOR_ORDER:
        cells = {}
        for pp in FREE_PPS:
            key = f"{name}|{pp}"
            if key in payload["baselines"][track]:
                s = free_series(payload["baselines"], track, key)
                cells[pp] = (median_of(s, "mse"), median_of(s, "rho"))
        rows.append((name, cells))
    for arch in NEURAL:
        cells = {}
        for pp in NEURAL_PPS:
            s = neural_series(runs, track, arch, pp, 1.0, seeds)
            if s:
                cells[pp] = (median_of(s, "mse"), median_of(s, "rho"))
        rows.append((arch, cells))
    return rows


def analyze(payload, seeds, emit, gates=None):
    runs = payload["runs"]
    res: dict = {"tables": {}, "S1": {}, "S2": {}, "S3": {}, "S4": {}, "B": {}}
    n_seeds = len(seeds)

    def nser(track, arch, pp, frac=1.0, sd=seeds):
        s = neural_series(runs, track, arch, pp, frac, sd)
        if not s:
            raise SystemExit(f"No runs for {track} {arch} {pp} frac={frac}")
        short = {d: len(e["mse"]) for d, e in s.items() if len(e["mse"]) != len(sd)}
        if short:
            raise SystemExit(f"Incomplete seeds for {track} {arch} {pp} frac={frac}: {short}")
        return s

    for track in TRACK_NAMES:
        res["tables"][track] = [
            {"predictor": n, "cells": {pp: {"median_mse": a, "median_rho": b} for pp, (a, b) in c.items()}}
            for n, c in predictor_table(payload, track, runs, seeds)
        ]
        emit(f"## {track}: median MSE (median Spearman)")
        for n, c in predictor_table(payload, track, runs, seeds):
            emit(f"- {n}: " + ", ".join(f"{pp} {a:.5f} ({b:.3f})" for pp, (a, b) in c.items()))

    th_fam = families_of(nser("thermal", "fno", "P0"))
    emit("## S1 thermal: fno vs unet32 (Holm m=2)")
    r = run_comparisons(
        [(f"fno vs unet32 {pp}", nser("thermal", "fno", pp), nser("thermal", "unet32", pp)) for pp in NEURAL_PPS],
        th_fam,
        n_seeds,
        emit,
    )
    res["S1"] = {"comparisons": r, "label": s1_label(r[0], r[1])}
    emit(f"S1 label: {res['S1']['label']}")

    for track in TRACK_NAMES:
        fam = families_of(nser(track, "fno", "P0"))
        fs = lambda n: free_series(payload["baselines"], track, n)  # noqa: E731
        emit(f"## S2 {track}: learned P1 vs free scaled baselines (Holm m=4)")
        main = run_comparisons(
            [
                (f"{a} P1 vs {b}", nser(track, a, "P1"), fs(b))
                for b in ("blur_sel|P1", "ridge_mm|P1")
                for a in NEURAL
            ],
            fam,
            n_seeds,
            emit,
        )
        emit(f"## S2 {track} (flagged, favours baseline: two fitted parameters) vs blur_sel P1+A (Holm m=2)")
        flagged = run_comparisons(
            [(f"{a} P1 vs blur_sel|P1+A", nser(track, a, "P1"), fs("blur_sel|P1+A")) for a in NEURAL],
            fam,
            n_seeds,
            emit,
        )
        better_than_blur = any(c["verdict"] == "Better" for c in main if c["label"].endswith("blur_sel|P1"))
        res["S2"][track] = {
            "main": main,
            "flagged_P1A": flagged,
            "label": "a learned model is Better than blur_sel P1"
            if better_than_blur
            else "no demonstrated advantage over a free scaled baseline on this track",
        }
        emit(f"S2 {track} label: {res['S2'][track]['label']}")

    fam = families_of(nser("irdrop", "fno", "P0"))
    fs = lambda n: free_series(payload["baselines"], "irdrop", n)  # noqa: E731
    emit("## S3 irdrop: learned vs constant / mean map (Holm m=8)")
    specs = [
        (f"{a} {pp} vs {c}", nser("irdrop", a, pp), fs(f"{c}|P0"))
        for c in ("const_mean", "mean_map")
        for a in NEURAL
        for pp in NEURAL_PPS
    ]
    s3 = run_comparisons(specs, fam, n_seeds, emit)
    beats_const_p1 = any(c["verdict"] == "Better" for c in s3 if c["label"] in {f"{a} P1 vs const_mean" for a in NEURAL})
    res["S3"] = {
        "comparisons": s3,
        "label": "a learned model is Better than const_mean in P1"
        if beats_const_p1
        else "IR-drop learned models are not shown to beat a constant on MSE",
    }
    emit(f"S3 label: {res['S3']['label']}")
    emit("### S3 supplementary (not in the Holm family): learned P1 vs P1-processed comparators")
    supp = []
    for c in ("const_mean", "mean_map"):
        for a in NEURAL:
            cand, base = nser("irdrop", a, "P1"), fs(f"{c}|P1")
            sp = compare(cand, base, fam, n_seeds, wilcoxon_p(cand, base), emit, f"{a} P1 vs {c}|P1")
            supp.append({"label": f"{a} P1 vs {c}|P1", "verdict": sp[0], **sp[1]})
    res["S3"]["supplementary_vs_P1_comparators"] = supp

    emit("## S4 noise: seed sd of per-design MSE, and 5-seed vs 10-seed medians")
    for track in TRACK_NAMES:
        for arch in NEURAL:
            for pp in NEURAL_PPS:
                s10 = nser(track, arch, pp)
                s5 = nser(track, arch, pp, 1.0, [x for x in seeds if x <= 4])
                sd = list(per_design_sd(s10).values())
                row = {
                    "track": track,
                    "arch": arch,
                    "pp": pp,
                    "sd_median": float(np.median(sd)),
                    "sd_mean": float(np.mean(sd)),
                    "median_5seed": median_of(s5, "mse"),
                    "median_10seed": median_of(s10, "mse"),
                }
                res["S4"].setdefault(track, []).append(row)
                emit(
                    f"{track} {arch} {pp}: seed sd median {row['sd_median']:.5f} mean {row['sd_mean']:.5f}; "
                    f"median MSE 5 seeds {row['median_5seed']:.5f} vs 10 seeds {row['median_10seed']:.5f}"
                )

    if any(r["frac"] != 1.0 and r["track"] == "thermal" for r in runs):
        emit("## Stage B thermal: 75% vs 100% of training designs")
        for arch in NEURAL:
            for pp in NEURAL_PPS:
                s75, s100 = nser("thermal", arch, pp, 0.75), nser("thermal", arch, pp, 1.0)
                m75, m100 = per_design(s75, "mse"), per_design(s100, "mse")
                common = sorted(set(m75) & set(m100))
                deltas = [m75[k] - m100[k] for k in common]
                p = paired_p(deltas)
                label, change = stage_b_label(float(np.median([m75[k] for k in common])), float(np.median([m100[k] for k in common])), p)
                row = {
                    "arch": arch,
                    "pp": pp,
                    "median_75": float(np.median([m75[k] for k in common])),
                    "median_100": float(np.median([m100[k] for k in common])),
                    "rel_improvement_100_over_75": change,
                    "wilcoxon_p": p,
                    "n_100_better": int(sum(d > 0 for d in deltas)),
                    "n": len(common),
                    "label": label,
                }
                res["B"][f"{arch}|{pp}"] = row
                emit(f"B {arch} {pp}: median 75% {row['median_75']:.5f} -> 100% {row['median_100']:.5f} ({change:+.1%}), p={p:.4f}, 100% better on {row['n_100_better']}/{row['n']} -> {label}")
    if gates is not None:
        res["gates"] = gates
    return res


# ---------------------------------------------------------------- report


def _fmt(v, nd=4):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def _cmp_sentence(c):
    note = ""
    if (
        c["verdict"] == "Indistinguishable"
        and c["median_delta"] < 0
        and c["wins"] >= 20
        and c["adjusted_p"] < ALPHA_SIG
        and c["median_delta_rho"] < -0.02
    ):
        note = ", held back only by worse rank agreement"
    waived = ""
    if c.get("verdict_rank_waived") and c["verdict_rank_waived"] != c["verdict"]:
        waived = (
            f"; with the undefined rank check set aside (not pre-registered) it would read "
            f"{c['verdict_rank_waived']}"
        )
    return (
        f"{c['label']}: {c['verdict']} (median change in error {c['median_delta']:+.4f}, "
        f"lower for the first model on {c['wins']}/{c['n']} designs, adjusted p {c['adjusted_p']:.3g}, "
        f"lower for the first model in {c['families_negative']}/{c['n_families']} families{note}{waived})"
    )


def answers(res):
    out = []
    s1 = res["S1"]
    p0, p1_ = s1["comparisons"]
    out.append(
        f"Thermal, fno vs unet32: {s1['label']}. Verdict with no post-processing: {p0['verdict']} "
        f"(median error change {p0['median_delta']:+.4f}); after per-design rescaling: {p1_['verdict']} "
        f"(median error change {p1_['median_delta']:+.4f})."
    )
    for track in TRACK_NAMES:
        s2 = res["S2"][track]
        parts = "; ".join(_cmp_sentence(c) for c in s2["main"])
        name = {"thermal": "Thermal", "irdrop": "IR-drop"}[track]
        out.append(f"{name}, learned model vs free baseline: {s2['label']}. {parts}.")
        if track == "irdrop":
            med = {r["predictor"]: r["cells"] for r in res["tables"][track]}
            out[-1] += (
                f" Context: per-design rescaling (P1) raises the median error of both networks "
                f"(unet32 {med['unet32']['P0']['median_mse']:.4f} -> {med['unet32']['P1']['median_mse']:.4f}, "
                f"fno {med['fno']['P0']['median_mse']:.4f} -> {med['fno']['P1']['median_mse']:.4f}), "
                f"because their outputs are compressed and stretching them overshoots. "
                f"The P1 comparisons therefore say little about the unscaled models. Unscaled medians: unet32 {med['unet32']['P0']['median_mse']:.4f}, "
                f"fno {med['fno']['P0']['median_mse']:.4f}, ridge_raw {med['ridge_raw']['P0']['median_mse']:.4f}, "
                f"const_mean {med['const_mean']['P0']['median_mse']:.4f}."
            )
    s3 = res["S3"]
    out.append(
        f"IR-drop, learned model vs a constant: {s3['label']}. "
        + "; ".join(_cmp_sentence(c) for c in s3["comparisons"] if "const_mean" in c["label"])
        + ". Note: a constant has no spatial ranking, so the rank check is undefined and this "
        "comparison can only ever come out Indistinguishable by the pre-registered rule; the "
        "substantive result is the unscaled (P0) comparison, which shows no difference."
    )
    if res["B"]:
        out.append(
            "Thermal, would more training designs help (75% vs 100%): "
            + "; ".join(f"{k.replace('|', ' ')}: {v['label']} ({v['rel_improvement_100_over_75']:+.1%}, p={v['wilcoxon_p']:.3g})" for k, v in res["B"].items())
            + "."
        )
    return out


def build_html(res):
    esc = html.escape
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Symmetric re-evaluation</title><style>",
        ":root{color-scheme:light dark;--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ccc}",
        "@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--mut:#9a9a9a;--line:#444}}",
        "body{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem}",
        "table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid var(--line);padding:.25rem .6rem;text-align:right}",
        "th:first-child,td:first-child{text-align:left}.mut{color:var(--mut)}li{margin:.5rem 0}",
        "</style></head><body><h1>Do the learned models beat a properly scaled free baseline?</h1>",
        "<h2>Answers</h2><ol>",
    ]
    parts += [f"<li>{esc(a)}</li>" for a in answers(res)]
    parts.append("</ol>")
    for track in TRACK_NAMES:
        parts.append(f"<h2>{track}: every predictor, median over 30 designs</h2>")
        parts.append(
            "<p class='mut'>Each cell is median error (MSE, lower is better) with median rank agreement "
            "(Spearman, higher is better) in brackets. P1+A and A are only available to free baselines.</p>"
        )
        parts.append("<table><tr><th>predictor</th>" + "".join(f"<th>{pp}</th>" for pp in FREE_PPS) + "</tr>")
        for row in res["tables"][track]:
            cells = []
            for pp in FREE_PPS:
                c = row["cells"].get(pp)
                cells.append(f"<td>{_fmt(c['median_mse'], 4)} ({_fmt(c['median_rho'], 2)})</td>" if c else "<td class='mut'>-</td>")
            parts.append(f"<tr><td>{esc(row['predictor'])}</td>{''.join(cells)}</tr>")
        parts.append("</table>")
    parts += [
        "<h2>Glossary</h2><ul>",
        "<li><b>P0</b>: the predicted map exactly as produced.</li>",
        "<li><b>P1</b>: the predicted map stretched so its own lowest value is 0 and highest is 1. It uses nothing but the prediction itself. A flat map becomes all zeros.</li>",
        "<li><b>A</b>: two numbers (scale and offset) fitted on the training designs to line a free baseline up with the targets. Only given to baselines, so it favours them. P1+A applies it after P1; A alone applies it to the raw map.</li>",
        "<li><b>Median MSE</b>: mean squared error per design, then the middle value across the 30 designs. Lower is better.</li>",
        "<li><b>Spearman</b>: how well the predicted ordering of pixels (hot to cold) matches the truth; 1 is perfect, 0 is none. It ignores scaling.</li>",
        "<li><b>Better / Worse / Indistinguishable</b>: the project's rule, which needs every one of seven checks: median and mean error difference both larger than seed noise, at least 20 of 30 designs on the same side, adjusted p below 0.05, at least 7 of 9 design families agreeing, and rank agreement not moving the other way. Anything else is Indistinguishable.</li>",
        "<li><b>Design families</b>: designs from the same source (for example aes and aes_lvt) are held out together, so the 30 designs are not fully independent.</li>",
        "</ul></body></html>",
    ]
    return "".join(parts)


def write_summary(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def stage_analyze(args):
    payload = load_out(args.out)
    lines: list[str] = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("## Parity gates")
    gates = check_gates(payload, emit)
    res = analyze(payload, args.seeds, emit, gates)
    payload["analysis"] = res
    save_out(args.out, payload)
    write_summary(args.summary, lines)
    with open(args.html, "w") as f:
        f.write(build_html(res))
    failed = [k for k, v in gates.items() if not v["pass"]]
    if failed:
        raise SystemExit(f"Parity gate(s) failed: {failed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["train", "baselines", "analyze"], required=True)
    ap.add_argument("--track", choices=TRACK_NAMES, default="thermal")
    ap.add_argument("--archs", default="unet32,fno")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--folds", default=None, help="Comma-separated fold names")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=None, help="Override; default is the stored bake-off epochs")
    ap.add_argument("--budget-min", type=float, default=float("inf"))
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="print one line per training run (default: one line per 20 runs)",
    )
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary", default=DEFAULT_SUMMARY)
    ap.add_argument("--html", default=DEFAULT_HTML)
    ap.add_argument("--tracks", default=",".join(TRACK_NAMES), help="Tracks for --stage baselines")
    args = ap.parse_args()
    args.seeds = parse_seeds(args.seeds)
    args.archs = args.archs.split(",")
    args.tracks = args.tracks.split(",")
    for a in args.archs:
        if a not in NEURAL:
            raise SystemExit(f"--archs supports {NEURAL}, got {a!r}")
    {"train": stage_train, "baselines": stage_baselines, "analyze": stage_analyze}[args.stage](args)


if __name__ == "__main__":
    main()
