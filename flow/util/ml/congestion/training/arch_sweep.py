"""
Architecture comparison for the thermal or IR-drop track: U-Net (3 widths)
vs a per-pixel XGBoost baseline vs a Fourier Neural Operator, with a free
Gaussian-blur baseline for calibration — under leakage-free design-family
holdout (not `laplacian_sweep.py`'s single-key LODO).

Why a separate harness (not an extension of `laplacian_sweep.py`'s
`TRACKS` registry): the tree arm has no gradient loop, the fold protocol
differs (family holdout vs key holdout — a family fold can hold out up to
7 designs at once), and evaluation must be per-design (`laplacian_sweep`'s
`_run_one` averages the whole held-out batch together, which would blur
distinct designs' scores). `laplacian_sweep.py` is left byte-for-byte
unchanged; this module imports from it (`TRACKS`, `_parse_key`,
`_spearman`) and from `loop/train_lodo.py` (`family_of`) rather than
duplicating them, so the two harnesses cannot drift apart silently.

Protocols:
  lofo (default) — leave-one-design-*family*-out, 9 folds. The real
                    comparison: every design's score comes from a model
                    that never saw any design in its family.
  lodo            — leave-one-design-*key*-out, 30 folds, single-key
                    holdout. Exists ONLY to reproduce `laplacian_sweep.py`'s
                    `unet32` numbers exactly, as a parity gate on this
                    harness's training loop; never use it for the
                    architecture comparison itself (a held-out `aes` key
                    still trains on six other `aes` placements).

Precision note: `heldout_mse` for the torch arms (unet32/16/8, fno) is
computed via the same torch/float32/GPU reduction `laplacian_sweep.py`
uses, so it is bit-reproducible with that harness's numbers (this is what
the parity gate checks). `heldout_mse` for `blur` and `xgb` is computed
via numpy/float64/CPU in `_score` instead, since there is no GPU forward
pass to match against. The two paths agree to about 9 significant figures
on this data, so the torch-vs-blur and torch-vs-xgb comparisons are not
materially affected, but the comparison is not bit-for-bit homogeneous
across all arms — only within the torch arms, and within blur/xgb.
`heldout_mae`, `spatial_spearman`, and `top10_abs_err` are always computed
via the numpy/float64/CPU path, for every arch, since none of those are
parity-gated against `laplacian_sweep.py`.

Usage (from flow/):
  python3 util/ml/congestion/training/arch_sweep.py \\
      --archs unet32,unet16,unet8,fno \\
      --seeds 0,1,2,3,4 --epochs 120 \\
      --out util/ml/congestion/experiments/arch_sweep.json

  python3 util/ml/congestion/training/arch_sweep.py \\
      --out util/ml/congestion/experiments/arch_sweep.json --analyze

  python3 util/ml/congestion/training/arch_sweep.py --track irdrop \\
      --archs unet32,unet16,unet8,fno,blur \\
      --seeds 0,1,2,3,4 --epochs 120 \\
      --out util/ml/congestion/experiments/irdrop_arch_sweep.json

  python3 util/ml/congestion/training/arch_sweep.py --track irdrop \\
      --out util/ml/congestion/experiments/irdrop_arch_sweep.json --analyze

Parity gate (must match laplacian_sweep.py's unet32/lodo numbers exactly):
  python3 util/ml/congestion/training/arch_sweep.py --protocol lodo \\
      --folds nangate45_gcd_base --archs unet32 --seeds 0 --epochs 3 \\
      --out /tmp/arch_parity.json

  python3 util/ml/congestion/training/arch_sweep.py --track irdrop \\
      --protocol lodo --folds nangate45_gcd_base --archs unet32 \\
      --seeds 0 --epochs 3 --out /tmp/ir_arch_parity.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "models"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "loop"))

from fno import FNO2d  # noqa: E402
from laplacian_sweep import TRACKS, _parse_key, _spearman  # noqa: E402
from thermal_metrics import blur_proxy, shape_metrics  # noqa: E402
from train_lodo import family_of  # noqa: E402
from unet import CongestionUNet  # noqa: E402

try:
    from pixel_tree import fit_predict as _xgb_fit_predict

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Corrected 16-key coarse-grid ("upsampled") stratum for the thermal track,
# per DESIGN_RUNS.md "2026-09-16 (later)" (the audit that found
# `laplacian_sweep.UPSAMPLED_KEYS` is a stale, wrong 5-key set). Computed
# fresh here rather than importing the stale constant: 5 originally
# documented + 7 of the 18 2026-09-16 designs that solved on a coarse
# native HotSpot grid + 4 pre-existing designs the audit additionally
# found to be coarse-grid by checking effective rank of the stored map.
UPSAMPLED_KEYS_CORRECTED = {
    "asap7_gcd_base",
    "asap7_aes_base",
    "asap7_riscv32i_base",
    "nangate45_gcd_base",
    "sky130hd_gcd_base",
    "ihp-sg13g2_gcd_base",
    "sky130hs_gcd_base",
    "asap7_aes_lvt_base",
    "asap7_ethmac_base",
    "asap7_ibex_base",
    "asap7_jpeg_base",
    "asap7_jpeg_lvt_base",
    "nangate45_aes_base",
    "nangate45_dynamic_node_base",
    "nangate45_ibex_base",
    "nangate45_tinyRocket_base",
}

HOLM_CANDIDATE_COUNT = 4  # unet16, unet8, fno, xgb — fixed regardless of availability


def _blur_cell_density(x: np.ndarray) -> np.ndarray:
    """IR-drop's zero-parameter blur baseline: a Gaussian blur of cell
    density (x[0]), max-normalised, computed on the fly since the IR-drop
    dataset has no pre-blurred channel (unlike thermal's x[4]). See
    DESIGN_RUNS.md's IR-drop architecture comparison pre-registration for
    the rationale."""
    b = blur_proxy(x[0])
    m = b.max()
    return b / m if m > 0 else b


def _blur_pred(track: str, x: np.ndarray) -> np.ndarray:
    if track == "thermal":
        return x[4]  # cell_density_blur, already /max-normalised
    if track == "irdrop":
        return _blur_cell_density(x)
    raise KeyError(f"No blur-baseline definition for track={track!r}")


def _thermal_strata_for_analysis(designs: dict, keys: list[str]):
    upsampled = [k for k in keys if k in UPSAMPLED_KEYS_CORRECTED]
    native = [k for k in keys if k not in UPSAMPLED_KEYS_CORRECTED]
    return [("upsampled", upsampled), ("native", native)], None


ANALYSIS = {
    "thermal": {
        "confound_key": "contrast",
        "confound_label": "log10 contrast",
        "confound_transform": "log10",
        "strata_fn": _thermal_strata_for_analysis,
        "report_only_rho_keys": [],
    },
    "irdrop": {
        "confound_key": "worst_drop_mv",
        "confound_label": "log10 worst_drop_mv",
        "confound_transform": "log10",
        "strata_fn": TRACKS["irdrop"]["strata_fn"],
        "report_only_rho_keys": ["occupancy", "rel_drop"],
    },
}


def _unet_build(base_features):
    return lambda c: CongestionUNet(in_channels=c, base_features=base_features, num_heatmap_layers=1)


def _unet_forward(model, x):
    return model(x).heatmap


ARCHS = {
    "unet32": {"kind": "torch", "build": _unet_build(32), "forward": _unet_forward},
    "unet16": {"kind": "torch", "build": _unet_build(16), "forward": _unet_forward},
    "unet8": {"kind": "torch", "build": _unet_build(8), "forward": _unet_forward},
    "fno": {"kind": "torch", "build": lambda c: FNO2d(in_channels=c), "forward": lambda m, x: m(x)},
    "blur": {"kind": "fixed"},
}
if XGB_AVAILABLE:
    ARCHS["xgb"] = {"kind": "tree", "fit_predict": _xgb_fit_predict}


def _make_family_folds(keys: list[str]) -> list[dict]:
    families: dict[str, list[str]] = {}
    for k in keys:
        _, design = _parse_key(k)
        fam = family_of(design)
        families.setdefault(fam, []).append(k)
    return [
        {"name": fam, "held_out": sorted(ks)} for fam, ks in sorted(families.items())
    ]


def _make_key_folds(keys: list[str]) -> list[dict]:
    """Single-key holdout, parity-gate use only — see module docstring."""
    return [{"name": k, "held_out": [k]} for k in sorted(keys)]


def _score(pred: np.ndarray, target: np.ndarray) -> dict:
    diff = pred.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    if np.std(pred) == 0 or np.std(target) == 0:
        rho = float("nan")
    else:
        rho = float(stats.spearmanr(pred.ravel(), target.ravel())[0])
    top10_abs_err = float(
        abs(shape_metrics(pred)["top10_ratio"] - shape_metrics(target)["top10_ratio"])
    )
    return {
        "heldout_mse": mse,
        "heldout_mae": mae,
        "spatial_spearman": rho,
        "top10_abs_err": top10_abs_err,
    }


def _base_record(track, protocol, arch, fold, design, seed, epochs, n_params, wall_s):
    return {
        "track": track,
        "protocol": protocol,
        "arch": arch,
        "fold": fold,
        "design": design,
        "seed": seed,
        "epochs": epochs,
        "n_params": n_params,
        "wall_s": wall_s,
        "heldout_mse_best_epoch": None,
        "train_mse_final": None,
        "epoch1_train_mse": None,
        "heldout_eval_mse_history": None,
    }


def _run_torch(
    arch_name: str,
    track: str,
    train_ds,
    eval_ds,
    held_out_keys: list[str],
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[list[dict], float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # cuDNN determinism (same flags as laplacian_sweep._run_one); FNO's
    # cuFFT path is not covered by this flag and is spot-checked separately
    # in Stage 1 (DESIGN_RUNS.md risk list).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    spec = TRACKS[track]
    loss_fn = spec["loss"]
    target_key = spec["target_key"]
    arch = ARCHS[arch_name]
    build = arch["build"]
    forward = arch["forward"]

    held_out_set = set(held_out_keys)
    train_idx = [i for i, k in enumerate(train_ds.keys) if k not in held_out_set]
    eval_idx = [i for i, k in enumerate(eval_ds.keys) if k in held_out_set]

    train_subset = Subset(train_ds, train_idx)
    eval_subset = Subset(eval_ds, eval_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build(spec["in_channels"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_mse = float("inf")
    train_mse_final = 0.0
    epoch1_train_mse = None
    heldout_eval_mse_history: list[float] = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_mse = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch[target_key].to(device)
            pred = forward(model, x)
            loss = loss_fn(pred, target, 0.0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                epoch_mse += loss_fn(pred, target, 0.0).item()
            n_batches += 1
        train_mse_final = epoch_mse / n_batches
        if epoch == 1:
            epoch1_train_mse = train_mse_final
        scheduler.step()

        # Kept every epoch (not just at the end) to preserve this harness's
        # RNG stream parity with laplacian_sweep._run_one -- see module
        # docstring and DESIGN_RUNS.md parity-gate entry.
        model.eval()
        eval_mse = 0.0
        n_eval_batches = 0
        with torch.no_grad():
            for batch in eval_loader:
                x = batch["x"].to(device)
                target = batch[target_key].to(device)
                pred = forward(model, x)
                eval_mse += loss_fn(pred, target, 0.0).item()
                n_eval_batches += 1
        eval_mse /= n_eval_batches
        heldout_eval_mse_history.append(eval_mse)
        if eval_mse < best_mse:
            best_mse = eval_mse
    wall_s = time.time() - t0

    model.eval()
    records = []
    with torch.no_grad():
        for idx in eval_idx:
            key = eval_ds.keys[idx]
            sample = eval_ds[idx]
            x = sample["x"].unsqueeze(0).to(device)
            target = sample[target_key].unsqueeze(0).to(device)
            pred = forward(model, x)
            # Computed via the same torch float32 reduction loss_fn uses
            # (not the numpy float64 one in _score) so this value is
            # bit-identical to laplacian_sweep.py's heldout_mse_final under
            # the lodo parity gate -- see module docstring.
            heldout_mse_torch = float(loss_fn(pred, target, 0.0).item())
            pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()
            target_np = target.squeeze(0).squeeze(0).cpu().numpy()
            record = _base_record(
                track, None, arch_name, None, key, seed, epochs, n_params, wall_s
            )
            record.update(_score(pred_np, target_np))
            record["heldout_mse"] = heldout_mse_torch
            record["heldout_mse_best_epoch"] = best_mse
            record["train_mse_final"] = train_mse_final
            record["epoch1_train_mse"] = epoch1_train_mse
            record["heldout_eval_mse_history"] = heldout_eval_mse_history
            records.append(record)
    return records, epoch1_train_mse


def _run_tree(
    arch_name: str, track: str, eval_ds, held_out_keys: list[str], seed: int
) -> list[dict]:
    spec = TRACKS[track]
    target_key = spec["target_key"]
    held_out_set = set(held_out_keys)

    train_x, train_y, test_x, test_keys = [], [], [], []
    for idx, key in enumerate(eval_ds.keys):
        sample = eval_ds[idx]
        x = sample["x"].numpy()
        y = sample[target_key].squeeze(0).numpy()
        if key in held_out_set:
            test_x.append(x)
            test_keys.append(key)
        else:
            train_x.append(x)
            train_y.append(y)

    t0 = time.time()
    preds = ARCHS[arch_name]["fit_predict"](train_x, train_y, test_x, seed)
    wall_s = time.time() - t0

    records = []
    for key, pred, x in zip(test_keys, preds, test_x):
        idx = eval_ds.keys.index(key)
        target_np = eval_ds[idx][target_key].squeeze(0).numpy()
        record = _base_record(track, None, arch_name, None, key, seed, None, 0, wall_s)
        record.update(_score(pred, target_np))
        records.append(record)
    return records


def _run_fixed(arch_name: str, track: str, eval_ds, held_out_keys: list[str]) -> list[dict]:
    spec = TRACKS[track]
    target_key = spec["target_key"]
    held_out_set = set(held_out_keys)

    records = []
    t0 = time.time()
    for idx, key in enumerate(eval_ds.keys):
        if key not in held_out_set:
            continue
        sample = eval_ds[idx]
        x = sample["x"].numpy()
        target_np = sample[target_key].squeeze(0).numpy()
        pred = _blur_pred(track, x)
        record = _base_record(track, None, arch_name, None, key, None, None, 0, time.time() - t0)
        record.update(_score(pred, target_np))
        records.append(record)
    return records


def _dedup_key(r: dict) -> tuple:
    return (r["track"], r["protocol"], r["arch"], r["fold"], r["design"], r["seed"])


def _infer_legacy_track(d: dict) -> str:
    if "worst_drop_mv" in d:
        return "irdrop"
    if "ptp_c" in d:
        return "thermal"
    return "unknown"


def _train_mode(args):
    if args.track == "irdrop" and "xgb" in args.archs:
        raise SystemExit(
            "--track irdrop refuses --archs xgb: the pixel_tree feature set "
            "(blur/cell/pin gaussian filters) is thermal-only."
        )
    for a in args.archs:
        if a not in ARCHS:
            if a == "xgb":
                raise SystemExit(
                    "--archs xgb requested but xgboost/pixel_tree.py is not "
                    "available on this host. Not installing automatically "
                    "(per project policy) -- run with the other arches."
                )
            raise SystemExit(f"Unknown arch {a!r}; available: {sorted(ARCHS)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    spec = TRACKS[args.track]
    dataset_cls = spec["dataset_cls"]
    train_ds = dataset_cls(args.data_dir, augment=True)
    eval_ds = dataset_cls(args.data_dir, augment=False)
    assert train_ds.keys == eval_ds.keys

    designs = spec["metadata_fn"](args.data_dir, eval_ds.keys)
    for d in designs:
        d["track"] = args.track
    print(f"Loaded {len(designs)} designs.")

    if args.protocol == "lofo":
        folds = _make_family_folds(eval_ds.keys)
    elif args.protocol == "lodo":
        folds = _make_key_folds(eval_ds.keys)
    else:
        raise ValueError(f"Unknown protocol: {args.protocol}")

    if args.folds is not None:
        wanted = set(args.folds)
        folds = [f for f in folds if f["name"] in wanted]
        if not folds:
            raise ValueError(f"--folds {args.folds} matched no fold names")

    runs = []
    for arch_name in args.archs:
        kind = ARCHS[arch_name]["kind"]
        seeds = [None] if kind == "fixed" else args.seeds
        total = len(folds) * len(seeds)
        done = 0
        for fold in folds:
            for seed in seeds:
                if kind == "torch":
                    recs, epoch1_mse = _run_torch(
                        arch_name,
                        args.track,
                        train_ds,
                        eval_ds,
                        fold["held_out"],
                        seed,
                        args.epochs,
                        args.batch_size,
                        args.lr,
                        device,
                    )
                elif kind == "tree":
                    recs = _run_tree(arch_name, args.track, eval_ds, fold["held_out"], seed)
                elif kind == "fixed":
                    recs = _run_fixed(arch_name, args.track, eval_ds, fold["held_out"])
                else:
                    raise ValueError(f"Unknown arch kind: {kind}")
                for r in recs:
                    r["protocol"] = args.protocol
                    r["fold"] = fold["name"]
                runs.extend(recs)
                done += 1
                mean_mse = np.mean([r["heldout_mse"] for r in recs]) if recs else float("nan")
                print(
                    f"[{done}/{total}] arch={arch_name} protocol={args.protocol} "
                    f"fold={fold['name']} seed={seed} n_designs={len(recs)} "
                    f"mean_heldout_mse={mean_mse:.5f} wall_s={(recs[0]['wall_s'] if recs else float('nan')):.1f}"
                )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {"designs": designs, "runs": runs}
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        existing_designs_raw = existing.get("designs", [])
        existing_design_tracks = {
            d.get("track") or _infer_legacy_track(d) for d in existing_designs_raw
        }
        if existing_design_tracks - {args.track}:
            raise SystemExit(
                f"Refusing to write track={args.track!r} designs into "
                f"{args.out}: it holds design metadata for track(s) "
                f"{sorted(existing_design_tracks - {args.track})}. Use a "
                "different --out for this track."
            )
        existing_designs = {
            (d.get("track") or _infer_legacy_track(d), d["key"]): d
            for d in existing_designs_raw
        }
        existing_designs.update({(args.track, d["key"]): d for d in designs})
        payload["designs"] = list(existing_designs.values())

        merged = {_dedup_key(r): r for r in existing.get("runs", [])}
        merged.update({_dedup_key(r): r for r in runs})
        payload["runs"] = list(merged.values())
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(payload['runs'])} run records to {args.out}")


def _per_design_means(runs: list[dict], arch: str, metric: str) -> dict[str, float]:
    by_design: dict[str, list[float]] = {}
    for r in runs:
        if r["arch"] != arch:
            continue
        by_design.setdefault(r["design"], []).append(r[metric])
    return {d: float(np.mean(v)) for d, v in by_design.items()}


def _per_design_sds(runs: list[dict], arch: str, metric: str) -> dict[str, float]:
    by_design: dict[str, list[float]] = {}
    for r in runs:
        if r["arch"] != arch:
            continue
        by_design.setdefault(r["design"], []).append(r[metric])
    return {d: float(np.std(v)) for d, v in by_design.items()}


def _holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction; returns adjusted p-values in
    the original order."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * pvalues[idx], 1.0)
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted


def _verdict(
    deltas: dict[str, float],
    delta_rhos: dict[str, float],
    sds_c: dict[str, float],
    sds_b: dict[str, float],
    designs_by_family: dict[str, str],
    n_seeds_c: int,
    adjusted_p: float,
    emit,
    label: str,
) -> tuple[str, dict]:
    keys = sorted(deltas)
    vals = [deltas[k] for k in keys]
    median_d = float(np.median(vals))
    mean_d = float(np.mean(vals))
    wins = sum(1 for v in vals if v < 0)
    losses = sum(1 for v in vals if v > 0)

    fam_means: dict[str, list[float]] = {}
    for k in keys:
        fam_means.setdefault(designs_by_family[k], []).append(deltas[k])
    fam_mean_vals = {f: float(np.mean(v)) for f, v in fam_means.items()}
    families_negative = sum(1 for v in fam_mean_vals.values() if v < 0)
    families_positive = sum(1 for v in fam_mean_vals.values() if v > 0)
    n_families = len(fam_mean_vals)

    thresholds = [
        np.sqrt((sds_b.get(k, 0.0) ** 2 + sds_c.get(k, 0.0) ** 2) / max(n_seeds_c, 1))
        for k in keys
    ]
    T = float(np.mean(thresholds))

    rho_vals = [delta_rhos[k] for k in keys if k in delta_rhos and not np.isnan(delta_rhos[k])]
    median_drho = float(np.median(rho_vals)) if rho_vals else float("nan")

    emit(
        f"{label}: median_delta={median_d:.6f} mean_delta={mean_d:.6f} "
        f"wins/losses={wins}/{losses} (n={len(vals)}) adj_p={adjusted_p:.5f} "
        f"T={T:.6f} families_neg/pos={families_negative}/{families_positive} "
        f"(n_families={n_families}) median_delta_rho={median_drho:.4f}"
    )

    better = (
        median_d < 0
        and wins >= 20
        and adjusted_p < 0.05
        and families_negative >= 7
        and abs(median_d) > T
        and abs(mean_d) > T
        and median_drho >= -0.02
    )
    worse = (
        median_d > 0
        and losses >= 20
        and adjusted_p < 0.05
        and families_positive >= 7
        and abs(median_d) > T
        and abs(mean_d) > T
        and median_drho <= 0.02
    )
    if better:
        verdict = "Better"
    elif worse:
        verdict = "Worse"
    else:
        verdict = "Indistinguishable"

    stats_out = {
        "median_delta": median_d,
        "mean_delta": mean_d,
        "wins": wins,
        "losses": losses,
        "n": len(vals),
        "adjusted_p": adjusted_p,
        "T": T,
        "families_negative": families_negative,
        "families_positive": families_positive,
        "n_families": n_families,
        "median_delta_rho": median_drho,
        "verdict": verdict,
    }
    return verdict, stats_out


def _analyze_mode(args):
    with open(args.out) as f:
        payload = json.load(f)
    designs = {
        d["key"]: d
        for d in payload["designs"]
        if (d.get("track") or _infer_legacy_track(d)) == args.track
    }
    if not designs:
        raise SystemExit(f"No design records for track={args.track!r} in {args.out}")

    runs = [
        r
        for r in payload["runs"]
        if r["protocol"] == "lofo" and (r.get("track") or "thermal") == args.track
    ]
    archs_present = sorted({r["arch"] for r in runs})
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("## Design metadata")
    columns = TRACKS[args.track]["metadata_columns"]
    emit("| Design | PDK | " + " | ".join(columns) + " |\n|---|---|" + "---|" * len(columns))
    for key in sorted(designs):
        d = designs[key]
        vals = [str(d[c]) if isinstance(d[c], (int, bool)) else f"{d[c]:.4f}" for c in columns]
        emit(f"| {key} | {d['pdk']} | " + " | ".join(vals) + " |")
    emit()

    emit("## Per-design held-out MSE (mean over seeds), by arch")
    emit("| Design | " + " | ".join(archs_present) + " |")
    emit("|" + "---|" * (1 + len(archs_present)))
    per_arch_mse = {a: _per_design_means(runs, a, "heldout_mse") for a in archs_present}
    for key in sorted(designs):
        row = [key] + [f"{per_arch_mse[a].get(key, float('nan')):.5f}" for a in archs_present]
        emit("| " + " | ".join(row) + " |")
    emit()

    designs_by_family = {}
    for key in designs:
        _, design_name = _parse_key(key)
        designs_by_family[key] = family_of(design_name)

    sds = {a: _per_design_sds(runs, a, "heldout_mse") for a in archs_present}
    n_seeds = {a: len(args.seeds) if a != "blur" else 1 for a in archs_present}

    candidates = [a for a in ["unet16", "unet8", "fno", "xgb"] if a in archs_present]
    pvalues_by_candidate = {}
    for cand in candidates:
        if cand not in per_arch_mse or "unet32" not in per_arch_mse:
            continue
        common = sorted(set(per_arch_mse[cand]) & set(per_arch_mse["unet32"]))
        deltas = [per_arch_mse[cand][k] - per_arch_mse["unet32"][k] for k in common]
        try:
            _, p = stats.wilcoxon(deltas)
        except ValueError:
            p = float("nan")
        pvalues_by_candidate[cand] = p

    emit("## Verdicts vs unet32 (Holm-corrected across the 4 pre-registered candidate tests)")
    holm_order = ["unet16", "unet8", "fno", "xgb"]
    pvals_in_order = [pvalues_by_candidate.get(c, 1.0) for c in holm_order]
    adjusted = _holm(pvals_in_order)
    adjusted_by_candidate = dict(zip(holm_order, adjusted))

    # Sensitivity cut (n-1): re-run the Wilcoxon test and the Holm correction
    # on the excluded set, rather than reusing the n=30 adjusted p-value —
    # criterion (3) must actually be re-tested at n-1, not assumed to carry
    # over from the full-sample result.
    pvalues_by_candidate_n1 = {}
    if args.sensitivity_exclude:
        for cand in holm_order:
            if cand not in per_arch_mse or "unet32" not in per_arch_mse:
                continue
            common_n1 = sorted(
                (set(per_arch_mse[cand]) & set(per_arch_mse["unet32"]))
                - {args.sensitivity_exclude}
            )
            deltas_n1_raw = [
                per_arch_mse[cand][k] - per_arch_mse["unet32"][k] for k in common_n1
            ]
            try:
                _, p_n1 = stats.wilcoxon(deltas_n1_raw)
            except ValueError:
                p_n1 = float("nan")
            pvalues_by_candidate_n1[cand] = p_n1
        pvals_n1_in_order = [pvalues_by_candidate_n1.get(c, 1.0) for c in holm_order]
        adjusted_n1 = _holm(pvals_n1_in_order)
        adjusted_by_candidate_n1 = dict(zip(holm_order, adjusted_n1))
    else:
        adjusted_by_candidate_n1 = {}

    report_vs_unet32 = []
    for cand in candidates:
        common = sorted(set(per_arch_mse[cand]) & set(per_arch_mse["unet32"]))
        deltas = {k: per_arch_mse[cand][k] - per_arch_mse["unet32"][k] for k in common}
        rho_c = _per_design_means(runs, cand, "spatial_spearman")
        rho_b = _per_design_means(runs, "unet32", "spatial_spearman")
        delta_rhos = {
            k: rho_c[k] - rho_b[k]
            for k in common
            if k in rho_c and k in rho_b and not np.isnan(rho_c[k]) and not np.isnan(rho_b[k])
        }
        verdict, vstats = _verdict(
            deltas,
            delta_rhos,
            sds.get(cand, {}),
            sds.get("unet32", {}),
            designs_by_family,
            n_seeds.get(cand, 1),
            adjusted_by_candidate.get(cand, 1.0),
            emit,
            f"{cand} vs unet32 (n={len(common)})",
        )
        emit(f"  -> {verdict}")
        row = {
            "candidate": cand,
            "baseline": "unet32",
            "baseline_median_mse": float(np.median([per_arch_mse["unet32"][k] for k in common])),
            "candidate_median_mse": float(np.median([per_arch_mse[cand][k] for k in common])),
            **vstats,
        }

        if args.sensitivity_exclude and args.sensitivity_exclude in deltas:
            common_n1 = [k for k in common if k != args.sensitivity_exclude]
            deltas_n1 = {k: deltas[k] for k in common_n1}
            verdict_n1, vstats_n1 = _verdict(
                deltas_n1,
                {k: v for k, v in delta_rhos.items() if k in common_n1},
                sds.get(cand, {}),
                sds.get("unet32", {}),
                designs_by_family,
                n_seeds.get(cand, 1),
                adjusted_by_candidate_n1.get(cand, 1.0),
                emit,
                f"{cand} vs unet32, sensitivity cut excluding {args.sensitivity_exclude} (n={len(common_n1)})",
            )
            if verdict == "Better" and verdict_n1 != "Better":
                emit(f"  -> Better (not robust to the sensitivity cut)")
            elif verdict == "Worse" and verdict_n1 != "Worse":
                emit(f"  -> Worse (not robust to the sensitivity cut)")
            else:
                emit(f"  -> sensitivity cut: {verdict_n1}")
            row["sensitivity_cut"] = {
                "excluded": args.sensitivity_exclude,
                "verdict": verdict_n1,
                **vstats_n1,
            }
        report_vs_unet32.append(row)
    emit()

    emit("## Calibration vs blur (no Holm correction)")
    report_vs_blur = []
    for cand in [a for a in archs_present if a != "blur"]:
        if "blur" not in per_arch_mse:
            continue
        common = sorted(set(per_arch_mse[cand]) & set(per_arch_mse["blur"]))
        deltas = {k: per_arch_mse[cand][k] - per_arch_mse["blur"][k] for k in common}
        rho_c = _per_design_means(runs, cand, "spatial_spearman")
        rho_b = _per_design_means(runs, "blur", "spatial_spearman")
        delta_rhos = {
            k: rho_c[k] - rho_b[k]
            for k in common
            if k in rho_c and k in rho_b and not np.isnan(rho_c[k]) and not np.isnan(rho_b[k])
        }
        try:
            _, p_cal = stats.wilcoxon(list(deltas.values()))
        except ValueError:
            p_cal = float("nan")
        verdict, vstats = _verdict(
            deltas,
            delta_rhos,
            sds.get(cand, {}),
            sds.get("blur", {}),
            designs_by_family,
            n_seeds.get(cand, 1),
            p_cal,
            emit,
            f"{cand} vs blur (n={len(common)})",
        )
        emit(f"  -> {verdict}")
        report_vs_blur.append(
            {
                "candidate": cand,
                "baseline": "blur",
                "baseline_median_mse": float(np.median([per_arch_mse["blur"][k] for k in common])),
                "candidate_median_mse": float(np.median([per_arch_mse[cand][k] for k in common])),
                **vstats,
            }
        )
    emit()

    analysis_cfg = ANALYSIS[args.track]
    confound_key = analysis_cfg["confound_key"]
    confound_label = analysis_cfg["confound_label"]
    confound_transform = analysis_cfg["confound_transform"]

    emit(f"## Confound: Spearman(delta vs unet32, {confound_label}) and strata")
    strata_note = None
    for cand in candidates:
        common = sorted(set(per_arch_mse[cand]) & set(per_arch_mse["unet32"]))
        deltas_ordered = [per_arch_mse[cand][k] - per_arch_mse["unet32"][k] for k in common]
        confound_vals = [designs[k][confound_key] for k in common]
        if confound_transform == "log10":
            confound_vals = [np.log10(v) for v in confound_vals]
        rho, p = _spearman(deltas_ordered, confound_vals)

        strata_groups, strata_note = analysis_cfg["strata_fn"](designs, common)
        (s1_label, s1_keys), (s2_label, s2_keys) = strata_groups

        thresholds = [
            np.sqrt((sds["unet32"].get(k, 0.0) ** 2 + sds[cand].get(k, 0.0) ** 2) / n_seeds[cand])
            for k in common
        ]
        T = float(np.mean(thresholds)) if thresholds else float("nan")
        s1_mean = float(
            np.mean([per_arch_mse[cand][k] - per_arch_mse["unet32"][k] for k in s1_keys])
        ) if s1_keys else float("nan")
        s2_mean = float(
            np.mean([per_arch_mse[cand][k] - per_arch_mse["unet32"][k] for k in s2_keys])
        ) if s2_keys else float("nan")
        opposite_sign_confound = (
            s1_keys
            and s2_keys
            and np.sign(s1_mean) != np.sign(s2_mean)
            and abs(s1_mean) > T
            and abs(s2_mean) > T
        )
        confound = (abs(rho) >= 0.6 and p < 0.05) or opposite_sign_confound
        emit(
            f"{cand}: rho(delta,{confound_label})={rho:.4f} p={p:.4f} "
            f"{s1_label}(n={len(s1_keys)})_mean_delta={s1_mean:.6f} "
            f"{s2_label}(n={len(s2_keys)})_mean_delta={s2_mean:.6f} "
            f"{'CONFOUNDED' if confound else 'no confound signal'}"
        )
    if strata_note:
        emit(f"    Note: {strata_note}")
    emit()

    if args.track == "irdrop":
        emit(
            "## Report-only (IR-drop): occupancy / rel_drop correlations, "
            "asap7 voltage-regime stratum"
        )
        for cand in candidates:
            common = sorted(set(per_arch_mse[cand]) & set(per_arch_mse["unet32"]))
            deltas_ordered = [
                per_arch_mse[cand][k] - per_arch_mse["unet32"][k] for k in common
            ]
            parts = []
            for rho_key in analysis_cfg["report_only_rho_keys"]:
                vals = [designs[k][rho_key] for k in common]
                rho, p = _spearman(deltas_ordered, vals)
                parts.append(f"rho(delta,{rho_key})={rho:.4f} p={p:.4f}")
            emit(f"{cand}: " + "  ".join(parts))
        for a in archs_present:
            asap7_vals = [
                per_arch_mse[a][k] for k in per_arch_mse[a] if designs[k]["pdk"] == "asap7"
            ]
            other_vals = [
                per_arch_mse[a][k] for k in per_arch_mse[a] if designs[k]["pdk"] != "asap7"
            ]
            emit(
                f"{a}: asap7(n={len(asap7_vals)})_mean_mse="
                f"{np.mean(asap7_vals) if asap7_vals else float('nan'):.5f} "
                f"other(n={len(other_vals)})_mean_mse="
                f"{np.mean(other_vals) if other_vals else float('nan'):.5f}"
            )
        emit()

    emit("## Report-only: wire-RC stratum, PDK, top10_abs_err, n_params, train_mse_final")
    # The 6 designs actually routed under the old (pre-2026-09-16) wire-RC
    # settings: the first-batch asap7 and sky130hd designs only (see
    # DESIGN_RUNS.md's 2026-09-16 (later) entry). nangate45 was never on the
    # affected platform list, and sky130hd/ibex and sky130hd/jpeg were routed
    # after the fix, so neither belongs in this set.
    wire_rc_early = {
        "asap7_aes_base",
        "asap7_gcd_base",
        "asap7_riscv32i_base",
        "sky130hd_aes_base",
        "sky130hd_gcd_base",
        "sky130hd_riscv32i_base",
    }
    for a in archs_present:
        early = [per_arch_mse[a][k] for k in per_arch_mse[a] if k in wire_rc_early]
        other = [per_arch_mse[a][k] for k in per_arch_mse[a] if k not in wire_rc_early]
        top10 = _per_design_means(runs, a, "top10_abs_err")
        n_params_vals = sorted({r["n_params"] for r in runs if r["arch"] == a})
        train_mse_vals = [r["train_mse_final"] for r in runs if r["arch"] == a and r["train_mse_final"] is not None]
        emit(
            f"{a}: wire_rc_early(n={len(early)})_mean_mse={np.mean(early) if early else float('nan'):.5f} "
            f"other(n={len(other)})_mean_mse={np.mean(other) if other else float('nan'):.5f} "
            f"mean_top10_abs_err={np.mean(list(top10.values())) if top10 else float('nan'):.5f} "
            f"n_params={n_params_vals} "
            f"mean_train_mse_final={np.mean(train_mse_vals) if train_mse_vals else float('nan'):.6f}"
        )
    emit()

    summary_path = os.path.splitext(args.out)[0] + "_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote summary to {summary_path}")

    # Structured counterpart to the markdown summary above, for
    # view_report.py to render as charts without recomputing any
    # statistic — this harness's verdict logic stays the single source of
    # truth for both the text and the chart.
    report_data_path = os.path.splitext(args.out)[0] + "_report_data.json"
    with open(report_data_path, "w") as f:
        json.dump(
            {
                "track": args.track,
                "n_designs": len(designs),
                "archs_present": archs_present,
                "vs_unet32": report_vs_unet32,
                "vs_blur": report_vs_blur,
            },
            f,
            indent=2,
        )
    print(f"Wrote report data to {report_data_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default="util/ml/congestion/data")
    ap.add_argument("--track", choices=["thermal", "irdrop"], default="thermal")
    ap.add_argument("--archs", default="unet32,unet16,unet8,fno,blur")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--protocol", choices=["lofo", "lodo"], default="lofo")
    ap.add_argument("--folds", default=None, help="Comma-separated fold names to restrict to")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--out", default="util/ml/congestion/experiments/arch_sweep.json"
    )
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--sensitivity-exclude", default=None)
    args = ap.parse_args()

    args.archs = args.archs.split(",")
    args.seeds = [int(s) for s in args.seeds.split(",")]
    if args.folds is not None:
        args.folds = args.folds.split(",")
    if args.sensitivity_exclude is None:
        args.sensitivity_exclude = TRACKS[args.track]["default_sensitivity_exclude"]

    if args.analyze:
        _analyze_mode(args)
    else:
        _train_mode(args)


if __name__ == "__main__":
    main()
