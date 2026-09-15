"""
Re-evaluate the Laplacian smoothness loss on the 12-design thermal dataset.

`train_thermal.py`'s hard-coded seed, 0.7/0.15/0.15 split (unusable at n=12,
degenerating to train 8 / val 1 / test 3), augmented validation subset, and
"best epoch" selection-on-the-eval-set make it unable to produce a fair,
reproducible comparison across `--laplacian-weight` values. This module adds
a leave-one-design-out (LODO) / leave-one-PDK-out (LOPO) sweep harness around
the *same* loss (`train_thermal._loss`), model, and optimizer settings, with
explicit seeding and an `augment=False` held-out evaluation set, so results
can be paired per design and averaged over seeds.

Usage (from flow/):
  python3 util/ml/congestion/training/laplacian_sweep.py \\
      --data-dir util/ml/congestion/data \\
      --out util/ml/congestion/experiments/laplacian_sweep.json \\
      --lambdas 0,0.01,0.1 --seeds 0,1,2 --protocol lodo --epochs 120

  python3 util/ml/congestion/training/laplacian_sweep.py \\
      --out util/ml/congestion/experiments/laplacian_sweep.json --analyze
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from thermal_dataset import ThermalDataset
from train_thermal import _loss
from unet import CongestionUNet

# Coarse-grid (<64x64 native HotSpot solve, bilinearly upsampled) designs
# documented in DESIGN_RUNS.md; used as a fallback stratification if
# `distinct_values` doesn't cleanly partition the dataset the same way.
UPSAMPLED_KEYS = {
    "asap7_gcd_base",
    "asap7_aes_base",
    "asap7_riscv32i_base",
    "nangate45_gcd_base",
    "sky130hd_gcd_base",
}


def _parse_key(key: str) -> tuple[str, str]:
    pdk, rest = key.split("_", 1)
    design = rest.rsplit("_base", 1)[0]
    return pdk, design


def _design_metadata(data_dir: str, keys: list[str]) -> list[dict]:
    designs = []
    for key in keys:
        pdk, design = _parse_key(key)
        therm = np.load(os.path.join(data_dir, f"{key}_thermal_labels.npz"))
        t = therm["thermal_map"].astype(np.float64)
        p = therm["power_grid"].astype(np.float64)
        ptp = float(t.max() - t.min())
        tmean = float(t.mean())
        contrast = ptp / (tmean - 45.0)
        corr = float(np.corrcoef(t.flatten(), p.flatten())[0, 1])
        designs.append(
            {
                "key": key,
                "pdk": pdk,
                "design": design,
                "ptp_c": ptp,
                "tmean_c": tmean,
                "contrast": contrast,
                "corr_t_p": corr,
                "distinct_values": int(np.unique(t).size),
            }
        )
    return designs


def _make_folds(designs: list[dict], protocol: str) -> list[dict]:
    if protocol == "lodo":
        return [{"name": d["key"], "held_out": [d["key"]]} for d in designs]
    if protocol == "lopo":
        pdks = sorted({d["pdk"] for d in designs})
        return [
            {
                "name": pdk,
                "held_out": [d["key"] for d in designs if d["pdk"] == pdk],
            }
            for pdk in pdks
        ]
    raise ValueError(f"Unknown protocol: {protocol}")


def _run_one(
    train_ds: ThermalDataset,
    eval_ds: ThermalDataset,
    held_out_keys: list[str],
    lam: float,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # cuDNN algorithm selection is nondeterministic across runs by default even
    # with a fixed seed; pin it so a same-seed rerun reproduces exactly. This
    # does not reduce the seed-to-seed variance the sweep measures.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    held_out_set = set(held_out_keys)
    train_idx = [i for i, k in enumerate(train_ds.keys) if k not in held_out_set]
    eval_idx = [i for i, k in enumerate(eval_ds.keys) if k in held_out_set]

    train_subset = Subset(train_ds, train_idx)
    eval_subset = Subset(eval_ds, eval_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    eval_loader = DataLoader(
        eval_subset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = CongestionUNet(in_channels=5, base_features=32, num_heatmap_layers=1).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_epoch = 0
    best_mse = float("inf")
    train_loss_final = 0.0
    train_mse_final = 0.0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch["thermal"].to(device)
            pred = model(x)
            thermal_pred = pred.heatmap
            loss = _loss(thermal_pred, target, lam)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                epoch_mse += _loss(thermal_pred, target, 0.0).item()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss_final = epoch_loss / n_batches
        train_mse_final = epoch_mse / n_batches
        scheduler.step()

        model.eval()
        eval_mse = 0.0
        n_eval_batches = 0
        with torch.no_grad():
            for batch in eval_loader:
                x = batch["x"].to(device)
                target = batch["thermal"].to(device)
                pred = model(x)
                eval_mse += _loss(pred.heatmap, target, 0.0).item()
                n_eval_batches += 1
        eval_mse /= n_eval_batches
        if eval_mse < best_mse:
            best_mse = eval_mse
            best_epoch = epoch

    model.eval()
    heldout_mse_final = 0.0
    heldout_mae_final = 0.0
    n_eval_batches = 0
    with torch.no_grad():
        for batch in eval_loader:
            x = batch["x"].to(device)
            target = batch["thermal"].to(device)
            pred = model(x)
            heldout_mse_final += _loss(pred.heatmap, target, 0.0).item()
            heldout_mae_final += (pred.heatmap - target).abs().mean().item()
            n_eval_batches += 1
    heldout_mse_final /= n_eval_batches
    heldout_mae_final /= n_eval_batches
    wall_s = time.time() - t0

    return {
        "lambda": lam,
        "seed": seed,
        "epochs": epochs,
        "heldout_mse_final": heldout_mse_final,
        "heldout_mae_final": heldout_mae_final,
        "heldout_mse_best_epoch": best_mse,
        "best_epoch": best_epoch,
        "train_loss_final": train_loss_final,
        "train_mse_final": train_mse_final,
        "wall_s": wall_s,
    }


def _train_mode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = ThermalDataset(args.data_dir, augment=True)
    eval_ds = ThermalDataset(args.data_dir, augment=False)
    assert train_ds.keys == eval_ds.keys

    designs = _design_metadata(args.data_dir, eval_ds.keys)
    print(f"Loaded {len(designs)} designs.")

    folds = _make_folds(designs, args.protocol)
    if args.folds is not None:
        wanted = set(args.folds)
        folds = [f for f in folds if f["name"] in wanted]
        if not folds:
            raise ValueError(f"--folds {args.folds} matched no fold names")

    lambdas = [float(x) for x in args.lambdas]
    seeds = [int(x) for x in args.seeds]

    runs = []
    total = len(folds) * len(lambdas) * len(seeds)
    done = 0
    for fold in folds:
        for lam in lambdas:
            for seed in seeds:
                record = _run_one(
                    train_ds,
                    eval_ds,
                    fold["held_out"],
                    lam,
                    seed,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    device,
                )
                record["protocol"] = args.protocol
                record["fold"] = fold["name"]
                runs.append(record)
                done += 1
                print(
                    f"[{done}/{total}] protocol={args.protocol} fold={fold['name']} "
                    f"lambda={lam} seed={seed} "
                    f"heldout_mse_final={record['heldout_mse_final']:.5f} "
                    f"wall_s={record['wall_s']:.1f}"
                )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {"designs": designs, "runs": runs}
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        existing_designs = {d["key"]: d for d in existing.get("designs", [])}
        existing_designs.update({d["key"]: d for d in designs})
        payload["designs"] = list(existing_designs.values())
        # Dedup on (protocol, fold, lambda, seed): a rerun of the same grid
        # to the same --out must replace, not duplicate, matching records —
        # otherwise --analyze silently averages stale and fresh runs
        # together (e.g. pre/post a determinism fix, as happened once here).
        merged = {
            (r["protocol"], r["fold"], r["lambda"], r["seed"]): r
            for r in existing.get("runs", [])
        }
        merged.update(
            {(r["protocol"], r["fold"], r["lambda"], r["seed"]): r for r in runs}
        )
        payload["runs"] = list(merged.values())
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(payload['runs'])} run records to {args.out}")


def _spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def _analyze_mode(args):
    with open(args.out) as f:
        payload = json.load(f)
    designs = {d["key"]: d for d in payload["designs"]}
    runs = [r for r in payload["runs"] if r["protocol"] == "lodo"]

    lambdas = sorted({r["lambda"] for r in runs})
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    # (1) per-design table
    emit("## Design metadata")
    emit(
        "| Design | PDK | ptp_c | contrast | corr_t_p | distinct_values |"
        "\n|---|---|---|---|---|---|"
    )
    for key in sorted(designs):
        d = designs[key]
        emit(
            f"| {key} | {d['pdk']} | {d['ptp_c']:.4f} | {d['contrast']:.4f} | "
            f"{d['corr_t_p']:.4f} | {d['distinct_values']} |"
        )
    emit()

    emit("## Per-design held-out MSE (mean +/- sd over seeds), final epoch")
    header = "| Design | " + " | ".join(f"lambda={l}" for l in lambdas)
    header += " | " + " | ".join(f"delta(lambda={l} - 0)" for l in lambdas if l != 0.0)
    header += " |"
    emit(header)
    emit("|" + "---|" * (1 + len(lambdas) + (len(lambdas) - 1)))

    per_design_mean = {key: {} for key in designs}
    per_design_sd = {key: {} for key in designs}
    for key in sorted(designs):
        row = [key]
        for lam in lambdas:
            vals = [
                r["heldout_mse_final"]
                for r in runs
                if r["fold"] == key and r["lambda"] == lam
            ]
            mean = float(np.mean(vals)) if vals else float("nan")
            sd = float(np.std(vals)) if vals else float("nan")
            per_design_mean[key][lam] = mean
            per_design_sd[key][lam] = sd
            row.append(f"{mean:.5f} +/- {sd:.5f}")
        base = per_design_mean[key].get(0.0, float("nan"))
        for lam in lambdas:
            if lam == 0.0:
                continue
            row.append(f"{per_design_mean[key][lam] - base:.5f}")
        emit("| " + " | ".join(row) + " |")
    emit()

    def aggregate(keys, label):
        emit(f"## Aggregate ({label}, n={len(keys)})")
        for lam in lambdas:
            if lam == 0.0:
                continue
            deltas = [
                per_design_mean[k][lam] - per_design_mean[k][0.0]
                for k in keys
                if lam in per_design_mean[k] and 0.0 in per_design_mean[k]
            ]
            if not deltas:
                continue
            wins = sum(1 for d in deltas if d < 0)
            losses = sum(1 for d in deltas if d > 0)
            ties = sum(1 for d in deltas if d == 0)
            try:
                wstat, wp = stats.wilcoxon(deltas)
            except ValueError:
                wstat, wp = float("nan"), float("nan")
            emit(
                f"lambda={lam}: mean_delta={np.mean(deltas):.5f} "
                f"median_delta={np.median(deltas):.5f} "
                f"wins/losses/ties={wins}/{losses}/{ties} "
                f"wilcoxon_p={wp:.5f}"
            )
        emit()
        return {
            lam: [
                per_design_mean[k][lam] - per_design_mean[k][0.0]
                for k in keys
                if lam in per_design_mean[k] and 0.0 in per_design_mean[k]
            ]
            for lam in lambdas
            if lam != 0.0
        }

    all_keys = sorted(designs)
    agg_n12 = aggregate(all_keys, "n=12")

    # (3) confound table
    emit("## Confound: Spearman(delta, contrast)")
    for lam in lambdas:
        if lam == 0.0:
            continue
        keys_ordered = [
            k
            for k in all_keys
            if lam in per_design_mean[k] and 0.0 in per_design_mean[k]
        ]
        deltas = [
            per_design_mean[k][lam] - per_design_mean[k][0.0] for k in keys_ordered
        ]
        contrasts = [designs[k]["contrast"] for k in keys_ordered]
        log_contrasts = [np.log10(c) for c in contrasts]
        rho, p = _spearman(deltas, contrasts)
        rho_log, p_log = _spearman(deltas, log_contrasts)
        emit(
            f"lambda={lam}: rho(delta,contrast)={rho:.4f} p={p:.4f}  "
            f"rho(delta,log10 contrast)={rho_log:.4f} p={p_log:.4f}"
        )

        for group_label, group_keys in [
            (
                "nangate45",
                [k for k in keys_ordered if designs[k]["pdk"] == "nangate45"],
            ),
            ("asap7", [k for k in keys_ordered if designs[k]["pdk"] == "asap7"]),
            ("sky130hd", [k for k in keys_ordered if designs[k]["pdk"] == "sky130hd"]),
        ]:
            gd = [per_design_mean[k][lam] - per_design_mean[k][0.0] for k in group_keys]
            if gd:
                emit(
                    f"    PDK={group_label} (n={len(gd)}): mean_delta={np.mean(gd):.5f}"
                )

        upsampled_keys = [k for k in keys_ordered if k in UPSAMPLED_KEYS]
        native_keys = [k for k in keys_ordered if k not in UPSAMPLED_KEYS]
        for group_label, group_keys in [
            ("upsampled", upsampled_keys),
            ("native", native_keys),
        ]:
            gd = [per_design_mean[k][lam] - per_design_mean[k][0.0] for k in group_keys]
            if gd:
                emit(
                    f"    stratum={group_label} (n={len(gd)}): mean_delta={np.mean(gd):.5f}"
                )
    emit()

    # (4) sensitivity — drop asap7_gcd_base
    keys_n11 = [k for k in all_keys if k != "asap7_gcd_base"]
    agg_n11 = aggregate(keys_n11, "n=11, asap7_gcd_base excluded")

    # (5) seed-noise floor
    emit("## Seed-noise floor (lambda=0)")
    sds = [per_design_sd[k][0.0] for k in all_keys if 0.0 in per_design_sd[k]]
    emit(f"mean per-design sd across seeds at lambda=0: {np.mean(sds):.5f}")
    emit()

    summary_path = os.path.splitext(args.out)[0] + "_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote summary to {summary_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default="util/ml/congestion/data")
    ap.add_argument(
        "--out", default="util/ml/congestion/experiments/laplacian_sweep.json"
    )
    ap.add_argument("--lambdas", default="0,0.01,0.1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--protocol", choices=["lodo", "lopo"], default="lodo")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--folds", default=None, help="Comma-separated fold names to restrict to"
    )
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()

    args.lambdas = args.lambdas.split(",")
    args.seeds = args.seeds.split(",")
    if args.folds is not None:
        args.folds = args.folds.split(",")

    if args.analyze:
        _analyze_mode(args)
    else:
        _train_mode(args)


if __name__ == "__main__":
    main()
