"""
Re-evaluate the Laplacian smoothness loss on the 12-design dataset, for
either the thermal or IR-drop track.

`train_thermal.py`/`train_irdrop.py`'s hard-coded seed, 0.7/0.15/0.15 split
(unusable at n=12, degenerating to train 8 / val 1 / test 3), augmented
validation subset, and "best epoch" selection-on-the-eval-set make them
unable to produce a fair, reproducible comparison across
`--laplacian-weight` values. This module adds a leave-one-design-out (LODO)
/ leave-one-PDK-out (LOPO) sweep harness around the *same* loss
(`train_thermal._loss` / `train_irdrop._loss`), model, and optimizer
settings, with explicit seeding and an `augment=False` held-out evaluation
set, so results can be paired per design and averaged over seeds.

Usage (from flow/):
  python3 util/ml/congestion/training/laplacian_sweep.py \\
      --data-dir util/ml/congestion/data \\
      --out util/ml/congestion/experiments/laplacian_sweep.json \\
      --lambdas 0,0.01,0.1 --seeds 0,1,2 --protocol lodo --epochs 120

  python3 util/ml/congestion/training/laplacian_sweep.py \\
      --out util/ml/congestion/experiments/laplacian_sweep.json --analyze

  python3 util/ml/congestion/training/laplacian_sweep.py \\
      --track irdrop --lambdas 0,0.01,0.1 --seeds 0,1,2,3,4 \\
      --protocol lodo --epochs 120

  python3 util/ml/congestion/training/laplacian_sweep.py \\
      --track irdrop --analyze
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

from irdrop_dataset import IRDropDataset
from thermal_dataset import ThermalDataset
from train_irdrop import _laplacian as _irdrop_laplacian
from train_irdrop import _loss as _irdrop_loss
from train_thermal import _laplacian as _thermal_laplacian
from train_thermal import _loss as _thermal_loss
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


def _thermal_metadata(data_dir: str, keys: list[str]) -> list[dict]:
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


def _irdrop_metadata(data_dir: str, keys: list[str]) -> list[dict]:
    designs = []
    for key in keys:
        pdk, design = _parse_key(key)
        drop = np.load(os.path.join(data_dir, f"{key}_irdrop_labels.npz"))
        d = drop["irdrop_map"].astype(np.float64)
        v = drop["voltage_map"].astype(np.float64)
        cur = drop["current_density_proxy"].astype(np.float64)
        worst_drop_mv = float(d.max() * 1e3)
        ptp_mv = float((d.max() - d.min()) * 1e3)
        v_nom = float((d + v).mean())
        rel_drop = (worst_drop_mv / 1e3) / v_nom
        if np.std(d) == 0 or np.std(cur) == 0:
            corr_d_i = float("nan")
        else:
            corr_d_i = float(np.corrcoef(d.flatten(), cur.flatten())[0, 1])
        distinct_values = int(np.unique(d).size)
        occupancy = distinct_values / 4096
        designs.append(
            {
                "key": key,
                "pdk": pdk,
                "design": design,
                "worst_drop_mv": worst_drop_mv,
                "ptp_mv": ptp_mv,
                "v_nom": v_nom,
                "rel_drop": rel_drop,
                "corr_d_i": corr_d_i,
                "distinct_values": distinct_values,
                "occupancy": occupancy,
                "degenerate": bool(ptp_mv == 0.0),
            }
        )
    return designs


def _thermal_strata(designs: dict, keys: list[str]):
    upsampled = [k for k in keys if k in UPSAMPLED_KEYS]
    native = [k for k in keys if k not in UPSAMPLED_KEYS]
    return [("upsampled", upsampled), ("native", native)], None


def _irdrop_strata(designs: dict, keys: list[str]):
    threshold = 0.5
    fill_dominated = [k for k in keys if designs[k]["occupancy"] < threshold]
    well_populated = [k for k in keys if designs[k]["occupancy"] >= threshold]
    note = None
    if len(fill_dominated) < 3 or len(well_populated) < 3:
        median = float(np.median([designs[k]["occupancy"] for k in keys]))
        fill_dominated = [k for k in keys if designs[k]["occupancy"] < median]
        well_populated = [k for k in keys if designs[k]["occupancy"] >= median]
        note = (
            f"0.5 occupancy threshold produced a degenerate split (n<3 on "
            f"one side); fell back to a median split at occupancy={median:.4f}."
        )
    return [("fill_dominated", fill_dominated), ("well_populated", well_populated)], note


TRACKS = {
    "thermal": {
        "dataset_cls": ThermalDataset,
        "loss": _thermal_loss,
        "laplacian_fn": _thermal_laplacian,
        "target_key": "thermal",
        "label_suffix": "_thermal_labels.npz",
        "in_channels": 5,
        "metadata_fn": _thermal_metadata,
        "metadata_columns": ["ptp_c", "contrast", "corr_t_p", "distinct_values"],
        "confound_specs": [{"key": "contrast", "transforms": ["raw", "log10"]}],
        "strata_fn": _thermal_strata,
        "default_out": "util/ml/congestion/experiments/laplacian_sweep.json",
        "default_sensitivity_exclude": "asap7_gcd_base",
    },
    "irdrop": {
        "dataset_cls": IRDropDataset,
        "loss": _irdrop_loss,
        "laplacian_fn": _irdrop_laplacian,
        "target_key": "irdrop",
        "label_suffix": "_irdrop_labels.npz",
        "in_channels": 6,
        "metadata_fn": _irdrop_metadata,
        "metadata_columns": [
            "worst_drop_mv",
            "ptp_mv",
            "rel_drop",
            "corr_d_i",
            "distinct_values",
            "occupancy",
        ],
        "confound_specs": [
            {"key": "rel_drop", "transforms": ["raw"]},
            {"key": "worst_drop_mv", "transforms": ["log10"]},
            {"key": "occupancy", "transforms": ["raw"]},
        ],
        "strata_fn": _irdrop_strata,
        "default_out": "util/ml/congestion/experiments/irdrop_laplacian_sweep.json",
        "default_sensitivity_exclude": "nangate45_gcd_base",
    },
}


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
    track: str,
    train_ds,
    eval_ds,
    held_out_keys: list[str],
    lam: float,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    strict_determinism: bool = False,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # cuDNN algorithm selection is nondeterministic across runs by default even
    # with a fixed seed; pin it so a same-seed rerun reproduces exactly. This
    # does not reduce the seed-to-seed variance the sweep measures.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict_determinism:
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as e:
            print(
                f"use_deterministic_algorithms(True) failed ({e}); "
                "falling back to cudnn-only determinism"
            )

    spec = TRACKS[track]
    loss_fn = spec["loss"]
    laplacian_fn = spec["laplacian_fn"]
    target_key = spec["target_key"]

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

    model = CongestionUNet(
        in_channels=spec["in_channels"], base_features=32, num_heatmap_layers=1
    ).to(device)
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
            target = batch[target_key].to(device)
            pred = model(x)
            heatmap_pred = pred.heatmap
            loss = loss_fn(heatmap_pred, target, lam)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                epoch_mse += loss_fn(heatmap_pred, target, 0.0).item()
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
                target = batch[target_key].to(device)
                pred = model(x)
                eval_mse += loss_fn(pred.heatmap, target, 0.0).item()
                n_eval_batches += 1
        eval_mse /= n_eval_batches
        if eval_mse < best_mse:
            best_mse = eval_mse
            best_epoch = epoch

    model.eval()
    heldout_mse_final = 0.0
    heldout_mae_final = 0.0
    smoothness_final = 0.0
    n_eval_batches = 0
    with torch.no_grad():
        for batch in eval_loader:
            x = batch["x"].to(device)
            target = batch[target_key].to(device)
            pred = model(x)
            heldout_mse_final += loss_fn(pred.heatmap, target, 0.0).item()
            heldout_mae_final += (pred.heatmap - target).abs().mean().item()
            smoothness_final += laplacian_fn(pred.heatmap).pow(2).mean().item()
            n_eval_batches += 1
    heldout_mse_final /= n_eval_batches
    heldout_mae_final /= n_eval_batches
    smoothness_final /= n_eval_batches
    wall_s = time.time() - t0

    return {
        "track": track,
        "lambda": lam,
        "seed": seed,
        "epochs": epochs,
        "heldout_mse_final": heldout_mse_final,
        "heldout_mae_final": heldout_mae_final,
        "heldout_mse_best_epoch": best_mse,
        "best_epoch": best_epoch,
        "train_loss_final": train_loss_final,
        "train_mse_final": train_mse_final,
        "smoothness_final": smoothness_final,
        "wall_s": wall_s,
    }


def _train_mode(args):
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
                    args.track,
                    train_ds,
                    eval_ds,
                    fold["held_out"],
                    lam,
                    seed,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    device,
                    args.strict_determinism,
                )
                record["protocol"] = args.protocol
                record["fold"] = fold["name"]
                runs.append(record)
                done += 1
                print(
                    f"[{done}/{total}] track={args.track} protocol={args.protocol} "
                    f"fold={fold['name']} lambda={lam} seed={seed} "
                    f"heldout_mse_final={record['heldout_mse_final']:.5f} "
                    f"wall_s={record['wall_s']:.1f}"
                )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {"designs": designs, "runs": runs}
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        existing_designs_raw = existing.get("designs", [])
        # Legacy (pre-track-tag) files have design records with no "track"
        # key at all. Infer their track from a distinguishing field
        # (_infer_legacy_track) rather than assuming "thermal" — a naive
        # default is wrong exactly when it matters: writing --track=thermal
        # into a legacy IR-drop file would default that file's own designs
        # to "thermal" too, making the two indistinguishable and the guard
        # a no-op for that direction. Field-signature inference correctly
        # flags a legacy IR-drop file regardless of which track is being
        # written, and vice versa. Post-fix files (every design already
        # tagged with "track") never hit this path — the explicit tags are
        # used as-is.
        existing_design_tracks = {
            d.get("track") or _infer_legacy_track(d) for d in existing_designs_raw
        }
        if existing_design_tracks - {args.track}:
            raise SystemExit(
                f"Refusing to write track={args.track!r} designs into "
                f"{args.out}: it holds design metadata for track(s) "
                f"{sorted(existing_design_tracks - {args.track})} "
                "(legacy/untagged records were inferred from their fields, "
                "not assumed). Use a different --out for this track to "
                "avoid destroying that designs block."
            )
        # Designs are keyed by (track, key), not just key: design keys
        # (e.g. "nangate45_gcd_base") are identical across tracks, so a
        # key-only merge would let an IR-drop write silently overwrite the
        # thermal designs block (or vice versa) even though the run
        # records themselves are safely track-qualified below.
        existing_designs = {
            (d.get("track") or _infer_legacy_track(d), d["key"]): d
            for d in existing_designs_raw
        }
        existing_designs.update({(args.track, d["key"]): d for d in designs})
        payload["designs"] = list(existing_designs.values())
        # Dedup on (track, protocol, fold, lambda, seed): a rerun of the same
        # grid to the same --out must replace, not duplicate, matching
        # records — otherwise --analyze silently averages stale and fresh
        # runs together (e.g. pre/post a determinism fix, as happened once
        # here). Records written before the "track" field existed are
        # backfilled as "thermal" so the pre-existing thermal artifact
        # dedups/analyzes identically to before this change.
        merged = {
            (
                r.get("track", "thermal"),
                r["protocol"],
                r["fold"],
                r["lambda"],
                r["seed"],
            ): r
            for r in existing.get("runs", [])
        }
        merged.update(
            {
                (
                    r.get("track", "thermal"),
                    r["protocol"],
                    r["fold"],
                    r["lambda"],
                    r["seed"],
                ): r
                for r in runs
            }
        )
        payload["runs"] = list(merged.values())
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(payload['runs'])} run records to {args.out}")


def _infer_legacy_track(d: dict) -> str:
    """Infer a pre-track-tag design record's track from a distinguishing
    field, rather than assuming "thermal". Run records can't be used for
    this (their schema is track-agnostic — heldout_mse_final/fold/lambda/
    seed exist identically in both tracks), which is why the write guard
    below inspects the *designs* block instead of run records: a legacy
    IR-drop file's designs always carry "worst_drop_mv" (thermal designs
    never do), so this correctly flags a thermal write into a legacy
    IR-drop file, not just the reverse."""
    if "worst_drop_mv" in d:
        return "irdrop"
    if "ptp_c" in d:
        return "thermal"
    return "unknown"


def _spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def _analyze_mode(args):
    with open(args.out) as f:
        payload = json.load(f)
    spec = TRACKS[args.track]
    designs = {
        d["key"]: d
        for d in payload["designs"]
        if (d.get("track") or _infer_legacy_track(d)) == args.track
    }
    if not designs:
        # A file that holds only the *other* track's data (e.g. a legacy,
        # pre-track-tag artifact, or --out pointed at the wrong file) would
        # otherwise silently filter to zero designs and emit a meaningless
        # empty/nan summary that overwrites whatever real summary was
        # there. Fail loudly instead.
        present = sorted(
            {(d.get("track") or _infer_legacy_track(d)) for d in payload["designs"]}
        )
        raise SystemExit(
            f"No design records for track={args.track!r} found in "
            f"{args.out} (it holds track(s) {present}). Wrong --out, or "
            "this track's sweep hasn't been run against this file yet."
        )
    runs = [
        r
        for r in payload["runs"]
        if r["protocol"] == "lodo"
        and (r.get("track") or "thermal") == args.track
    ]

    lambdas = sorted({r["lambda"] for r in runs})
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    # (1) per-design table
    columns = spec["metadata_columns"]
    emit("## Design metadata")
    emit(
        "| Design | PDK | "
        + " | ".join(columns)
        + " |\n|---|---|"
        + "---|" * len(columns)
    )
    for key in sorted(designs):
        d = designs[key]
        vals = []
        for col in columns:
            v = d[col]
            vals.append(str(v) if isinstance(v, (int, bool)) else f"{v:.4f}")
        emit(f"| {key} | {d['pdk']} | " + " | ".join(vals) + " |")
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
    agg_n12 = aggregate(all_keys, f"n={len(all_keys)}")

    # (3) confound table
    confound_labels = ", ".join(s["key"] for s in spec["confound_specs"])
    emit(f"## Confound: Spearman(delta, {confound_labels})")
    strata_groups, strata_note = spec["strata_fn"](designs, all_keys)
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

        parts = []
        for cspec in spec["confound_specs"]:
            key = cspec["key"]
            for transform in cspec["transforms"]:
                vals = [designs[k][key] for k in keys_ordered]
                if transform == "log10":
                    vals = [np.log10(v) for v in vals]
                    label = f"log10 {key}"
                else:
                    label = key
                rho, p = _spearman(deltas, vals)
                parts.append(f"rho(delta,{label})={rho:.4f} p={p:.4f}")
        emit(f"lambda={lam}: " + "  ".join(parts))

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

        for group_label, group_all_keys in strata_groups:
            group_keys = [k for k in group_all_keys if k in keys_ordered]
            gd = [per_design_mean[k][lam] - per_design_mean[k][0.0] for k in group_keys]
            if gd:
                emit(
                    f"    stratum={group_label} (n={len(gd)}): mean_delta={np.mean(gd):.5f}"
                )
    if strata_note:
        emit(f"    Note: {strata_note}")
    emit()

    # (4) sensitivity cut
    exclude_key = args.sensitivity_exclude
    if exclude_key is None:
        exclude_key = spec["default_sensitivity_exclude"]
    if exclude_key is not None:
        keys_n11 = [k for k in all_keys if k != exclude_key]
        agg_n11 = aggregate(keys_n11, f"n={len(keys_n11)}, {exclude_key} excluded")
    else:
        emit("## Sensitivity cut: skipped (no --sensitivity-exclude given)")
        emit()

    # (5) seed-noise floor
    emit(f"## Seed-noise floor (lambda=0)")
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
    ap.add_argument("--track", choices=["thermal", "irdrop"], default="thermal")
    ap.add_argument("--out", default=None)
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
    ap.add_argument(
        "--sensitivity-exclude",
        default=None,
        help="Fold name to drop for the n-1 sensitivity cut in --analyze",
    )
    ap.add_argument(
        "--strict-determinism",
        action="store_true",
        help="Attempt torch.use_deterministic_algorithms(True); falls back "
        "to cudnn-only determinism if unsupported by an op in the model",
    )
    args = ap.parse_args()

    if args.out is None:
        args.out = TRACKS[args.track]["default_out"]

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
