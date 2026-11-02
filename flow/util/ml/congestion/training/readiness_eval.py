"""
FNO production-readiness evaluation for the thermal track: the same models
`arch_sweep.py` already compared under its own bake-off harness, but trained
and scored under `train_thermal.py`'s actual production recipe (production
split, production batch size/epochs/num_workers, best-validation-checkpoint
selection) instead of the bake-off's own settings. See DESIGN_RUNS.md's "FNO
production-readiness pass, pre-registration" entry for the gates (G1-G4)
this tool's output feeds.

This module is read-only with respect to `arch_sweep.py`, `laplacian_sweep.py`,
`train_thermal.py`, and `models/thermal_arch.py` -- it only imports from them.
It never trains `unet16`/`unet8`/`xgb` (production never runs those); the only
learned arches it knows how to train are `unet32` (-> `thermal_arch`'s "unet"
at the default base_features=32) and `fno`. `blur` is also accepted and reuses
`arch_sweep._run_fixed` unchanged (no training).

Usage (from flow/):
  python3 util/ml/congestion/training/readiness_eval.py \\
      --archs unet32,fno,blur --seeds 0,1,2,3,4 \\
      --out util/ml/congestion/experiments/readiness_sweep.json

  python3 util/ml/congestion/training/readiness_eval.py \\
      --check-identical fileA.json fileB.json --archs fno,unet32 --seeds 0,1,2

  python3 util/ml/congestion/training/readiness_eval.py \\
      --compare unet32 fno_lr3e-4,fno_lr3e-3 --holm-m 2 \\
      --report-from util/ml/congestion/experiments/readiness_sweep.json,/tmp/readiness_lr.json

  python3 util/ml/congestion/training/readiness_eval.py \\
      --wirerc-report --report-from util/ml/congestion/experiments/readiness_sweep.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "models"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "loop"))

import arch_sweep  # noqa: E402
from laplacian_sweep import TRACKS  # noqa: E402
from thermal_arch import DEFAULT_ARCH_KWARGS, build_thermal_model, thermal_heatmap  # noqa: E402
from thermal_dataset import ThermalDataset, split_thermal_dataset  # noqa: E402
from train_lodo import family_of  # noqa: E402
from train_thermal import fit  # noqa: E402

# readiness_eval only ever trains the production architecture family
# ("unet32" at thermal_arch's default base_features=32, and "fno") plus the
# zero-parameter "blur" calibration reference -- unlike arch_sweep.py, it
# has no reason to touch unet16/unet8/xgb, since production never runs
# those either.
ARCH_LABEL_TO_THERMAL_ARCH = {"unet32": "unet", "fno": "fno"}
TRAINABLE_ARCH_LABELS = set(ARCH_LABEL_TO_THERMAL_ARCH)
ALL_ARCH_LABELS = TRAINABLE_ARCH_LABELS | {"blur"}

# Resolved relative to this file's location (training/), not the invocation
# cwd, so the refusal holds regardless of where the tool is run from.
_CONGESTION_DIR = os.path.dirname(_HERE)
_FORBIDDEN_OUT_PATHS = {
    os.path.realpath(os.path.join(_CONGESTION_DIR, "experiments", "arch_sweep.json")),
    os.path.realpath(os.path.join(_CONGESTION_DIR, "experiments", "irdrop_arch_sweep.json")),
}
_FORBIDDEN_CKPT_BASENAMES = {"thermal_best.pt", "thermal_last.pt"}


def _check_out_allowed(out_path: str) -> None:
    real = os.path.realpath(out_path)
    if real in _FORBIDDEN_OUT_PATHS:
        raise SystemExit(
            f"Refusing to write to {out_path}: it is one of the already-reviewed "
            "bake-off result files. Use a different --out."
        )
    if os.path.basename(real) in _FORBIDDEN_CKPT_BASENAMES:
        raise SystemExit(
            f"Refusing to write to {out_path}: it looks like the real production "
            "checkpoint path. readiness_eval.py never writes checkpoints to disk "
            "in train mode in any case."
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _dedup_key(r: dict) -> tuple:
    return (r["track"], r["protocol"], r["arch"], r["fold"], r["design"], r["seed"])


def _load_merged(paths_csv: str) -> dict:
    merged_designs: dict[str, dict] = {}
    merged_runs: dict[tuple, dict] = {}
    for path in paths_csv.split(","):
        with open(path) as f:
            payload = json.load(f)
        for d in payload.get("designs", []):
            merged_designs[d["key"]] = d
        for r in payload.get("runs", []):
            merged_runs[_dedup_key(r)] = r
    return {"designs": list(merged_designs.values()), "runs": list(merged_runs.values())}


def _designs_by_family(designs: list[dict]) -> dict[str, str]:
    return {d["key"]: family_of(d["design"]) for d in designs}


# ---------------------------------------------------------------------------
# Train mode
# ---------------------------------------------------------------------------


def _train_one(
    arch_label: str,
    thermal_arch_name: str,
    arch_kwargs: dict,
    fold: dict,
    seed,
    train_ds_full: ThermalDataset,
    eval_ds: ThermalDataset,
    args,
    device,
) -> list[dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    held_out = set(fold["held_out"])
    # Production's own random 70/15/15 split (split_thermal_dataset, fixed
    # seed 42), restricted to designs outside this fold's held-out family so
    # it can never leak into train/val -- see the pre-registration entry.
    non_heldout_idx = [i for i, k in enumerate(train_ds_full.keys) if k not in held_out]
    restricted = Subset(train_ds_full, non_heldout_idx)
    train_part, val_part, _test_part_unused = split_thermal_dataset(restricted)

    train_loader = DataLoader(
        train_part, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_part, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    model = build_thermal_model(thermal_arch_name, in_channels=5, **arch_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.time()
    result = fit(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        laplacian_weight=0.0,
        device=device,
    )
    wall_s = time.time() - t0

    scored_model = build_thermal_model(thermal_arch_name, in_channels=5, **arch_kwargs).to(device)
    scored_model.load_state_dict(result["best_state"])
    scored_model.eval()

    records = []
    arch_out = args.label or arch_label
    with torch.no_grad():
        for key in fold["held_out"]:
            idx = eval_ds.keys.index(key)
            sample = eval_ds[idx]
            x = sample["x"].unsqueeze(0).to(device)
            target = sample["thermal"].unsqueeze(0).to(device)
            pred = thermal_heatmap(scored_model, x)
            # torch/float32/GPU path, same convention arch_sweep._run_torch
            # uses for heldout_mse (see that module's precision note).
            heldout_mse_torch = float(torch.nn.functional.mse_loss(pred, target).item())
            pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()
            target_np = target.squeeze(0).squeeze(0).cpu().numpy()
            record = arch_sweep._base_record(
                "thermal", "lofo", arch_out, fold["name"], key, seed, args.epochs, n_params, wall_s
            )
            record.update(arch_sweep._score(pred_np, target_np))
            record["heldout_mse"] = heldout_mse_torch
            record["train_mse_final"] = result["train_mse_final"]
            record["epoch1_train_mse"] = result["epoch1_train_mse"]
            record["recipe"] = "production"
            record["lr"] = args.lr
            record["best_epoch"] = result["best_epoch"]
            record["best_val_mse"] = result["best_val_mse"]
            records.append(record)
    return records


def _train_blur(arch_label: str, fold: dict, eval_ds: ThermalDataset, args) -> list[dict]:
    recs = arch_sweep._run_fixed(arch_label, "thermal", eval_ds, fold["held_out"])
    arch_out = args.label or arch_label
    for r in recs:
        r["protocol"] = "lofo"
        r["fold"] = fold["name"]
        r["arch"] = arch_out
        r["recipe"] = "production"
        r["lr"] = None
        r["best_epoch"] = None
        r["best_val_mse"] = None
    return recs


def _train_mode(args):
    _check_out_allowed(args.out)

    for a in args.archs:
        if a not in ALL_ARCH_LABELS:
            raise SystemExit(
                f"Unknown arch {a!r}; readiness_eval.py only trains "
                f"{sorted(ALL_ARCH_LABELS)} (production never runs anything else)."
            )
    if args.deterministic_algorithms and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        raise SystemExit(
            "--deterministic-algorithms requires the CUBLAS_WORKSPACE_CONFIG "
            "environment variable to be set (e.g. CUBLAS_WORKSPACE_CONFIG=:4096:8)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds_full = ThermalDataset(args.data_dir, augment=True)
    eval_ds = ThermalDataset(args.data_dir, augment=False)
    assert train_ds_full.keys == eval_ds.keys

    designs = TRACKS["thermal"]["metadata_fn"](args.data_dir, eval_ds.keys)
    for d in designs:
        d["track"] = "thermal"
    print(f"Loaded {len(designs)} designs.")

    folds = arch_sweep._make_family_folds(eval_ds.keys)
    if args.folds is not None:
        wanted = set(args.folds)
        folds = [f for f in folds if f["name"] in wanted]
        if not folds:
            raise ValueError(f"--folds {args.folds} matched no fold names")

    runs = []
    for arch_label in args.archs:
        seeds = [None] if arch_label == "blur" else args.seeds
        total = len(folds) * len(seeds)
        done = 0
        for fold in folds:
            for seed in seeds:
                if arch_label == "blur":
                    recs = _train_blur(arch_label, fold, eval_ds, args)
                else:
                    thermal_arch_name = ARCH_LABEL_TO_THERMAL_ARCH[arch_label]
                    arch_kwargs = DEFAULT_ARCH_KWARGS[thermal_arch_name]
                    recs = _train_one(
                        arch_label,
                        thermal_arch_name,
                        arch_kwargs,
                        fold,
                        seed,
                        train_ds_full,
                        eval_ds,
                        args,
                        device,
                    )
                runs.extend(recs)
                done += 1
                mean_mse = np.mean([r["heldout_mse"] for r in recs]) if recs else float("nan")
                wall_s = recs[0]["wall_s"] if recs else float("nan")
                print(
                    f"[{done}/{total}] arch={arch_label} fold={fold['name']} seed={seed} "
                    f"n_designs={len(recs)} mean_heldout_mse={mean_mse:.5f} wall_s={wall_s:.1f}"
                )

    payload = {"designs": designs, "runs": runs}
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        existing_designs = {d["key"]: d for d in existing.get("designs", [])}
        existing_designs.update({d["key"]: d for d in designs})
        payload["designs"] = list(existing_designs.values())
        merged = {_dedup_key(r): r for r in existing.get("runs", [])}
        merged.update({_dedup_key(r): r for r in runs})
        payload["runs"] = list(merged.values())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(payload['runs'])} run records to {args.out}")


# ---------------------------------------------------------------------------
# --check-identical mode
# ---------------------------------------------------------------------------


def _check_identical_mode(args):
    path_a, path_b = args.check_identical
    with open(path_a) as f:
        payload_a = json.load(f)
    with open(path_b) as f:
        payload_b = json.load(f)

    archs = set(args.archs)
    seeds = set(args.seeds)

    def _index(payload):
        idx = {}
        for r in payload.get("runs", []):
            if r["arch"] not in archs:
                continue
            if r["seed"] not in seeds:
                continue
            idx[(r["arch"], r["fold"], r["design"], r["seed"])] = r["heldout_mse"]
        return idx

    idx_a = _index(payload_a)
    idx_b = _index(payload_b)
    common = sorted(set(idx_a) & set(idx_b))

    mismatches = []
    for key in common:
        va, vb = idx_a[key], idx_b[key]
        if repr(va) != repr(vb):
            mismatches.append((key, va, vb))

    print(f"{len(common)} matched, {len(mismatches)} mismatched")
    for key, va, vb in mismatches:
        print(f"  MISMATCH arch/fold/design/seed={key}: {va!r} != {vb!r}")

    if mismatches:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# --compare mode
# ---------------------------------------------------------------------------


def _compare_mode(args):
    base = args.compare[0]
    cands = args.compare[1].split(",")
    holm_m = args.holm_m if args.holm_m is not None else len(cands)

    payload = _load_merged(args.report_from)
    runs = payload["runs"]
    designs_by_family = _designs_by_family(payload["designs"])

    base_means = arch_sweep._per_design_means(runs, base, "heldout_mse")
    base_sds = arch_sweep._per_design_sds(runs, base, "heldout_mse")
    base_rhos = arch_sweep._per_design_means(runs, base, "spatial_spearman")

    raw_p = []
    per_cand = []
    for cand in cands:
        cand_means = arch_sweep._per_design_means(runs, cand, "heldout_mse")
        common = sorted(set(cand_means) & set(base_means))
        deltas = {k: cand_means[k] - base_means[k] for k in common}
        try:
            _, p = stats.wilcoxon(list(deltas.values()))
        except ValueError:
            p = float("nan")
        raw_p.append(p)
        per_cand.append((cand, deltas, common))

    pvals_padded = raw_p + [1.0] * max(holm_m - len(raw_p), 0)
    adjusted_padded = arch_sweep._holm(pvals_padded)
    adjusted = adjusted_padded[: len(raw_p)]

    for (cand, deltas, common), adj_p in zip(per_cand, adjusted):
        cand_sds = arch_sweep._per_design_sds(runs, cand, "heldout_mse")
        cand_rhos = arch_sweep._per_design_means(runs, cand, "spatial_spearman")
        n_seeds_c = len({r["seed"] for r in runs if r["arch"] == cand}) or 1
        delta_rhos = {
            k: cand_rhos[k] - base_rhos[k]
            for k in common
            if k in cand_rhos
            and k in base_rhos
            and not np.isnan(cand_rhos[k])
            and not np.isnan(base_rhos[k])
        }
        verdict, _vstats = arch_sweep._verdict(
            deltas,
            delta_rhos,
            cand_sds,
            base_sds,
            designs_by_family,
            n_seeds_c,
            adj_p,
            print,
            f"{cand} vs {base} (n={len(common)})",
        )
        print(f"  -> {verdict}")


# ---------------------------------------------------------------------------
# --wirerc-report mode
# ---------------------------------------------------------------------------

_GROUP_L_ASAP7 = {
    "asap7_aes_lvt_base",
    "asap7_ethmac_base",
    "asap7_ibex_base",
    "asap7_jpeg_base",
    "asap7_jpeg_lvt_base",
}
_GROUP_L_SKY = {"sky130hd_ibex_base", "sky130hd_jpeg_base"}
_GROUP_E = {
    "asap7_aes_base",
    "asap7_gcd_base",
    "asap7_riscv32i_base",
    "sky130hd_aes_base",
    "sky130hd_gcd_base",
    "sky130hd_riscv32i_base",
}
_GROUP_L = _GROUP_L_ASAP7 | _GROUP_L_SKY


def _wirerc_report_mode(args):
    payload = _load_merged(args.report_from)
    runs = payload["runs"]
    all_keys = {d["key"] for d in payload["designs"]}
    group_o = all_keys - _GROUP_E - _GROUP_L

    groups = [
        ("E", _GROUP_E & all_keys),
        ("L_asap7", _GROUP_L_ASAP7 & all_keys),
        ("L_sky", _GROUP_L_SKY & all_keys),
        ("L", _GROUP_L & all_keys),
        ("O", group_o),
    ]

    archs_present = sorted({r["arch"] for r in runs if r["arch"] in ("unet32", "fno", "blur")})
    per_arch_mse = {a: arch_sweep._per_design_means(runs, a, "heldout_mse") for a in archs_present}
    sds = {a: arch_sweep._per_design_sds(runs, a, "heldout_mse") for a in archs_present}
    n_seeds = {
        a: len({r["seed"] for r in runs if r["arch"] == a}) or 1 for a in archs_present
    }

    print(f"Arches present: {archs_present}")
    for group_name, keys in groups:
        for arch in archs_present:
            vals = [per_arch_mse[arch][k] for k in sorted(keys) if k in per_arch_mse[arch]]
            mean_mse = float(np.mean(vals)) if vals else float("nan")
            print(f"{group_name} (n={len(vals)}) {arch}: mean_mse={mean_mse:.6f}")
        if "fno" in per_arch_mse and "unet32" in per_arch_mse:
            common = sorted(k for k in keys if k in per_arch_mse["fno"] and k in per_arch_mse["unet32"])
            if common:
                delta = float(
                    np.mean([per_arch_mse["fno"][k] - per_arch_mse["unet32"][k] for k in common])
                )
                thresholds = [
                    np.sqrt(
                        (sds["unet32"].get(k, 0.0) ** 2 + sds["fno"].get(k, 0.0) ** 2)
                        / n_seeds["fno"]
                    )
                    for k in common
                ]
                T = float(np.mean(thresholds))
                print(
                    f"{group_name} (n={len(common)}): fno-unet32 mean_delta={delta:.6f} "
                    # G3 is signed, not absolute: the harmful direction is
                    # fno doing WORSE than unet32 (delta > 0). A favorable
                    # (negative) delta always passes regardless of magnitude.
                    f"T={T:.6f} {'<=T (pass)' if delta <= T else '>T (fail)'}"
                )
        print()

    print("## Per-design rows, E union L")
    for key in sorted((_GROUP_E | _GROUP_L) & all_keys):
        row = " ".join(
            f"{a}={per_arch_mse[a].get(key, float('nan')):.6f}" for a in archs_present
        )
        print(f"{key}: {row}")
    print()

    print("## Per-design rows, riscv32i family")
    for key in sorted(all_keys):
        pdk, design_key = key.split("_", 1)
        design = design_key.rsplit("_base", 1)[0]
        if family_of(design) != "riscv32i":
            continue
        row = " ".join(
            f"{a}={per_arch_mse[a].get(key, float('nan')):.6f}" for a in archs_present
        )
        print(f"{key}: {row}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default="util/ml/congestion/data")
    ap.add_argument("--archs", default="unet32,fno,blur")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--folds", default=None, help="Comma-separated fold names to restrict to")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--label", default=None, help="Override the arch field in written records")
    ap.add_argument("--deterministic-algorithms", action="store_true")
    ap.add_argument("--out", default="util/ml/congestion/experiments/readiness_sweep.json")

    ap.add_argument("--check-identical", nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument(
        "--compare", nargs=2, default=None, metavar=("BASE", "CANDS"),
        help="BASE arch name, CANDS comma-separated candidate arch names",
    )
    ap.add_argument("--holm-m", type=int, default=None)
    ap.add_argument(
        "--report-from", default=None,
        help="Comma-separated readiness_sweep-style JSON file path(s), read-only",
    )
    ap.add_argument("--wirerc-report", action="store_true")

    args = ap.parse_args()

    args.archs = args.archs.split(",")
    args.seeds = [int(s) for s in args.seeds.split(",")]
    if args.folds is not None:
        args.folds = args.folds.split(",")

    if args.wirerc_report:
        if not args.report_from:
            raise SystemExit("--wirerc-report requires --report-from")
        _wirerc_report_mode(args)
    elif args.check_identical is not None:
        _check_identical_mode(args)
    elif args.compare is not None:
        if not args.report_from:
            raise SystemExit("--compare requires --report-from")
        _compare_mode(args)
    else:
        _train_mode(args)


if __name__ == "__main__":
    main()
