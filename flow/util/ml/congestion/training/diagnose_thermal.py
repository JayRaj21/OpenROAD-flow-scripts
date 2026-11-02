"""
Diagnostic study: why is the thermal U-Net statistically tied with the
zero-parameter blur baseline in the architecture bake-off?

Read-only with respect to the bake-off: imports its helpers from
`arch_sweep.py` / `thermal_dataset.py` and only ever reads
`experiments/arch_sweep.json` and `experiments/readiness_sweep.json`.

Known limits (found in review, see DESIGN_RUNS.md's results entry): the
"calibrated" blur and the oracle get a per-design min-max rescale that the
ridge model (D3) and the neural models do not, so D1-D3 are not an
apples-to-apples ranking; the R2/R3 labels printed by the report stage are
the pre-registered mechanical ones and overstate what 3 seeds can resolve.
All output goes to `experiments/thermal_diagnostics.json` (stage results,
merged per stage so every stage is re-runnable) and
`experiments/thermal_diagnostics_summary.md`.

Stages (`--stage`):
  p1      parity: D1 raw_max cell blur sigma=3 == stored `blur` arm, per design
  p2      parity: unet32 through the D4 wrapper at fraction 1.0 (ibex, gcd x
          seeds 0,1) is bit-identical to the stored unet32 records
  d1      floor and baselines (const_mean, mean_map, cell-density blur sweep)
  d2      oracle-input ceiling: blur sweep on the true power_grid
  d3      per-pixel ridge regression over a multi-scale filter bank
  d4      learning curve (trains; resumable, see --time-budget-s)
  d5      failure anatomy from the stored sweeps
  report  reading rules R1-R4 and the markdown summary

power_grid is NOT available at inference; D2 and the `ridge_inputs_plus_power`
variant are diagnostic oracles only. power_grid is divided by its own max per
design before use (scale differs by orders of magnitude across PDKs).

Run from flow/:
  python3 util/ml/congestion/training/diagnose_thermal.py --stage p1
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "models"))
sys.path.insert(0, os.path.join(_HERE, "..", "loop"))

import arch_sweep  # noqa: E402
from arch_sweep import (  # noqa: E402
    UPSAMPLED_KEYS_CORRECTED,
    _make_family_folds,
    _per_design_means,
    _score,
)
from thermal_dataset import BLUR_SIGMA, ThermalDataset  # noqa: E402

_CONGESTION = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_DATA = "util/ml/congestion/data"
DEFAULT_OUT = os.path.join(_CONGESTION, "experiments", "thermal_diagnostics.json")
DEFAULT_SUMMARY = os.path.join(_CONGESTION, "experiments", "thermal_diagnostics_summary.md")
DEFAULT_ARCH = os.path.join(_CONGESTION, "experiments", "arch_sweep.json")
DEFAULT_READY = os.path.join(_CONGESTION, "experiments", "readiness_sweep.json")

SIGMAS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
VARIANTS = ("raw_max", "minmax", "calibrated")
RIDGE_SIGMAS = (0.0, 1.0, 2.0, 4.0, 8.0)
RIDGE_CHANNELS = (0, 1, 2, 3)
RIDGE_ALPHAS = (0.01, 1.0, 100.0)
HEADLINE_ALPHA = 1.0
LC_FRACTIONS = (0.25, 0.5, 0.75)
META_FIELDS = ("ptp_c", "contrast", "corr_t_p", "distinct_values")
EPOCHS, BATCH_SIZE, LR = 200, 4, 1e-3
ROUND_DIGITS = 12


def load_designs(data_dir: str):
    ds = ThermalDataset(data_dir, augment=False)
    keys = list(ds.keys)
    x = np.stack([ds[i]["x"].numpy() for i in range(len(ds))])
    target = np.stack([ds[i]["thermal"].numpy()[0] for i in range(len(ds))])
    power = []
    for k in keys:
        p = np.load(os.path.join(data_dir, f"{k}_thermal_labels.npz"))["power_grid"].astype(np.float64)
        m = p.max()
        power.append(p / m if m > 0 else p)
    return ds, keys, x, target, np.stack(power)


def fold_indices(keys, fold):
    held = set(fold["held_out"])
    test_idx = [i for i, k in enumerate(keys) if k in held]
    train_idx = [i for i, k in enumerate(keys) if k not in held]
    return train_idx, test_idx


def _scored(pred, target):
    s = _score(pred, target)
    return {"heldout_mse": s["heldout_mse"], "spatial_spearman": s["spatial_spearman"]}


def _per_design_mse(preds, targets):
    return float(np.mean([np.mean((p.astype(np.float64) - t.astype(np.float64)) ** 2) for p, t in zip(preds, targets)]))


def _max_normalise(m):
    mx = m.max()
    return m / mx if mx > 0 else m


def _minmax_normalise(m):
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)


def fit_affine(m: np.ndarray, t: np.ndarray):
    """Least-squares a, b minimising ||a*m + b - t||^2 over all pooled pixels."""
    mf = m.astype(np.float64).ravel()
    tf = t.astype(np.float64).ravel()
    A = np.stack([mf, np.ones_like(mf)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, tf, rcond=None)
    return float(a), float(b)


def variant_predictions(blurred: np.ndarray, targets: np.ndarray, train_idx, variant: str):
    """Predictions for every design from blurred maps (n,64,64), fitting any
    global parameter on `train_idx` only."""
    if variant == "raw_max":
        return np.stack([_max_normalise(b) for b in blurred])
    mm = np.stack([_minmax_normalise(b.astype(np.float64)) for b in blurred])
    if variant == "minmax":
        return mm
    if variant == "calibrated":
        a, b = fit_affine(mm[train_idx], targets[train_idx])
        return np.clip(a * mm + b, 0.0, 1.0)
    raise ValueError(f"Unknown variant {variant!r}")


def choose_sigma(train_mse_by_sigma: dict) -> float:
    """Lowest training-fold MSE; ties go to the smaller sigma."""
    return min(sorted(train_mse_by_sigma), key=lambda s: train_mse_by_sigma[s])


def blur_fold(maps_by_sigma: dict, targets, train_idx, test_idx, variant: str):
    """Per-sigma training MSE and held-out predictions for one fold/variant."""
    out = {}
    for sigma, blurred in maps_by_sigma.items():
        pred = variant_predictions(blurred, targets, train_idx, variant)
        out[sigma] = {
            "train_mse": _per_design_mse(pred[train_idx], targets[train_idx]),
            "test_pred": pred[test_idx],
        }
    return out


def run_blur_study(maps_by_sigma: dict, targets, keys, folds):
    """Full sweep (every sigma, held-out) and train-selected result per variant."""
    sweep = {v: {str(s): {} for s in maps_by_sigma} for v in VARIANTS}
    selected = {v: {} for v in VARIANTS}
    for fold in folds:
        train_idx, test_idx = fold_indices(keys, fold)
        for v in VARIANTS:
            res = blur_fold(maps_by_sigma, targets, train_idx, test_idx, v)
            best = choose_sigma({s: r["train_mse"] for s, r in res.items()})
            for s, r in res.items():
                for j, i in enumerate(test_idx):
                    sc = _scored(r["test_pred"][j], targets[i])
                    sweep[v][str(s)][keys[i]] = sc
                    if s == best:
                        selected[v][keys[i]] = dict(sc, sigma=s)
    return {"sweep": sweep, "selected": selected}


def cell_maps(x):
    return {s: np.stack([gaussian_filter(xi[0], s) for xi in x]) for s in SIGMAS}


def power_maps(power):
    return {s: np.stack([gaussian_filter(p, s) for p in power]) for s in SIGMAS}


def baseline_floor(targets, keys, folds):
    const, mean_map = {}, {}
    for fold in folds:
        train_idx, test_idx = fold_indices(keys, fold)
        c = float(targets[train_idx].mean())
        mm = targets[train_idx].mean(axis=0)
        for i in test_idx:
            const[keys[i]] = _scored(np.full_like(targets[i], c), targets[i])
            mean_map[keys[i]] = _scored(mm, targets[i])
    return {"const_mean": const, "mean_map": mean_map}


def _blur_features(img, sigmas):
    return [img.astype(np.float64) if s == 0 else gaussian_filter(img.astype(np.float64), s) for s in sigmas]


def design_features(x_i, power_i, with_power: bool):
    feats = []
    for c in RIDGE_CHANNELS:
        feats += _blur_features(x_i[c], RIDGE_SIGMAS)
    if with_power:
        feats += _blur_features(power_i, RIDGE_SIGMAS)
    return np.stack(feats, axis=-1).reshape(-1, len(feats))


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float):
    """Ridge on standardised features with an unpenalised intercept, via the
    normal equations. Zero-variance features get sd=1 (they standardise to 0)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    y_mean = y.mean()
    G = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(G, Z.T @ (y - y_mean))
    return {"mu": mu, "sd": sd, "w": w, "b": float(y_mean)}


def ridge_predict(model, X):
    return np.clip(((X - model["mu"]) / model["sd"]) @ model["w"] + model["b"], 0.0, 1.0)


def run_ridge(x, power, targets, keys, folds):
    out = {}
    for name, with_power in (("ridge_inputs", False), ("ridge_inputs_plus_power", True)):
        feats = [design_features(x[i], power[i], with_power) for i in range(len(keys))]
        for alpha in RIDGE_ALPHAS:
            res = {}
            for fold in folds:
                train_idx, test_idx = fold_indices(keys, fold)
                X = np.concatenate([feats[i] for i in train_idx])
                y = np.concatenate([targets[i].astype(np.float64).ravel() for i in train_idx])
                model = ridge_fit(X, y, alpha)
                for i in test_idx:
                    pred = ridge_predict(model, feats[i]).reshape(targets[i].shape)
                    res[keys[i]] = _scored(pred, targets[i])
            out[f"{name}@alpha={alpha:g}"] = res
    return out


class RestrictedDataset:
    """Thin view of a ThermalDataset exposing only `allowed_keys`, with the
    attributes `arch_sweep._run_torch` uses (`keys`, `__len__`, `__getitem__`).
    The base dataset's order is preserved so a full-key view is identical to
    the unrestricted dataset."""

    def __init__(self, base, allowed_keys):
        allowed = set(allowed_keys)
        self._base = base
        self._idx = [i for i, k in enumerate(base.keys) if k in allowed]
        self.keys = [base.keys[i] for i in self._idx]

    def __len__(self):
        return len(self._idx)

    def __getitem__(self, i):
        return self._base[self._idx[i]]


def nested_train_keys(train_keys, fold_index: int, seed: int, frac: float):
    """First round(frac*n) keys of one fixed seeded permutation of the sorted
    training keys, so smaller fractions are nested inside larger ones."""
    ordered = sorted(train_keys)
    perm = np.random.default_rng([fold_index, seed]).permutation(len(ordered))
    k = max(1, int(round(frac * len(ordered))))
    return [ordered[i] for i in perm[:k]]


def run_wrapped(arch, train_ds, eval_ds, fold, fold_index, seed, frac, device, epochs):
    held = list(fold["held_out"])
    train_keys = [k for k in train_ds.keys if k not in set(held)]
    allowed = nested_train_keys(train_keys, fold_index, seed, frac) + held
    recs, _ = arch_sweep._run_torch(
        arch,
        "thermal",
        RestrictedDataset(train_ds, allowed),
        RestrictedDataset(eval_ds, allowed),
        held,
        seed,
        epochs,
        BATCH_SIZE,
        LR,
        device,
    )
    return recs


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def update_out(path, stage, payload):
    existing = _load_json(path) if os.path.exists(path) else {}
    existing[stage] = payload
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def stored_runs(path):
    return _load_json(path)["runs"]


def stored_per_design(path, arch, metric="heldout_mse", seeds=None):
    runs = [r for r in stored_runs(path) if seeds is None or r["seed"] in seeds]
    return _per_design_means(runs, arch, metric)


def _median(vals):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], dtype=np.float64)
    return float(np.median(v)) if v.size else float("nan")


def _mean(vals):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], dtype=np.float64)
    return float(np.mean(v)) if v.size else float("nan")


def summarise(per_design: dict, field="heldout_mse"):
    return _median([r[field] for r in per_design.values()]), _mean([r[field] for r in per_design.values()])


def reading_rules(m_unet, m_oracle, lc75, lc100, m_ridge, m_cal3, m_raw3):
    r1_ratio = round(m_oracle / m_unet, ROUND_DIGITS)
    if r1_ratio <= 0.5:
        r1 = "input features are the bottleneck (the real power map is far more predictive than anything the model sees)"
    elif r1_ratio >= 0.8:
        r1 = "even the true power map does not help; the limit is the label/physics/normalisation, not the inputs"
    else:
        r1 = "partially input-limited"
    drop = round((lc75 - lc100) / lc75, ROUND_DIGITS)
    if drop >= 0.10:
        r2 = "data-limited (more designs should help)"
    elif drop < 0.05:
        r2 = "saturated (more designs of the same kind will not help much)"
    else:
        r2 = "ambiguous"
    rel = round((m_ridge - m_unet) / m_unet, ROUND_DIGITS)
    if abs(rel) <= 0.10:
        r3 = "the U-Net is doing no more than a linear filter bank over these inputs"
    elif rel < 0:
        r3 = "a linear filter bank beats the U-Net by more than 10% (the U-Net underperforms even a linear model)"
    else:
        r3 = "the U-Net beats the linear filter bank by more than 10% (nonlinearity is doing real work)"
    if m_cal3 < m_unet:
        r4 = "calibrated blur beats unet32: calibration changes the 'U-Net ties blur' conclusion in blur's favour"
    elif m_cal3 > m_unet:
        r4 = "unet32 still beats calibrated blur on median MSE"
    else:
        r4 = "calibrated blur and unet32 tie exactly"
    return {
        "R1": {"ratio": r1_ratio, "label": r1},
        "R2": {"drop": drop, "label": r2},
        "R3": {"rel_diff": rel, "label": r3},
        "R4": {"m_calibrated_sigma3": m_cal3, "m_raw_sigma3": m_raw3, "m_unet32": m_unet, "label": r4},
    }


def stage_p1(args):
    ds, keys, x, target, power = load_designs(args.data_dir)
    stored = {r["design"]: r["heldout_mse"] for r in stored_runs(args.arch_sweep) if r["arch"] == "blur"}
    res = {}
    for i, k in enumerate(keys):
        pred = _max_normalise(gaussian_filter(x[i][0], BLUR_SIGMA))
        mine = _score(pred, target[i])["heldout_mse"]
        res[k] = {"mine": repr(mine), "stored": repr(stored[k]), "match": repr(mine) == repr(stored[k])}
        ch4 = _score(x[i][4], target[i])["heldout_mse"]
        res[k]["channel4_match"] = repr(ch4) == repr(stored[k])
    ok = len(res) == len(stored) == 30 and all(v["match"] and v["channel4_match"] for v in res.values())
    print(f"P1 {'PASS' if ok else 'FAIL'}: {sum(v['match'] for v in res.values())}/{len(res)} designs match stored blur arm")
    update_out(args.out, "p1", {"pass": ok, "designs": res})
    if not ok:
        raise SystemExit("P1 failed")


def stage_p2(args):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ThermalDataset(args.data_dir, augment=True)
    eval_ds = ThermalDataset(args.data_dir, augment=False)
    folds = _make_family_folds(eval_ds.keys)
    stored = {
        (r["design"], r["seed"]): r["heldout_mse"]
        for r in stored_runs(args.arch_sweep)
        if r["arch"] == "unet32"
    }
    res = {}
    for fi, fold in enumerate(folds):
        if fold["name"] not in ("ibex", "gcd"):
            continue
        for seed in (0, 1):
            for r in run_wrapped("unet32", train_ds, eval_ds, fold, fi, seed, 1.0, device, EPOCHS):
                ref = stored[(r["design"], seed)]
                res[f"{r['design']}|seed{seed}"] = {
                    "mine": repr(r["heldout_mse"]),
                    "stored": repr(ref),
                    "match": repr(r["heldout_mse"]) == repr(ref),
                }
    ok = bool(res) and all(v["match"] for v in res.values())
    print(f"P2 {'PASS' if ok else 'FAIL'}: {sum(v['match'] for v in res.values())}/{len(res)} records bit-identical")
    update_out(args.out, "p2", {"pass": ok, "records": res})
    if not ok:
        raise SystemExit("P2 failed")


def stage_d1(args):
    ds, keys, x, target, power = load_designs(args.data_dir)
    folds = _make_family_folds(keys)
    payload = baseline_floor(target, keys, folds)
    payload["cell_blur"] = run_blur_study(cell_maps(x), target, keys, folds)
    update_out(args.out, "d1", payload)
    print("D1 written")


def stage_d2(args):
    ds, keys, x, target, power = load_designs(args.data_dir)
    folds = _make_family_folds(keys)
    update_out(args.out, "d2", {"power_blur": run_blur_study(power_maps(power), target, keys, folds)})
    print("D2 written")


def stage_d3(args):
    ds, keys, x, target, power = load_designs(args.data_dir)
    folds = _make_family_folds(keys)
    update_out(args.out, "d3", run_ridge(x, power, target, keys, folds))
    print("D3 written")


def stage_d4(args):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ThermalDataset(args.data_dir, augment=True)
    eval_ds = ThermalDataset(args.data_dir, augment=False)
    folds = _make_family_folds(eval_ds.keys)
    existing = _load_json(args.out).get("d4", {"runs": []}) if os.path.exists(args.out) else {"runs": []}
    runs = existing["runs"]
    done = {(r["arch"], r["frac"], r["seed"], r["fold"]) for r in runs}
    wanted_folds = set(args.folds.split(",")) if args.folds else None
    t0 = time.time()
    for frac in args.fractions:
        for arch in args.archs:
            for seed in args.seeds:
                for fi, fold in enumerate(folds):
                    if wanted_folds and fold["name"] not in wanted_folds:
                        continue
                    if (arch, frac, seed, fold["name"]) in done:
                        continue
                    if time.time() - t0 > args.time_budget_s:
                        update_out(args.out, "d4", {"runs": runs})
                        print(f"Time budget reached; {len(runs)} d4 records saved")
                        return
                    t1 = time.time()
                    recs = run_wrapped(arch, train_ds, eval_ds, fold, fi, seed, frac, device, args.epochs)
                    for r in recs:
                        runs.append(
                            {
                                "arch": arch,
                                "frac": frac,
                                "seed": seed,
                                "fold": fold["name"],
                                "design": r["design"],
                                "heldout_mse": r["heldout_mse"],
                                "spatial_spearman": r["spatial_spearman"],
                                "train_mse_final": r["train_mse_final"],
                                "n_train": len(nested_train_keys([k for k in train_ds.keys if k not in set(fold["held_out"])], fi, seed, frac)),
                            }
                        )
                    print(f"{arch} frac={frac} seed={seed} fold={fold['name']} {time.time() - t1:.1f}s")
        update_out(args.out, "d4", {"runs": runs})
    update_out(args.out, "d4", {"runs": runs})
    print(f"D4 pass complete; {len(runs)} records")


def learning_curve(d4_runs, stored, archs=("unet32", "fno"), seeds=(0, 1, 2)):
    """arch -> frac -> {median, mean (over designs of seed-mean MSE), per_seed_median, seed_sd}."""
    out = {}
    for arch in archs:
        out[arch] = {}
        for frac in (*LC_FRACTIONS, 1.0):
            if frac == 1.0:
                runs = [r for r in stored if r["arch"] == arch and r["seed"] in seeds]
                recs = [{"design": r["design"], "seed": r["seed"], "heldout_mse": r["heldout_mse"]} for r in runs]
            else:
                recs = [r for r in d4_runs if r["arch"] == arch and r["frac"] == frac and r["seed"] in seeds]
            if not recs:
                continue
            by_design, by_seed = {}, {}
            for r in recs:
                by_design.setdefault(r["design"], []).append(r["heldout_mse"])
                by_seed.setdefault(r["seed"], {}).setdefault(r["design"], []).append(r["heldout_mse"])
            per_design = [float(np.mean(v)) for v in by_design.values()]
            seed_medians = [_median([float(np.mean(v)) for v in d.values()]) for d in by_seed.values()]
            out[arch][str(frac)] = {
                "n_designs": len(per_design),
                "n_seeds": len(by_seed),
                "median": _median(per_design),
                "mean": _mean(per_design),
                "seed_median_mean": float(np.mean(seed_medians)),
                "seed_median_sd": float(np.std(seed_medians)),
            }
    return out


def _design_meta(arch_sweep_path):
    return {d["key"]: d for d in _load_json(arch_sweep_path)["designs"]}


def anatomy(path, meta):
    mse_u = _per_design_means([r for r in stored_runs(path)], "unet32", "heldout_mse")
    mse_b = _per_design_means(stored_runs(path), "blur", "heldout_mse")
    train_u = _per_design_means(stored_runs(path), "unet32", "train_mse_final")
    keys = sorted(mse_u)
    delta = {k: mse_u[k] - mse_b[k] for k in keys}
    corr = {}
    for f in META_FIELDS:
        rho, p = stats.spearmanr([delta[k] for k in keys], [meta[k][f] for k in keys])
        corr[f] = {"rho": float(rho), "p": float(p)}
    by_pdk = {}
    for k in keys:
        by_pdk.setdefault(meta[k]["pdk"], []).append(delta[k])
    up = [delta[k] for k in keys if k in UPSAMPLED_KEYS_CORRECTED]
    nat = [delta[k] for k in keys if k not in UPSAMPLED_KEYS_CORRECTED]
    order = sorted(keys, key=lambda k: delta[k])

    def row(k):
        return {"key": k, "delta": delta[k], "unet32": mse_u[k], "blur": mse_b[k], **{f: meta[k][f] for f in META_FIELDS}, "pdk": meta[k]["pdk"]}

    return {
        "delta": delta,
        "spearman_vs_meta": corr,
        "mean_delta_by_pdk": {p: {"mean": float(np.mean(v)), "n": len(v)} for p, v in sorted(by_pdk.items())},
        "mean_delta_upsampled": {"mean": float(np.mean(up)), "n": len(up)},
        "mean_delta_native": {"mean": float(np.mean(nat)), "n": len(nat)},
        "unet_loses_most": [row(k) for k in order[::-1][:5]],
        "unet_wins_most": [row(k) for k in order[:5]],
        "gap": {k: {"train_mse_final": train_u[k], "heldout_mse": mse_u[k], "gap": mse_u[k] - train_u[k]} for k in keys},
    }


def stage_d5(args):
    meta = _design_meta(args.arch_sweep)
    update_out(args.out, "d5", {"bakeoff": anatomy(args.arch_sweep, meta), "readiness": anatomy(args.readiness, meta)})
    print("D5 written")


def _fmt(v, nd=5):
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def build_report(results, arch_path, ready_path):
    d1, d2, d3, d4, d5 = (results[k] for k in ("d1", "d2", "d3", "d4", "d5"))
    meta_n = 30
    rows = []

    def add(name, per_design):
        med, mean = summarise(per_design)
        rho_med, _ = summarise(per_design, "spatial_spearman")
        rows.append((name, med, mean, rho_med))

    add("const_mean", d1["const_mean"])
    add("mean_map", d1["mean_map"])
    cb = d1["cell_blur"]
    add("cell blur sigma=3 raw_max (= stored blur)", cb["sweep"]["raw_max"]["3.0"])
    add("cell blur sigma=3 calibrated", cb["sweep"]["calibrated"]["3.0"])
    for v in VARIANTS:
        add(f"cell blur {v}, train-selected sigma", cb["selected"][v])
    pb = d2["power_blur"]
    for v in VARIANTS:
        add(f"ORACLE power blur {v}, train-selected sigma", pb["selected"][v])
    for name in (f"ridge_inputs@alpha={HEADLINE_ALPHA:g}", f"ridge_inputs_plus_power@alpha={HEADLINE_ALPHA:g}"):
        add(("ORACLE " if "power" in name else "") + name, d3[name])
    for a in RIDGE_ALPHAS:
        if a != HEADLINE_ALPHA:
            for base in ("ridge_inputs", "ridge_inputs_plus_power"):
                add(f"{base}@alpha={a:g} (sensitivity)", d3[f"{base}@alpha={a:g}"])

    def stored_rows(path, tag):
        out = {}
        for arch in ("unet32", "fno", "blur"):
            mse = stored_per_design(path, arch)
            rho = stored_per_design(path, arch, "spatial_spearman")
            out[arch] = {k: {"heldout_mse": mse[k], "spatial_spearman": rho[k]} for k in mse}
            add(f"stored {arch} ({tag})", out[arch])
        return out

    bake = stored_rows(arch_path, "bake-off, 200 ep, 5 seeds")
    ready = stored_rows(ready_path, "readiness recipe, 100 ep, 5 seeds")

    m_unet = summarise(bake["unet32"])[0]
    m_oracle = summarise(pb["selected"]["calibrated"])[0]
    stored_all = stored_runs(arch_path)
    lc = learning_curve(d4["runs"], stored_all)
    lc_u = lc["unet32"]
    rules = None
    if "0.75" in lc_u and "1.0" in lc_u:
        rules = reading_rules(
            m_unet,
            m_oracle,
            lc_u["0.75"]["median"],
            lc_u["1.0"]["median"],
            summarise(d3[f"ridge_inputs@alpha={HEADLINE_ALPHA:g}"])[0],
            summarise(cb["sweep"]["calibrated"]["3.0"])[0],
            summarise(cb["sweep"]["raw_max"]["3.0"])[0],
        )

    wins = sum(
        cb["sweep"]["calibrated"]["3.0"][k]["heldout_mse"] < bake["unet32"][k]["heldout_mse"] for k in bake["unet32"]
    )

    L = ["# Thermal U-Net vs blur diagnostics: summary", ""]
    L += ["M = median held-out MSE over the 30 designs (9 leave-one-family-out folds). Stored torch arches use the per-design mean over seeds. Rows marked ORACLE use the true power_grid, unavailable at inference.", ""]
    L += ["| predictor | M (median MSE) | mean MSE | median Spearman |", "|---|---|---|---|"]
    for name, med, mean, rho in rows:
        L.append(f"| {name} | {_fmt(med)} | {_fmt(mean)} | {_fmt(rho, 3)} |")
    L += ["", "## Full blur sweeps (held-out median MSE by sigma)", "", "| source | variant | " + " | ".join(f"s={s:g}" for s in SIGMAS) + " |", "|---|---|" + "---|" * len(SIGMAS)]
    for src, study in (("cell density", cb), ("ORACLE power", pb)):
        for v in VARIANTS:
            L.append(f"| {src} | {v} | " + " | ".join(_fmt(summarise(study["sweep"][v][str(s)])[0]) for s in SIGMAS) + " |")
    L += ["", "## D4 learning curve (mean over seeds 0,1,2 of per-design MSE; 100% is the stored bake-off result)", "", "| arch | fraction | median MSE | mean MSE | seed-to-seed sd of the median |", "|---|---|---|---|---|"]
    for arch, by in lc.items():
        for frac, v in by.items():
            L.append(f"| {arch} | {float(frac):.2f} | {_fmt(v['median'])} | {_fmt(v['mean'])} | {_fmt(v['seed_median_sd'])} |")
    L += ["", "## D5 failure anatomy (delta = MSE(unet32) - MSE(blur); negative = U-Net better)", ""]
    for tag, a in (("bake-off", d5["bakeoff"]), ("readiness recipe", d5["readiness"])):
        L += [f"### {tag}", "", "Spearman of delta vs design metadata:", ""]
        for f, v in a["spearman_vs_meta"].items():
            L.append(f"- {f}: rho={v['rho']:.3f}, p={v['p']:.3f}")
        L.append("")
        L.append("Mean delta by PDK: " + "; ".join(f"{p} {v['mean']:+.4f} (n={v['n']})" for p, v in a["mean_delta_by_pdk"].items()))
        L.append("")
        L.append(f"Mean delta, coarse-grid (upsampled) n={a['mean_delta_upsampled']['n']}: {a['mean_delta_upsampled']['mean']:+.4f}; native n={a['mean_delta_native']['n']}: {a['mean_delta_native']['mean']:+.4f}")
        L += ["", "U-Net loses to blur by the most:", "", "| design | delta | unet32 | blur | ptp_c | contrast | corr_t_p | distinct |", "|---|---|---|---|---|---|---|---|"]
        for r in a["unet_loses_most"]:
            L.append(f"| {r['key']} | {r['delta']:+.4f} | {r['unet32']:.4f} | {r['blur']:.4f} | {r['ptp_c']:.3g} | {r['contrast']:.3g} | {r['corr_t_p']:.3f} | {r['distinct_values']} |")
        L += ["", "U-Net wins over blur by the most:", "", "| design | delta | unet32 | blur | ptp_c | contrast | corr_t_p | distinct |", "|---|---|---|---|---|---|---|---|"]
        for r in a["unet_wins_most"]:
            L.append(f"| {r['key']} | {r['delta']:+.4f} | {r['unet32']:.4f} | {r['blur']:.4f} | {r['ptp_c']:.3g} | {r['contrast']:.3g} | {r['corr_t_p']:.3f} | {r['distinct_values']} |")
        gaps = [v["gap"] for v in a["gap"].values()]
        tr = [v["train_mse_final"] for v in a["gap"].values()]
        ho = [v["heldout_mse"] for v in a["gap"].values()]
        L += ["", f"Train-vs-held-out gap (unet32, mean over seeds): mean train_mse_final {np.mean(tr):.4f}, mean held-out {np.mean(ho):.4f}, mean gap {np.mean(gaps):.4f}; designs with gap > 0: {sum(g > 0 for g in gaps)}/{len(gaps)}.", ""]
    L += ["## Reading rules", ""]
    if rules is None:
        L.append("D4 learning curve incomplete: R2 not evaluated.")
    else:
        L += [
            f"- R1: M_oracle={m_oracle:.5f}, M_unet32={m_unet:.5f}, ratio={rules['R1']['ratio']:.3f} -> {rules['R1']['label']}",
            f"- R2: unet32 median MSE 75%={lc_u['0.75']['median']:.5f} -> 100%={lc_u['1.0']['median']:.5f}, relative drop={rules['R2']['drop']:.3f} -> {rules['R2']['label']}",
            f"- R3: ridge_inputs M={summarise(d3[f'ridge_inputs@alpha={HEADLINE_ALPHA:g}'])[0]:.5f} vs unet32 {m_unet:.5f}, relative difference={rules['R3']['rel_diff']:+.3f} -> {rules['R3']['label']}",
            f"- R4: calibrated blur sigma=3 M={rules['R4']['m_calibrated_sigma3']:.5f}, raw blur sigma=3 M={rules['R4']['m_raw_sigma3']:.5f}, unet32 M={m_unet:.5f}; calibrated blur has lower MSE than unet32 on {wins}/{meta_n} designs -> {rules['R4']['label']}",
        ]
    return "\n".join(L) + "\n", rules, lc


def stage_report(args):
    results = _load_json(args.out)
    missing = [s for s in ("d1", "d2", "d3", "d4", "d5") if s not in results]
    if missing:
        raise SystemExit(f"Missing stages in {args.out}: {missing}")
    text, rules, lc = build_report(results, args.arch_sweep, args.readiness)
    with open(args.summary, "w") as f:
        f.write(text)
    update_out(args.out, "report", {"rules": rules, "learning_curve": lc})
    print(text)


STAGES = {
    "p1": stage_p1,
    "p2": stage_p2,
    "d1": stage_d1,
    "d2": stage_d2,
    "d3": stage_d3,
    "d4": stage_d4,
    "d5": stage_d5,
    "report": stage_report,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--arch-sweep", default=DEFAULT_ARCH)
    ap.add_argument("--readiness", default=DEFAULT_READY)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary", default=DEFAULT_SUMMARY)
    ap.add_argument("--archs", default="unet32,fno", type=lambda s: s.split(","))
    ap.add_argument("--seeds", default="0,1,2", type=lambda s: [int(v) for v in s.split(",")])
    ap.add_argument("--fractions", default=",".join(str(f) for f in LC_FRACTIONS), type=lambda s: [float(v) for v in s.split(",")])
    ap.add_argument("--folds", default=None, help="Comma-separated fold names (d4 only)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--time-budget-s", type=float, default=float("inf"), help="d4: stop cleanly (resumable) after this many seconds")
    args = ap.parse_args()
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
