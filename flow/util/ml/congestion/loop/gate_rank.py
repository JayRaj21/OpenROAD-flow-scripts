"""
Stage B go/no-go gate: can the thermal U-Net surrogate rank placement-density
variants of one design in the same order as HotSpot?

The pass criteria are pre-registered in DESIGN_RUNS.md (2026-09-25) and fixed
here as constants; do not change them after results have been seen.

Usage (from flow/):
  python3 util/ml/congestion/loop/gate_rank.py \\
      --variant-dir util/ml/congestion/experiments/thermal_loop/data \\
      --designs sky130hd/riscv32i,gf180/riscv32i,sky130hs/aes,sky130hd/ibex \\
      --checkpoint-map riscv32i=<ckpt>,aes=<ckpt>,ibex=<ckpt> \\
      --out util/ml/congestion/experiments/thermal_loop/gate.json \\
      [--markdown gate.md] [--min-points 4] [--all-variants] [--data-dir <base data>] \\
      [--expect-epochs N] [--expect-seed S]

Exit status: 0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE (verdict written; includes
a smoke-test checkpoint trained for fewer than MIN_GATE_EPOCHS epochs),
3 = ERROR with no verdict (usage error, missing or corrupt checkpoint, refused
leaking checkpoint, checkpoint whose epochs or seed differ from --expect-epochs
or --expect-seed, unreadable data, inference failure, failure to write the
output, missing dependency).

Once the arguments are valid and before anything is computed, a pre-existing
--out and --markdown are deleted, so an ERROR never leaves an old verdict
beside a fresh error. The two outputs are written to temporary files and
renamed, so they appear together or not at all. In a smoke-test result the
per-criterion g1/g2/g3 read "INCONCLUSIVE (smoke test)"; the numbers computed
are kept under verdict.smoke_test_underlying.

Structure: scoring (score_design, evaluate_gate) is a pure function of
per-variant metric rows; file loading (discover_variants, build_rows) and
model inference (predict_top10) are separate.
"""

import argparse
import json
import os
import re
import sys
import tempfile

EXIT_PASS, EXIT_FAIL, EXIT_INCONCLUSIVE, EXIT_ERROR = 0, 1, 2, 3

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "inference"))
sys.path.insert(0, os.path.join(HERE, "..", "training"))

try:
    import numpy as np
    import torch
    from scipy import stats

    from predict_thermal import load_features
    from thermal_metrics import (
        absolute_metrics,
        blur_proxy,
        check_thermal_npz,
        shape_metrics,
    )
    from laplacian_sweep import _parse_key
    from train_lodo import data_dir_keys, family_of, sha256_file
    from unet import CongestionUNet
except Exception as _e:  # noqa: BLE001 - a broken environment is an ERROR (exit 3), never a verdict
    print(f"ERROR: cannot import gate_rank dependencies: {type(_e).__name__}: {_e}", file=sys.stderr)
    sys.exit(EXIT_ERROR)

GRID_TAGS = ["dn_005", "dn_015", "dn_030", "dn_045", "dn_060", "dn_075"]
MIN_DESIGNS = 3
G1_PTP_SPREAD_FRAC = 0.20
G1_TOP10_SPREAD = 0.05
G1_RHO = 0.89
G1_MIN_DESIGNS = 3
G2_MEDIAN_RHO = 0.7
G3_MEDIAN_RHO = 0.6
G3_MIN_RHO = 0.3
G3_MIN_RHO_DESIGNS = 3
G3_PROXY_MARGIN = 0.15
G3_TOP_K = 2
G3_TOP1_DESIGNS = 3
G3_RERUN_BAND = 0.1
MIN_GATE_EPOCHS = 100
# Spearman rho and differences of floats land a few ulps off the exact value
# (0.9 comes back as 0.8999999999999998), so a value exactly on a threshold
# would fail a plain >=. Comparisons pass values within EPS of the threshold.
# Without exact rank ties every rho and rho difference is a small-denominator
# rational, so EPS cannot flip a comparison that is genuinely below its
# threshold; only exactly tied metrics could, which float map metrics never are.
EPS = 1e-9
DEFAULT_MIN_POINTS = 4
SMOKE_LABEL = "INCONCLUSIVE (smoke test)"

THRESHOLDS = {
    "grid_tags": GRID_TAGS,
    "min_designs": MIN_DESIGNS,
    "g1_ptp_spread_frac": G1_PTP_SPREAD_FRAC,
    "g1_top10_spread": G1_TOP10_SPREAD,
    "g1_abs_rho": G1_RHO,
    "g1_min_designs": G1_MIN_DESIGNS,
    "g2_median_rho": G2_MEDIAN_RHO,
    "g3_median_rho": G3_MEDIAN_RHO,
    "g3_min_rho": G3_MIN_RHO,
    "g3_min_rho_designs": G3_MIN_RHO_DESIGNS,
    "g3_proxy_margin": G3_PROXY_MARGIN,
    "g3_top_k": G3_TOP_K,
    "g3_top1_designs": G3_TOP1_DESIGNS,
    "g3_rerun_band": G3_RERUN_BAND,
}


class LeakageError(RuntimeError):
    pass


class CheckpointMismatchError(RuntimeError):
    pass


class UsageParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"ERROR: {message}", file=sys.stderr)
        sys.exit(EXIT_ERROR)


def ge(value: float, threshold: float) -> bool:
    return value >= threshold - EPS


def le(value: float, threshold: float) -> bool:
    return value <= threshold + EPS


def spearman(a, b) -> float:
    """Spearman rho; a constant input has no rank order and gives 0.0."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(stats.spearmanr(a, b)[0])


def variant_metrics(thermal_map, cell_density, pred_map) -> dict:
    """Metrics of one variant from its loaded arrays and the model's prediction."""
    truth = shape_metrics(thermal_map)
    return {
        "truth_top10": truth["top10_ratio"],
        "truth_haf_0p8": truth["haf_0p8"],
        "truth_p2m": truth["p2m"],
        **{f"truth_{k}": v for k, v in absolute_metrics(thermal_map).items()},
        "pred_top10": shape_metrics(pred_map)["top10_ratio"],
        "proxy_top10": shape_metrics(blur_proxy(cell_density))["top10_ratio"],
    }


def score_design(rows: list[dict]) -> dict:
    """Score one design. rows: dicts with addon, truth_top10, truth_ptp_c,
    pred_top10, proxy_top10 (any order)."""
    rows = sorted(rows, key=lambda r: r["addon"])
    addon = [r["addon"] for r in rows]
    truth = [r["truth_top10"] for r in rows]
    ptp = [r["truth_ptp_c"] for r in rows]
    pred = [r["pred_top10"] for r in rows]
    proxy = [r["proxy_top10"] for r in rows]

    ptp_spread = float(max(ptp) - min(ptp))
    ptp_median = float(np.median(ptp))
    top10_spread = float(max(truth) - min(truth))
    rho_addon_top10 = spearman(addon, truth)
    rho_addon_ptp = spearman(addon, ptp)
    g1 = {
        "ptp_spread_c": ptp_spread,
        "ptp_median_c": ptp_median,
        "a_ptp_spread": bool(ge(ptp_spread, G1_PTP_SPREAD_FRAC * ptp_median)),
        "top10_spread": top10_spread,
        "b_top10_spread": bool(ge(top10_spread, G1_TOP10_SPREAD)),
        "rho_addon_top10": rho_addon_top10,
        "rho_addon_ptp": rho_addon_ptp,
        "c_monotonic": bool(ge(abs(rho_addon_top10), G1_RHO) and ge(abs(rho_addon_ptp), G1_RHO)),
    }
    g1["pass"] = g1["a_ptp_spread"] and g1["b_top10_spread"] and g1["c_monotonic"]

    pick = int(np.argmin(pred))
    pick_truth_rank = int(sum(1 for t in truth if t < truth[pick] - EPS)) + 1
    return {
        "n_variants": len(rows),
        "g1": g1,
        "g2_rho_top10_ptp": spearman(truth, ptp),
        "g3_rho_surrogate": spearman(pred, truth),
        "g3_rho_proxy": spearman(proxy, truth),
        "g3_top1_tag": rows[pick].get("tag"),
        "g3_top1_truth_rank": pick_truth_rank,
        "g3_top1_ok": bool(pick_truth_rank <= G3_TOP_K),
    }


def evaluate_gate(scored: dict) -> dict:
    """scored: {design: score_design(...) result} for the usable designs."""
    names = sorted(scored)
    notes = []
    if len(names) < MIN_DESIGNS:
        notes.append(f"only {len(names)} usable design(s); at least {MIN_DESIGNS} are needed for a verdict")
        return {"g1": "INCONCLUSIVE", "g2": "INCONCLUSIVE", "g3": "INCONCLUSIVE",
                "overall": "INCONCLUSIVE", "notes": notes, "detail": {}}

    g1_passes = sum(scored[d]["g1"]["pass"] for d in names)
    g1 = g1_passes >= G1_MIN_DESIGNS
    for d in names:
        if not scored[d]["g1"]["pass"]:
            failed = [k for k in ("a_ptp_spread", "b_top10_spread", "c_monotonic") if not scored[d]["g1"][k]]
            notes.append(f"G1 fails for {d}: {', '.join(failed)}")

    g2_median = float(np.median([scored[d]["g2_rho_top10_ptp"] for d in names]))
    g2 = ge(g2_median, G2_MEDIAN_RHO)

    sur = [scored[d]["g3_rho_surrogate"] for d in names]
    prx = [scored[d]["g3_rho_proxy"] for d in names]
    sur_median = float(np.median(sur))
    prx_median = float(np.median(prx))
    n_rho_ok = sum(ge(r, G3_MIN_RHO) for r in sur)
    c1 = ge(sur_median, G3_MEDIAN_RHO) and n_rho_ok >= G3_MIN_RHO_DESIGNS
    c2 = ge(sur_median - prx_median, G3_PROXY_MARGIN)
    c3 = sum(scored[d]["g3_top1_ok"] for d in names) >= G3_TOP1_DESIGNS
    g3 = c1 and c2 and c3
    if not c1:
        notes.append(f"G3(1) fails: median surrogate rho {sur_median:.3f}, designs with rho >= {G3_MIN_RHO}: {n_rho_ok}")
    if not c2:
        notes.append(f"G3(2) fails: surrogate median {sur_median:.3f} minus proxy median {prx_median:.3f} is below {G3_PROXY_MARGIN}")
    if not c3:
        notes.append(f"G3(3) fails: top-1 pick within best {G3_TOP_K} on {sum(scored[d]['g3_top1_ok'] for d in names)} designs, need {G3_TOP1_DESIGNS}")
    rerun = le(abs(sur_median - G3_MEDIAN_RHO), G3_RERUN_BAND)
    if rerun:
        notes.append(
            f"G3 median rho {sur_median:.3f} is within +/-{G3_RERUN_BAND} of {G3_MEDIAN_RHO}: "
            "a multi-seed rerun (seeds 0,1,2, mean) is warranted"
        )

    overall = "PASS" if (g1 and g2 and g3) else "FAIL"
    if overall == "FAIL":
        notes.append("failed: " + ", ".join(n for n, ok in (("G1", g1), ("G2", g2), ("G3", g3)) if not ok))
    label = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    return {
        "g1": label(g1), "g2": label(g2), "g3": label(g3),
        "overall": overall, "notes": notes,
        "detail": {
            "g1_designs_passing": g1_passes,
            "g2_median_rho": g2_median,
            "g3_median_surrogate_rho": sur_median,
            "g3_median_proxy_rho": prx_median,
            "g3_designs_rho_ge_min": int(n_rho_ok),
            "g3_top1_designs_ok": int(sum(scored[d]["g3_top1_ok"] for d in names)),
            "g3_rerun_multiseed_warranted": bool(rerun),
        },
    }


def check_checkpoint(sidecar: dict, ckpt: str, design: str, data_dir: str) -> None:
    """Refuse a checkpoint that was trained on the design (or its family)."""
    pdk, name = design.split("/")
    fam = family_of(name)
    key = f"{pdk}_{name}_base"
    if sidecar.get("holdout_design") != fam:
        raise LeakageError(
            f"{design}: checkpoint holdout family is '{sidecar.get('holdout_design')}', expected '{fam}'"
        )
    for field in ("pt_sha256", "epochs", "seed", "batch_size", "excluded_keys", "train_keys", "val_keys"):
        if field not in sidecar:
            raise LeakageError(f"{ckpt}.json has no '{field}'; retrain it with train_lodo.py")
    actual = sha256_file(ckpt)
    if actual != sidecar["pt_sha256"]:
        raise LeakageError(f"{ckpt} does not match its sidecar (sha256 {actual[:12]} != {sidecar['pt_sha256'][:12]}); it is not the checkpoint the sidecar describes")
    if key not in sidecar["excluded_keys"]:
        raise LeakageError(f"{design}: {key} is not among the checkpoint's excluded keys; it may have been trained on")
    family_keys = [k for k in data_dir_keys(data_dir) if family_of(_parse_key(k)[1]) == fam]
    missing = sorted(set(family_keys) - set(sidecar["excluded_keys"]))
    if missing:
        raise LeakageError(f"{design}: family '{fam}' keys present in {data_dir} but not excluded by the checkpoint: {', '.join(missing)}")
    trained = sorted(k for k in sidecar["train_keys"] + sidecar["val_keys"] if family_of(_parse_key(k)[1]) == fam)
    if trained:
        raise LeakageError(f"{design}: checkpoint was trained or validated on family '{fam}' keys: {', '.join(trained)}")


def load_checkpoint_sidecar(ckpt: str) -> dict:
    path = ckpt + ".json"
    if not os.path.isfile(ckpt) or not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint {ckpt} or sidecar {path} not found")
    with open(path) as f:
        return json.load(f)


def discover_variants(variant_dir: str, design: str, all_variants: bool) -> tuple[list[dict], list[str], list[dict]]:
    """Return (candidates, ignored off-grid tags, missing-grid records).

    A candidate has tag, addon and the two file paths (either may be None)."""
    pdk, name = design.split("/")
    prefix = f"{pdk}_{name}_"
    pat = re.compile(r"^" + re.escape(prefix) + r"(dn_(\d{3})(?:_p(\d+))?)_(features|thermal_labels)\.npz$")
    found: dict[str, dict] = {}
    for fn in sorted(os.listdir(variant_dir)):
        m = pat.match(fn)
        if not m:
            continue
        tag, pct, pad, kind = m.groups()
        rec = found.setdefault(tag, {"tag": tag, "addon": int(pct) / 100.0, "pad": pad, "features": None, "thermal": None})
        rec["features" if kind == "features" else "thermal"] = os.path.join(variant_dir, fn)
    candidates, ignored = [], []
    for tag in sorted(found):
        if all_variants or (tag in GRID_TAGS):
            candidates.append(found[tag])
        else:
            ignored.append(tag)
    missing = [{"tag": t, "reason": "no files found"} for t in GRID_TAGS if t not in found and not all_variants]
    return candidates, ignored, missing


def load_model(ckpt: str, device):
    model = CongestionUNet(in_channels=5, base_features=32, num_heatmap_layers=1).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def predict_map(model, features_path: str, device) -> np.ndarray:
    x = load_features(features_path).to(device)
    with torch.no_grad():
        return model(x).heatmap[0, 0].cpu().numpy().astype(np.float32)


def build_rows(candidates: list[dict], model, device) -> tuple[list[dict], list[dict]]:
    """Load, validate and score-prepare each candidate; returns (rows, skipped)."""
    rows, skipped = [], []
    for c in candidates:
        if c["features"] is None or c["thermal"] is None:
            missing = "features" if c["features"] is None else "thermal_labels"
            skipped.append({"tag": c["tag"], "reason": f"missing {missing} file"})
            continue
        try:
            check_thermal_npz(c["thermal"])
            with np.load(c["thermal"]) as t:
                thermal = t["thermal_map"]
            with np.load(c["features"]) as f:
                cell = f["cell_density"]
            pred = predict_map(model, c["features"], device)
            rows.append({"tag": c["tag"], "addon": c["addon"], **variant_metrics(thermal, cell, pred)})
        except (ValueError, KeyError, OSError) as e:
            skipped.append({"tag": c["tag"], "reason": f"extraction unusable: {e}"})
    return rows, skipped


def check_expected(sidecar: dict, ckpt: str, expect_epochs, expect_seed) -> None:
    for name, want in (("epochs", expect_epochs), ("seed", expect_seed)):
        if want is not None and sidecar.get(name) != want:
            raise CheckpointMismatchError(
                f"{ckpt}: sidecar {name} is {sidecar.get(name)!r}, expected {want!r}; "
                "it is not the checkpoint this run asked for (stale or from another run)"
            )


def run_gate(designs, ckpt_map, variant_dir, min_points, all_variants, device, data_dir,
             expect_epochs=None, expect_seed=None) -> dict:
    results = {"designs": {}, "excluded": {}, "checkpoints": {},
               "expected": {"epochs": expect_epochs, "seed": expect_seed}}
    scored = {}
    models = {}
    for design in designs:
        name = design.split("/")[1]
        fam = family_of(name)
        if fam not in ckpt_map:
            raise FileNotFoundError(f"no checkpoint given for family '{fam}' (needed by {design})")
        sidecar = load_checkpoint_sidecar(ckpt_map[fam])
        check_checkpoint(sidecar, ckpt_map[fam], design, data_dir)
        check_expected(sidecar, ckpt_map[fam], expect_epochs, expect_seed)
        results["checkpoints"][fam] = {
            "path": ckpt_map[fam], "sha256": sidecar["pt_sha256"], "epochs": sidecar["epochs"],
            "seed": sidecar["seed"], "batch_size": sidecar["batch_size"],
        }
        if fam not in models:
            models[fam] = load_model(ckpt_map[fam], device)
        candidates, ignored, missing = discover_variants(variant_dir, design, all_variants)
        rows, skipped = build_rows(candidates, models[fam], device)
        entry = {
            "checkpoint": ckpt_map[fam],
            "variants": rows,
            "skipped_variants": skipped + missing,
            "ignored_off_grid_tags": ignored,
        }
        if len(rows) < min_points:
            reason = f"only {len(rows)} usable variant(s), need {min_points}"
            entry["excluded_reason"] = reason
            results["excluded"][design] = reason
        else:
            entry["score"] = score_design(rows)
            scored[design] = entry["score"]
        results["designs"][design] = entry
    results["verdict"] = evaluate_gate(scored)
    smoke = sorted(f for f, c in results["checkpoints"].items() if c["epochs"] < MIN_GATE_EPOCHS)
    results["smoke_test"] = bool(smoke)
    results["min_gate_epochs"] = MIN_GATE_EPOCHS
    if smoke:
        v = results["verdict"]
        v["smoke_test_underlying"] = {k: v[k] for k in ("g1", "g2", "g3")}
        v["overall"] = "INCONCLUSIVE"
        for k in ("g1", "g2", "g3"):
            v[k] = SMOKE_LABEL
        results["verdict"]["notes"].append(
            f"SMOKE TEST: checkpoint(s) for {', '.join(smoke)} trained for fewer than {MIN_GATE_EPOCHS} epochs; "
            "this is not a gate verdict"
        )
    results["verdict"]["notes"] += [f"{d} excluded: {r}" for d, r in results["excluded"].items()]
    results["thresholds"] = THRESHOLDS
    return results


def to_markdown(results: dict) -> str:
    out = ["| Design | Tag | Add-on | truth top10 | truth ptp (C) | surrogate top10 | proxy top10 |", "|---|---|---|---|---|---|---|"]
    for design, e in results["designs"].items():
        for r in sorted(e["variants"], key=lambda r: r["addon"]):
            out.append(f"| {design} | {r['tag']} | {r['addon']:.2f} | {r['truth_top10']:.3f} | {r['truth_ptp_c']:.2f} | {r['pred_top10']:.3f} | {r['proxy_top10']:.3f} |")
    out += ["", "| Design | usable | G1 | spread ptp (C) | spread top10 | rho(addon,top10) | rho(addon,ptp) | G2 rho | G3 surrogate rho | G3 proxy rho | top-1 truth rank |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for design, e in results["designs"].items():
        s = e.get("score")
        if s is None:
            out.append(f"| {design} | {len(e['variants'])} | excluded: {e['excluded_reason']} | | | | | | | | |")
            continue
        g = s["g1"]
        g1_cell = SMOKE_LABEL if results["smoke_test"] else ("PASS" if g["pass"] else "FAIL")
        out.append(f"| {design} | {s['n_variants']} | {g1_cell} | {g['ptp_spread_c']:.2f} | {g['top10_spread']:.3f} | {g['rho_addon_top10']:.3f} | {g['rho_addon_ptp']:.3f} | {s['g2_rho_top10_ptp']:.3f} | {s['g3_rho_surrogate']:.3f} | {s['g3_rho_proxy']:.3f} | {s['g3_top1_truth_rank']} |")
    v = results["verdict"]
    if results["smoke_test"]:
        out += ["", f"SMOKE TEST: not a gate verdict (checkpoint trained for fewer than {MIN_GATE_EPOCHS} epochs)"]
    out += ["", f"Verdict: G1 {v['g1']}, G2 {v['g2']}, G3 {v['g3']}, overall {v['overall']}"]
    out += [f"- {n}" for n in v["notes"]]
    skipped = [(d, s) for d, e in results["designs"].items() for s in e["skipped_variants"]]
    if skipped:
        out += ["", "Skipped variants:"] + [f"- {d} {s['tag']}: {s['reason']}" for d, s in skipped]
    return "\n".join(out) + "\n"


def parse_ckpt_map(text: str) -> dict:
    out = {}
    for item in text.split(","):
        fam, sep, path = item.partition("=")
        if not sep or not fam or not path:
            raise argparse.ArgumentTypeError(f"bad --checkpoint-map entry '{item}', expected family=path")
        out[fam] = path
    return out


def write_outputs(results: dict, md: str, out: str, markdown) -> None:
    """Write the JSON and markdown through temporary files so both appear or neither does."""
    targets = [(out, json.dumps(results, indent=2))]
    if markdown:
        targets.append((markdown, md))
    tmps = []
    try:
        for path, text in targets:
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), prefix=".gate_", suffix=".tmp")
            tmps.append((tmp, path))
            with os.fdopen(fd, "w") as f:
                f.write(text)
        for tmp, path in reversed(tmps):
            os.replace(tmp, path)
    except BaseException:
        for tmp, path in tmps:
            for leftover in (tmp, path):
                if os.path.lexists(leftover):
                    os.remove(leftover)
        raise


def main():
    ap = UsageParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant-dir", required=True)
    ap.add_argument("--designs", required=True)
    ap.add_argument("--checkpoint-map", required=True, type=parse_ckpt_map)
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "..", "data"), help="base training data, used to check the leakage guard")
    ap.add_argument("--min-points", type=int, default=DEFAULT_MIN_POINTS)
    ap.add_argument("--all-variants", action="store_true", help="include tags off the pre-registered grid (exploratory)")
    ap.add_argument("--expect-epochs", type=int, help="refuse any checkpoint whose sidecar epochs differ")
    ap.add_argument("--expect-seed", type=int, help="refuse any checkpoint whose sidecar seed differs")
    args = ap.parse_args()

    designs = args.designs.split(",")
    for d in designs:
        if d.count("/") != 1:
            ap.error(f"design '{d}' must be <pdk>/<design>")

    try:
        for stale in (args.out, args.markdown):
            if stale and os.path.lexists(stale):
                os.remove(stale)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = run_gate(designs, args.checkpoint_map, args.variant_dir, args.min_points, args.all_variants,
                           device, args.data_dir, args.expect_epochs, args.expect_seed)
        md = to_markdown(results)
        write_outputs(results, md, args.out, args.markdown)
    except Exception as e:  # noqa: BLE001 - every failure is reported as ERROR (exit 3), never as a verdict
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(EXIT_ERROR)
    print(md)
    sys.exit({"PASS": EXIT_PASS, "FAIL": EXIT_FAIL}.get(results["verdict"]["overall"], EXIT_INCONCLUSIVE))


if __name__ == "__main__":
    main()
