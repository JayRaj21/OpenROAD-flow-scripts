"""
Scalar summaries of 64x64 thermal / cell-density maps for the placement loop.

All maps are 64x64 float32. Shape metrics are computed on the min-max
normalised map (m_hat in [0, 1]); a constant map has no range to normalise,
so normalize01 returns zeros and shape_metrics reports it as perfectly
uniform (p2m = top10_ratio = 1.0, haf_0p8 = 0.0).

Shape metrics (on m_hat, N = 4096 tiles):
  haf_0p8      fraction of tiles with m_hat >= 0.8
  p2m          max(m_hat) / mean(m_hat)
  top10_ratio  mean of the hottest 10% of tiles (ceil(0.1 N) tiles) / mean of all tiles

Absolute metrics (ground-truth thermal_map in degrees C only):
  ptp_c = max - min,  tmax_c = max,  tmean_c = mean

l1_distance(a, b) is the mean absolute difference of the two min-max
normalised maps (so it lies in [0, 1]). It measures how different two maps
are, not why: a change of wire-RC file between two placements moves it almost
as much as a large density change (see DESIGN_RUNS.md, 2026-09-20), so it
must be read against a nuisance reference.

Every library function raises ValueError on a map that is not finite, not
2-D, or not the expected shape (64x64 unless expected_shape is passed).

Sign convention: the optimisation target is LOWER top10_ratio (less
concentration); haf_0p8 is reported alongside.

CLI:
  python3 thermal_metrics.py --npz <file.npz> \\
      --key thermal_map|thermal_pred_norm|cell_density --json <out.json>
  python3 thermal_metrics.py --l1 <a.npz> <b.npz> --key cell_density
  python3 thermal_metrics.py --check-thermal-npz <thermal_labels.npz>
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from thermal_dataset import BLUR_SIGMA  # noqa: E402

TOP_FRAC = 0.10
HOT_THRESHOLD = 0.8
MAP_SHAPE = (64, 64)
THERMAL_KEYS = {"thermal_map", "power_grid"}
MIN_THERMAL_PTP_C = 1e-3


def _checked(m, name: str, expected_shape: tuple) -> np.ndarray:
    arr = np.asarray(m)
    if arr.shape != tuple(expected_shape):
        raise ValueError(f"{name}: expected shape {tuple(expected_shape)}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: contains NaN or Inf")
    return arr


def normalize01(m: np.ndarray, expected_shape: tuple = MAP_SHAPE) -> np.ndarray:
    m = _checked(m, "map", expected_shape).astype(np.float32)
    lo, hi = float(m.min()), float(m.max())
    if hi == lo:
        return np.zeros_like(m)
    return (m - lo) / (hi - lo)


def shape_metrics(m: np.ndarray, expected_shape: tuple = MAP_SHAPE) -> dict:
    m_hat = normalize01(m, expected_shape)
    if float(m_hat.max()) == 0.0:
        return {"haf_0p8": 0.0, "p2m": 1.0, "top10_ratio": 1.0}
    flat = np.sort(m_hat.ravel())[::-1]
    k = math.ceil(TOP_FRAC * flat.size)
    mean_all = float(flat.mean())
    return {
        "haf_0p8": float((m_hat >= HOT_THRESHOLD).mean()),
        "p2m": float(flat[0]) / mean_all,
        "top10_ratio": float(flat[:k].mean()) / mean_all,
    }


def absolute_metrics(m_celsius: np.ndarray, expected_shape: tuple = MAP_SHAPE) -> dict:
    m = _checked(m_celsius, "map", expected_shape).astype(np.float64)
    return {
        "ptp_c": float(m.max() - m.min()),
        "tmax_c": float(m.max()),
        "tmean_c": float(m.mean()),
    }


def blur_proxy(cell_density: np.ndarray, expected_shape: tuple = MAP_SHAPE) -> np.ndarray:
    m = _checked(cell_density, "cell_density", expected_shape).astype(np.float32)
    return gaussian_filter(m, sigma=BLUR_SIGMA)


def l1_distance(a: np.ndarray, b: np.ndarray, expected_shape: tuple = MAP_SHAPE) -> float:
    return float(np.abs(normalize01(a, expected_shape) - normalize01(b, expected_shape)).mean())


def check_thermal_npz(path: str) -> None:
    """Raise ValueError unless path is a usable thermal-label file."""
    try:
        f = np.load(path)
    except Exception as e:
        raise ValueError(f"{path}: cannot load ({type(e).__name__}: {e})") from e
    with f:
        keys = set(f.files)
        if keys != THERMAL_KEYS:
            raise ValueError(f"{path}: keys {sorted(keys)}, expected {sorted(THERMAL_KEYS)}")
        thermal = _checked(f["thermal_map"], f"{path}[thermal_map]", MAP_SHAPE)
        _checked(f["power_grid"], f"{path}[power_grid]", MAP_SHAPE)
    ptp = float(thermal.max() - thermal.min())
    if ptp <= MIN_THERMAL_PTP_C:
        raise ValueError(f"{path}[thermal_map]: flat map (ptp {ptp:.3g} C <= {MIN_THERMAL_PTP_C} C)")


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz")
    ap.add_argument("--key", choices=["thermal_map", "thermal_pred_norm", "cell_density"])
    ap.add_argument("--json")
    ap.add_argument("--l1", nargs=2, metavar=("A_NPZ", "B_NPZ"))
    ap.add_argument("--check-thermal-npz", metavar="NPZ")
    args = ap.parse_args()
    if args.check_thermal_npz:
        return args
    if args.l1:
        if not args.key:
            ap.error("--l1 requires --key")
    elif not (args.npz and args.key and args.json):
        ap.error("--npz, --key and --json are required (or use --l1 / --check-thermal-npz)")
    return args


def main():
    args = _parse_args()

    if args.check_thermal_npz:
        try:
            check_thermal_npz(args.check_thermal_npz)
        except (ValueError, TypeError) as e:
            sys.exit(f"BAD {e}")
        print(f"OK {args.check_thermal_npz}")
        return

    if args.l1:
        maps = []
        for path in args.l1:
            with np.load(path) as f:
                maps.append(f[args.key])
        print(f"{l1_distance(maps[0], maps[1]):.6f}")
        return

    with np.load(args.npz) as f:
        m = f[args.key]

    result = {"key": args.key, **shape_metrics(m)}
    if args.key == "thermal_map":
        result.update(absolute_metrics(m))

    with open(args.json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
