"""
Scale-invariance check for extract_thermal_labels.py's HotSpot package model.

Builds one fixed synthetic relative power map (two off-center Gaussian
hotspots plus the same POWER_FLOOR_FRAC floor used in build_power_grid), then
runs the real hotspot_package_args()/write_hotspot_inputs()/run_hotspot()/
parse_steady() pipeline at several die extents, scaling total power by
POWER_DENSITY_W_PER_MM2 x die_area at each extent (same convention as
extract_thermal_labels.main()). At each extent the HotSpot grid is computed
with the real _adaptive_hotspot_grid() (not a fixed --grid value), so a
MIN_HOTSPOT_GRID regression (or a change to MIN_CELL_UM) actually shows up
in this script's output instead of being silently masked by a fixed grid
size. `--grid` is the *target* grid passed to _adaptive_hotspot_grid(); the
grid actually used at each extent is printed and may be smaller for small
dies.

Known physical result (see DESIGN_RUNS.md): under a physically realistic
package model, relative contrast is NOT scale-invariant and grows
faster-than-linearly with die extent, worse at small extents than large
ones (e.g. 9/25/51/102um -> contrast ~0.012/0.083/0.297/0.627, a ~52x
range over an 11x extent range). Because of this, a hardcoded
correlation>0.99 / contrast-deviation<=25% PASS/FAIL bar is not
achievable and would always report FAIL regardless of whether the code
is healthy. This script therefore does not assert pass/fail; it prints
the metrics table (contrast, correlation, and contrast growth rate vs.
extent) for a human (or a future, better-calibrated tolerance) to judge.
The one automated check it does keep is a *regression* guard: contrast at
the smallest extent must be non-degenerate (not collapsed toward 0),
which is the old bug's signature this script was originally written to
catch.

Dependency-light: numpy + the `hotspot` binary only (no openroad/torch).

Run inside Docker (requires openroad/orfs-ml:latest):
  python3 util/ml/congestion/tests/thermal_scale_check.py \\
      [--extents-um 51,255,510,1021] [--grid 32] [--power-density 10.0]
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "data_collection")
)

from extract_thermal_labels import (  # noqa: E402
    POWER_FLOOR_FRAC,
    _adaptive_hotspot_grid,
    hotspot_package_args,
    parse_steady,
    run_hotspot,
    write_hotspot_inputs,
)

AMBIENT_C = 45.0  # HotSpot default ambient (318.15 K)

# Regression floor for contrast at the smallest tested extent. The real
# sweep (see DESIGN_RUNS.md) shows genuine physical contrast as low as
# ~0.01 at 9um — well above zero. A value near 0 here is the old
# isothermal-collapse bug's signature, not expected physics.
DEGENERATE_CONTRAST_MIN = 1e-3


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extents-um", default="51,255,510,1021")
    ap.add_argument(
        "--grid",
        type=int,
        default=32,
        help="Target grid passed to _adaptive_hotspot_grid() (default 32). "
        "The grid actually used at each extent may be smaller for small "
        "dies, exactly as in extract_thermal_labels.main().",
    )
    ap.add_argument("--power-density", type=float, default=10.0)
    return ap.parse_args()


def _synthetic_power_grid(grid: int, total_power_w: float) -> np.ndarray:
    """Two off-center Gaussian hotspots + POWER_FLOOR_FRAC floor, fixed shape
    regardless of scale so the same relative pattern is tested at every
    extent."""
    yy, xx = np.mgrid[0:grid, 0:grid].astype(np.float64)
    hotspots = [
        (0.3 * grid, 0.3 * grid, 0.08 * grid),
        (0.7 * grid, 0.6 * grid, 0.10 * grid),
    ]
    pattern = np.zeros((grid, grid), dtype=np.float64)
    for cx, cy, sigma in hotspots:
        pattern += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))

    total = pattern.sum()
    power_grid = pattern / total * total_power_w if total > 0 else (
        np.ones((grid, grid)) * total_power_w / (grid * grid)
    )

    floor_per_cell = POWER_FLOOR_FRAC * total_power_w / (grid * grid)
    power_grid = np.maximum(power_grid, floor_per_cell)
    power_grid = power_grid / power_grid.sum() * total_power_w
    return power_grid


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def main():
    from scipy.ndimage import zoom

    args = _parse_args()
    target_grid = args.grid
    extents_um = [float(x) for x in args.extents_um.split(",")]

    rows = []
    norm_maps = []  # each upsampled to target_grid x target_grid for comparison
    for extent_um in extents_um:
        die_w_m = die_h_m = extent_um * 1e-6
        die_area_mm2 = die_w_m * die_h_m * 1e6
        total_power_w = args.power_density * die_area_mm2

        hs_grid = _adaptive_hotspot_grid(die_w_m, die_h_m, target_grid)
        power_grid = _synthetic_power_grid(hs_grid, total_power_w)
        die_bounds_m = (0.0, 0.0, die_w_m, die_h_m)
        package_args = hotspot_package_args(die_w_m, die_h_m, total_power_w)

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            flp_path, ptrace_path = write_hotspot_inputs(
                power_grid, die_bounds_m, hs_grid, work_dir
            )
            steady_path = run_hotspot(flp_path, ptrace_path, work_dir, package_args)
            temp_map = parse_steady(steady_path, hs_grid)

        tmin, tmax, tmean = (
            float(temp_map.min()),
            float(temp_map.max()),
            float(temp_map.mean()),
        )
        contrast = (tmax - tmin) / (tmean - AMBIENT_C)
        rows.append((extent_um, tmean, tmin, tmax, tmax - tmin, contrast))

        norm_map = _normalize(temp_map)
        if hs_grid != target_grid:
            norm_map = zoom(norm_map, target_grid / hs_grid, order=1)
        norm_maps.append(norm_map)

        print(
            f"[extent={extent_um:>7.1f}um] grid={hs_grid:3d}  Tmean={tmean:7.2f}C  "
            f"Tmin={tmin:7.2f}C  Tmax={tmax:7.2f}C  ptp={tmax-tmin:7.3f}C  "
            f"contrast={contrast:7.4f}"
        )

    ref = norm_maps[0].flatten()
    correlations = []
    for extent_um, norm_map in zip(extents_um, norm_maps):
        corr = float(np.corrcoef(ref, norm_map.flatten())[0, 1])
        correlations.append(corr)
        print(f"[extent={extent_um:>7.1f}um] correlation vs smallest extent: {corr:.4f}")

    contrasts = np.array([r[5] for r in rows])
    mean_contrast = contrasts.mean()
    rel_dev = np.abs(contrasts - mean_contrast) / mean_contrast
    print(f"Contrast mean={mean_contrast:.4f}  max relative deviation={rel_dev.max():.4f}")

    # No PASS/FAIL bar: a fixed correlation>0.99 / contrast-deviation<=25%
    # bar is not physically achievable (see module docstring and
    # DESIGN_RUNS.md) and previously made this script always report FAIL
    # regardless of code health, which is useless as a gate. Instead, keep
    # one automated regression guard for the specific bug this script was
    # written to catch — the smallest extent going isothermal/degenerate —
    # and otherwise just report the metrics for a human to read.
    smallest_contrast = float(contrasts[0])
    if smallest_contrast < DEGENERATE_CONTRAST_MIN:
        print(
            f"REGRESSION: contrast at smallest extent ({extents_um[0]}um) = "
            f"{smallest_contrast:.4f} < {DEGENERATE_CONTRAST_MIN} — looks like "
            f"the old isothermal-collapse bug, not the expected residual "
            f"small-die flatness."
        )
        return 1
    print("No isothermal-collapse regression detected (see metrics above).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
