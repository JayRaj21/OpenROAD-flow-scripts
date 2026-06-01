"""
Extract a 2D cell-density grid from a post-placement ODB file.

Run inside the ORFS Docker container via:
    openroad -python extract_placement.py \
        --odb /work/results/nangate45/gcd/base/3_5_place_dp.odb \
        --out /work/ml/data/nangate45_gcd_placement.npy \
        --grid 64

Outputs a (grid_size, grid_size) float32 numpy array where each cell
contains the normalised count of standard-cell instances in that bin.
"""

import argparse
import sys
import numpy as np

try:
    import openroad
    from openroad import Tech, Design
    HAS_OPENROAD = True
except ImportError:
    HAS_OPENROAD = False


def extract_placement_grid(odb_path: str, grid_size: int = 64) -> np.ndarray:
    if not HAS_OPENROAD:
        raise RuntimeError(
            "openroad Python module not found. "
            "Run this script via: openroad -python extract_placement.py ..."
        )

    tech = Tech()
    design = Design(tech)
    design.readDb(odb_path)
    block = design.getBlock()

    core = block.getCoreArea()
    x_min, y_min = core.xMin(), core.yMin()
    x_max, y_max = core.xMax(), core.yMax()
    core_w = x_max - x_min
    core_h = y_max - y_min

    grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    for inst in block.getInsts():
        if inst.isFixed():
            continue
        loc = inst.getLocation()
        x, y = loc[0], loc[1]

        col = int((x - x_min) / core_w * grid_size)
        row = int((y - y_min) / core_h * grid_size)
        col = min(col, grid_size - 1)
        row = min(row, grid_size - 1)
        grid[row, col] += 1.0

    max_val = grid.max()
    if max_val > 0:
        grid /= max_val

    return grid


def main():
    parser = argparse.ArgumentParser(description="Extract placement density grid from ODB")
    parser.add_argument("--odb", required=True, help="Path to post-placement ODB file")
    parser.add_argument("--out", required=True, help="Output .npy file path")
    parser.add_argument("--grid", type=int, default=64, help="Grid resolution (default: 64)")
    args = parser.parse_args()

    grid = extract_placement_grid(args.odb, args.grid)
    np.save(args.out, grid)
    print(f"Saved placement grid {grid.shape} -> {args.out}")
    print(f"  Non-zero bins: {(grid > 0).sum()}, max density: {grid.max():.4f}")


if __name__ == "__main__":
    main()
