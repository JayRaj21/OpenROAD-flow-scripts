"""
Generate synthetic feature/label .npz files for testing the congestion pipeline
without needing real ORFS runs or Docker.

Creates N_DESIGNS fake designs in the given output directory.

Usage:
  python3 generate_synthetic_data.py --out-dir ../data --n-designs 8 --grid 64
"""

import argparse
import os
import numpy as np


def make_features(grid: int, rng: np.random.Generator) -> dict:
    """Synthetic placement features — spatially correlated via Gaussian blur."""
    from scipy.ndimage import gaussian_filter

    cell = rng.random((grid, grid)).astype(np.float32)
    cell = gaussian_filter(cell, sigma=3)
    cell /= cell.max() + 1e-9

    # Macros: 1-3 rectangular blobs
    macro = np.zeros((grid, grid), dtype=np.float32)
    for _ in range(rng.integers(1, 4)):
        x0, y0 = rng.integers(0, grid - 10, size=2)
        w, h = rng.integers(4, 12, size=2)
        macro[y0 : y0 + h, x0 : x0 + w] = 1.0

    pin = rng.random((grid, grid)).astype(np.float32) * 0.4
    pin = gaussian_filter(pin, sigma=2)
    pin /= pin.max() + 1e-9

    fanout = rng.random((grid, grid)).astype(np.float32)
    fanout = gaussian_filter(fanout, sigma=4)
    fanout /= fanout.max() + 1e-9

    return {
        "cell_density": cell,
        "macro_density": macro,
        "pin_density": pin,
        "fanout_density": fanout,
    }


def make_irdrop_labels(features: dict, grid: int, rng: np.random.Generator) -> dict:
    """
    Synthetic IR-drop labels + PDN geometry channels, for testing the
    IR-drop pipeline without OpenROAD/PDNSim.

    stripe_density / via_density approximate a PDN grid: a handful of
    horizontal/vertical stripes at regular intervals, plus via hits at
    their intersections. current_density_proxy reuses cell_density as a
    stand-in for cell-type-weighted current draw. irdrop_map is synthesized
    as roughly proportional to current draw and inversely related to local
    PDN density (further from stripes/vias = higher drop), plus noise —
    same spirit as make_labels' congestion proxy below.
    """
    from scipy.ndimage import gaussian_filter

    stripe_density = np.zeros((grid, grid), dtype=np.float32)
    via_density = np.zeros((grid, grid), dtype=np.float32)
    stride = max(4, grid // 8)
    for i in range(0, grid, stride):
        stripe_density[i, :] += 1.0
        stripe_density[:, i] += 1.0
        via_density[i, i] += 1.0

    current_density_proxy = features["cell_density"].copy()

    pdn_proximity = gaussian_filter(stripe_density, sigma=2)
    pdn_proximity /= pdn_proximity.max() + 1e-9

    base = current_density_proxy * 0.02 * (1.0 - 0.6 * pdn_proximity)
    noise = rng.random((grid, grid)).astype(np.float32) * 0.002
    irdrop_map = np.clip(base + noise, 0.0, None).astype(np.float32)  # volts
    voltage_map = (1.1 - irdrop_map).astype(np.float32)

    return {
        "irdrop_map": irdrop_map,
        "voltage_map": voltage_map,
        "stripe_density": stripe_density,
        "via_density": via_density,
        "current_density_proxy": current_density_proxy,
    }


def make_labels(features: dict, grid: int, rng: np.random.Generator) -> dict:
    """Synthetic labels derived from features with added noise."""
    from scipy.ndimage import gaussian_filter

    # Congestion correlates with high cell density + macro proximity
    base = features["cell_density"] * 0.6 + features["macro_density"] * 0.4
    base = gaussian_filter(base, sigma=2)

    heatmap = np.zeros((10, grid, grid), dtype=np.float32)
    for layer in range(10):
        noise = rng.random((grid, grid)).astype(np.float32) * 0.15
        scale = 0.3 + layer * 0.05  # higher layers slightly less congested
        heatmap[layer] = np.clip(base * scale + noise, 0.0, 1.0)

    hotspot = (heatmap.max(axis=0) > 0.4).astype(np.uint8)
    score = np.float32(heatmap.mean())

    return {"heatmap": heatmap, "hotspot": hotspot, "score": score}


def generate(out_dir: str, n_designs: int, grid: int, seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    platforms = ["nangate45", "sky130hd"]
    design_names = [
        "aes",
        "gcd",
        "jpeg",
        "ibex",
        "swerv",
        "tinyRocket",
        "coyote",
        "ariane",
    ]

    created = []
    for i in range(n_designs):
        platform = platforms[i % len(platforms)]
        design = design_names[i % len(design_names)]
        tag = f"{platform}_{design}_{i}"  # suffix avoids collisions

        feat = make_features(grid, rng)
        label = make_labels(feat, grid, rng)
        irdrop_label = make_irdrop_labels(feat, grid, rng)

        feat_path = os.path.join(out_dir, f"{tag}_features.npz")
        label_path = os.path.join(out_dir, f"{tag}_labels.npz")
        irdrop_path = os.path.join(out_dir, f"{tag}_irdrop_labels.npz")
        np.savez(feat_path, **feat)
        np.savez(label_path, **label)
        np.savez(irdrop_path, **irdrop_label)
        created.append(tag)
        print(
            f"  [{i+1}/{n_designs}] {tag}  score={label['score']:.4f}  "
            f"hotspot={label['hotspot'].sum()}/{grid**2}  "
            f"irdrop={irdrop_label['irdrop_map'].max()*1e3:.2f}mV max"
        )

    print(f"\nCreated {len(created)} synthetic designs in {out_dir}")
    return created


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../data")
    ap.add_argument("--n-designs", type=int, default=8)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(args.out_dir, args.n_designs, args.grid, args.seed)
