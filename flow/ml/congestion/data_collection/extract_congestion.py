"""
Extract per-layer routing congestion from an ORFS global-route log file.

Run from the host (no OpenROAD needed):
    python3 extract_congestion.py \
        --log /work/logs/nangate45/gcd/base/5_1_grt.log \
        --out /work/ml/data/nangate45_gcd_congestion.npy

Outputs a (num_layers,) float32 numpy array of usage fractions (0.0–1.0+)
for each routing layer (metal1 through metal10).

A per-gcell 2D congestion map requires the OpenROAD Python API and is
extracted separately by extract_congestion_map.py once the ODB is available.
"""

import argparse
import re
import numpy as np

LAYER_ORDER = [
    "metal1", "metal2", "metal3", "metal4", "metal5",
    "metal6", "metal7", "metal8", "metal9", "metal10",
]

# Matches lines like:
# metal2            1326            31            2.34%             0 /  0 /  0
_ROW_RE = re.compile(
    r"^(metal\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%"
)


def parse_congestion_report(log_path: str) -> np.ndarray:
    """
    Parse the GRT congestion table from an ORFS 5_1_grt.log file.
    Returns a float32 array of shape (10,) — one usage fraction per layer.
    Missing layers default to 0.0.
    """
    usage = {layer: 0.0 for layer in LAYER_ORDER}
    in_table = False

    with open(log_path) as f:
        for line in f:
            if "Final congestion report" in line:
                in_table = True
                continue
            if not in_table:
                continue
            m = _ROW_RE.match(line.strip())
            if m:
                layer = m.group(1)
                pct = float(m.group(4)) / 100.0
                if layer in usage:
                    usage[layer] = pct
            elif in_table and line.strip().startswith("---") and any(
                v > 0 for v in usage.values()
            ):
                break

    return np.array([usage[l] for l in LAYER_ORDER], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract congestion vector from GRT log")
    parser.add_argument("--log", required=True, help="Path to 5_1_grt.log")
    parser.add_argument("--out", required=True, help="Output .npy file path")
    args = parser.parse_args()

    vec = parse_congestion_report(args.log)
    np.save(args.out, vec)
    print(f"Saved congestion vector {vec.shape} -> {args.out}")
    for layer, val in zip(LAYER_ORDER, vec):
        print(f"  {layer:8s}: {val*100:.2f}%")


if __name__ == "__main__":
    main()
