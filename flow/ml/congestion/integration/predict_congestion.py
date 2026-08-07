"""
ORFS integration script: extract placement from a placed ODB and run
congestion prediction. Called from orfs_hook.tcl via python3.

Environment variables (set by the Makefile / hook):
    RESULTS_DIR          -- e.g. /work/results/nangate45/gcd/base
    ML_ROOT              -- e.g. /work/ml  (default: inferred from script location)
    GRID_SIZE            -- grid resolution (default 64)
    CONGESTION_THRESHOLD -- fraction above which to warn (default 0.5)

Writes:
    $RESULTS_DIR/placement_grid.npy
    $RESULTS_DIR/predicted_congestion.npy
    $RESULTS_DIR/predicted_congestion.png  (if matplotlib available)
"""

import os
import subprocess
import sys
import numpy as np

ML_ROOT     = os.environ.get("ML_ROOT", os.path.normpath(
                  os.path.join(os.path.dirname(__file__), "../..")))
RESULTS_DIR = os.environ.get("RESULTS_DIR", ".")
GRID_SIZE   = int(os.environ.get("GRID_SIZE", "64"))
THRESHOLD   = float(os.environ.get("CONGESTION_THRESHOLD", "0.5"))
CHECKPOINT  = os.path.join(ML_ROOT, "congestion", "model", "checkpoints", "best.pt")
ODB_PATH    = os.path.join(RESULTS_DIR, "3_5_place_dp.odb")
PLACEMENT_NPY  = os.path.join(RESULTS_DIR, "placement_grid.npy")
CONGESTION_NPY = os.path.join(RESULTS_DIR, "predicted_congestion.npy")
CONGESTION_IMG = os.path.join(RESULTS_DIR, "predicted_congestion.png")

LAYER_NAMES = [
    "metal1", "metal2", "metal3", "metal4", "metal5",
    "metal6", "metal7", "metal8", "metal9", "metal10",
]

EXTRACT_SCRIPT = os.path.join(ML_ROOT, "congestion", "data_collection", "extract_placement.py")


def run():
    if not os.path.exists(ODB_PATH):
        print(f"[ML] ERROR: ODB not found at {ODB_PATH}", file=sys.stderr)
        sys.exit(1)

    # Step 1: extract placement grid via openroad -python (needs OpenROAD Python API)
    print(f"[ML] Extracting placement grid from {ODB_PATH}")
    result = subprocess.run(
        ["openroad", "-python", EXTRACT_SCRIPT,
         "--odb", ODB_PATH, "--out", PLACEMENT_NPY, "--grid", str(GRID_SIZE)],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[ML] WARNING: placement extraction failed — skipping prediction")
        return

    # Step 2: run model inference (PyTorch only, no OpenROAD needed)
    if not os.path.exists(CHECKPOINT):
        print(f"[ML] WARNING: no checkpoint at {CHECKPOINT}")
        print("[ML] Skipping prediction (run: python3 ml/congestion/model/train.py)")
        return

    import torch
    sys.path.insert(0, os.path.join(ML_ROOT, "congestion", "model"))
    from unet import CongestionUNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CongestionUNet(in_channels=1, out_channels=10)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()

    grid = np.load(PLACEMENT_NPY).astype(np.float32)
    x = torch.tensor(grid).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)
    congestion = pred.squeeze(0).cpu().numpy()
    np.save(CONGESTION_NPY, congestion)

    # Step 3: report and warn
    print("[ML] Predicted congestion per layer:")
    max_congestion = 0.0
    worst_layer = ""
    for i, name in enumerate(LAYER_NAMES):
        mean_val = congestion[i].mean()
        max_val  = congestion[i].max()
        if max_val > max_congestion:
            max_congestion = max_val
            worst_layer = name
        print(f"  {name:8s}: mean={mean_val*100:.1f}%  max={max_val*100:.1f}%")

    if max_congestion > THRESHOLD:
        print(
            f"\n[ML] WARNING: predicted congestion on {worst_layer} "
            f"exceeds threshold ({max_congestion*100:.1f}% > {THRESHOLD*100:.0f}%)"
        )
        print("[ML] Consider re-running placement with a higher PLACE_DENSITY_LB_ADDON")
    else:
        print(f"\n[ML] Congestion looks OK (max={max_congestion*100:.1f}%)")

    # Step 4: save image
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(20, 4))
        gs = gridspec.GridSpec(2, 5, figure=fig)
        for i, name in enumerate(LAYER_NAMES):
            ax = fig.add_subplot(gs[i // 5, i % 5])
            im = ax.imshow(congestion[i], vmin=0, vmax=1, cmap="hot", origin="lower")
            ax.set_title(name, fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.suptitle("Predicted Congestion", fontsize=12)
        plt.tight_layout()
        plt.savefig(CONGESTION_IMG, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[ML] Saved congestion image -> {CONGESTION_IMG}")
    except ImportError:
        pass


if __name__ == "__main__":
    run()
