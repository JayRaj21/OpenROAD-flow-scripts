"""
Run congestion prediction on a placement ODB or a pre-extracted .npy file.

Usage (from host, with a pre-extracted placement grid):
    python3 ml/congestion/model/predict.py \
        --placement ml/data/nangate45_gcd_placement.npy \
        --checkpoint ml/congestion/model/checkpoints/best.pt \
        --out ml/data/nangate45_gcd_predicted_congestion.npy \
        --image ml/data/nangate45_gcd_predicted_congestion.png
"""

import argparse
import os
import numpy as np
import torch

from unet import CongestionUNet

LAYER_NAMES = [
    "metal1",
    "metal2",
    "metal3",
    "metal4",
    "metal5",
    "metal6",
    "metal7",
    "metal8",
    "metal9",
    "metal10",
]


def predict(placement: np.ndarray, checkpoint: str, device: torch.device) -> np.ndarray:
    model = CongestionUNet(in_channels=1, out_channels=10)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    x = (
        torch.tensor(placement, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        pred = model(x)
    return pred.squeeze(0).cpu().numpy()  # (10, H, W)


def save_image(congestion: np.ndarray, path: str):
    try:
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

        plt.suptitle("Predicted Congestion per Routing Layer", fontsize=12)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved image -> {path}")
    except ImportError:
        print("matplotlib not available; skipping image output")


def main():
    parser = argparse.ArgumentParser(
        description="Predict congestion from placement grid"
    )
    parser.add_argument("--placement", required=True, help="Input placement .npy file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint .pt file")
    parser.add_argument(
        "--out", required=True, help="Output predicted congestion .npy file"
    )
    parser.add_argument(
        "--image", default=None, help="Optional output image path (.png)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    placement = np.load(args.placement).astype(np.float32)
    congestion = predict(placement, args.checkpoint, device)

    np.save(args.out, congestion)
    print(f"Saved predicted congestion {congestion.shape} -> {args.out}")

    for i, name in enumerate(LAYER_NAMES):
        mean_val = congestion[i].mean() * 100
        max_val = congestion[i].max() * 100
        print(f"  {name:8s}: mean={mean_val:.1f}%  max={max_val:.1f}%")

    if args.image:
        save_image(congestion, args.image)


if __name__ == "__main__":
    main()
