"""
Train the U-Net IR-drop predictor.

Input  : 6-channel 64×64 placement + PDN feature map
         (*_features.npz + *_irdrop_labels.npz)
Target : 64×64 normalised IR-drop map                (*_irdrop_labels.npz)

Loss: MSE on the normalised IR-drop map, plus an optional Laplacian
smoothness term  λ·||∇²V_pred||²  that penalises curvature in the
prediction. Static IR drop, like temperature, is the solution to a smooth
DC resistive (harmonic-ish) field problem, so the same physical
justification used in train_thermal.py applies here — this regularises the
model towards physically plausible maps instead of noisy/blocky ones,
useful given a small dataset. The model outputs only the heatmap head;
hotspot and score heads are ignored for this task. At inference time,
denormalize with dataset.denormalize() to recover volts.

Usage (from flow/):
  python3 util/ml/congestion/training/train_irdrop.py \\
      --data-dir util/ml/congestion/data \\
      --checkpoint-dir util/ml/congestion/checkpoints \\
      --epochs 100 \\
      --laplacian-weight 0.1
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from irdrop_dataset import IRDropDataset, split_irdrop_dataset
from unet import CongestionUNet

# 5-point discrete Laplacian stencil, shared across calls.
_LAPLACIAN_KERNEL = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
).view(1, 1, 3, 3)


def _laplacian(x: torch.Tensor) -> torch.Tensor:
    """Discrete Laplacian ∇²x via 3x3 convolution, replicate-padded at the border."""
    kernel = _LAPLACIAN_KERNEL.to(device=x.device, dtype=x.dtype)
    x_padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(x_padded, kernel)


def _loss(
    pred_heatmap: torch.Tensor,
    target: torch.Tensor,
    laplacian_weight: float = 0.0,
) -> torch.Tensor:
    mse = nn.functional.mse_loss(pred_heatmap, target)
    if laplacian_weight <= 0.0:
        return mse
    smoothness = _laplacian(pred_heatmap).pow(2).mean()
    return mse + laplacian_weight * smoothness


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.laplacian_weight > 0.0:
        print(f"Laplacian smoothness penalty: λ={args.laplacian_weight}")

    dataset = IRDropDataset(args.data_dir, augment=True)
    print(f"Dataset: {len(dataset)} samples")
    print(
        f"  Per-sample drop range: {dataset.d_min*1e3:.1f}mV – {dataset.d_max*1e3:.1f}mV  "
        f"(each sample normalised independently)"
    )

    if len(dataset) < 3:
        print(
            "WARNING: fewer than 3 samples — results will not generalise. "
            "Run extract_irdrop_batch.sh to collect more data first."
        )

    train_set, val_set, test_set = split_irdrop_dataset(dataset)
    print(f"  Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # U-Net: 6-channel input, 1-channel IR-drop output.
    # num_heatmap_layers=1 gives a single drop map — no wasted channels.
    # base_features=32 gives ~7M parameters — appropriate for 64×64 spatial task.
    # in_channels=6: cell, macro, pin, fanout density + stripe_density,
    # via_density (PDN geometry channels, in place of thermal's blur channel).
    model = CongestionUNet(in_channels=6, base_features=32, num_heatmap_layers=1).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch["irdrop"].to(device)
            pred = model(x)
            irdrop_pred = pred.heatmap  # (B, 1, H, W) — single drop channel
            loss = _loss(irdrop_pred, target, args.laplacian_weight)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_mae_norm = 0.0  # normalised MAE [0, 1]
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                target = batch["irdrop"].to(device)
                pred = model(x)
                irdrop_pred = pred.heatmap  # (B, 1, H, W)
                val_loss += _loss(irdrop_pred, target).item()
                val_mae_norm += (irdrop_pred - target).abs().mean().item()

        val_loss /= len(val_loader)
        val_mae_norm /= len(val_loader)

        scheduler.step()

        train_label = "train_loss" if args.laplacian_weight > 0.0 else "train_mse"
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"{train_label}={train_loss:.5f}  val_mse={val_loss:.5f}  "
            f"val_mae={val_mae_norm:.4f} (norm)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = os.path.join(args.checkpoint_dir, "irdrop_best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  -> saved {ckpt}")

    torch.save(
        model.state_dict(),
        os.path.join(args.checkpoint_dir, "irdrop_last.pt"),
    )
    print(f"Training complete.  Best val MSE: {best_val_loss:.5f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-dir",
        default="util/ml/congestion/data",
        help="Directory containing *_features.npz and *_irdrop_labels.npz",
    )
    ap.add_argument("--checkpoint-dir", default="util/ml/congestion/checkpoints")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--laplacian-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the Laplacian smoothness penalty on train loss "
            "(0 disables it; val loss stays plain MSE for comparability)."
        ),
    )
    train(ap.parse_args())


if __name__ == "__main__":
    main()
