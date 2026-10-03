"""
Train the thermal predictor (U-Net by default, or FNO via --arch).

Input  : 5-channel 64×64 placement feature map  (*_features.npz)
Target : 64×64 normalised thermal map            (*_thermal_labels.npz)

Loss: MSE on the normalised thermal map, plus an optional Laplacian smoothness
term  λ·||∇²T_pred||²  that penalises curvature in the prediction. Heat
diffusion produces smooth temperature fields, so this regularises the model
towards physically plausible maps instead of noisy/blocky ones — useful given
the small (48-sample) dataset. The model outputs only the heatmap head;
hotspot and score heads are ignored for this task. At inference time,
denormalize with dataset.denormalize() to recover °C values.

Usage (from flow/):
  python3 util/ml/congestion/training/train_thermal.py \\
      --data-dir util/ml/congestion/data \\
      --checkpoint-dir util/ml/congestion/checkpoints \\
      --epochs 100 \\
      --laplacian-weight 0.1
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from thermal_dataset import ThermalDataset, split_thermal_dataset
from thermal_arch import ARCH_CHOICES, DEFAULT_ARCH_KWARGS, build_thermal_model, thermal_heatmap, write_sidecar

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


def _keys_sha256(keys) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def fit(
    model,
    train_loader,
    val_loader,
    *,
    epochs,
    lr,
    laplacian_weight,
    device,
    on_improve=None,
    log=print,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_val_mse = float("inf")
    best_epoch = None
    best_state = None
    train_mse_final = None
    epoch1_train_mse = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch["thermal"].to(device)
            thermal_pred = thermal_heatmap(model, x)
            loss = _loss(thermal_pred, target, laplacian_weight)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        if epoch == 1:
            epoch1_train_mse = train_loss
        train_mse_final = train_loss

        model.eval()
        val_loss = 0.0
        val_mae_norm = 0.0  # normalised MAE [0, 1]
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                target = batch["thermal"].to(device)
                thermal_pred = thermal_heatmap(model, x)
                val_loss += _loss(thermal_pred, target).item()
                val_mae_norm += (thermal_pred - target).abs().mean().item()

        val_loss /= len(val_loader)
        val_mae_norm /= len(val_loader)

        scheduler.step()

        train_label = "train_loss" if laplacian_weight > 0.0 else "train_mse"
        log(
            f"Epoch {epoch:3d}/{epochs}  "
            f"{train_label}={train_loss:.5f}  val_mse={val_loss:.5f}  "
            f"val_mae={val_mae_norm:.4f} (norm)"
        )

        if val_loss < best_val_mse:
            best_val_mse = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if on_improve is not None:
                on_improve(model, epoch, best_val_mse)

    return {
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "train_mse_final": train_mse_final,
        "epoch1_train_mse": epoch1_train_mse,
    }


def train(args):
    if args.deterministic_algorithms and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        raise SystemExit(
            "--deterministic-algorithms requires the CUBLAS_WORKSPACE_CONFIG "
            "environment variable to be set (e.g. CUBLAS_WORKSPACE_CONFIG=:4096:8)."
        )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.laplacian_weight > 0.0:
        print(f"Laplacian smoothness penalty: λ={args.laplacian_weight}")

    dataset = ThermalDataset(args.data_dir, augment=True)
    print(f"Dataset: {len(dataset)} samples")
    print(
        f"  Per-sample T range: {dataset.t_min:.1f}°C – {dataset.t_max:.1f}°C  "
        f"(each sample normalised independently)"
    )

    if len(dataset) < 3:
        print(
            "WARNING: fewer than 3 samples — results will not generalise. "
            "Run extract_thermal_batch.sh to collect more data first."
        )

    train_set, val_set, test_set = split_thermal_dataset(dataset)
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

    # in_channels=5: cell, macro, pin, fanout + Gaussian-blurred cell density
    # (pre-diffused channel approximates lateral thermal spreading).
    arch_kwargs = DEFAULT_ARCH_KWARGS[args.arch]
    model = build_thermal_model(args.arch, in_channels=5, **arch_kwargs).to(device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    ckpt_best = os.path.join(args.checkpoint_dir, "thermal_best.pt")

    def on_improve(m, epoch, val_mse):
        torch.save(m.state_dict(), ckpt_best)
        print(f"  -> saved {ckpt_best}")

    result = fit(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        laplacian_weight=args.laplacian_weight,
        device=device,
        on_improve=on_improve,
    )

    ckpt_last = os.path.join(args.checkpoint_dir, "thermal_last.pt")
    torch.save(model.state_dict(), ckpt_last)
    print(f"Training complete.  Best val MSE: {result['best_val_mse']:.5f}")

    all_keys = dataset.keys
    train_keys = sorted(all_keys[i] for i in train_set.indices)
    val_keys = sorted(all_keys[i] for i in val_set.indices)
    test_keys = sorted(all_keys[i] for i in test_set.indices)

    meta = {
        "arch": args.arch,
        "arch_kwargs": arch_kwargs,
        "in_channels": 5,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": 1e-4,
        "laplacian_weight": args.laplacian_weight,
        "seed": args.seed,
        "deterministic_algorithms": args.deterministic_algorithms,
        "best_val_mse": result["best_val_mse"],
        "best_epoch": result["best_epoch"],
        "data_dir": args.data_dir,
        "data_keys_sha256": _keys_sha256(all_keys),
        "train_keys": train_keys,
        "val_keys": val_keys,
        "test_keys": test_keys,
        "torch_version": torch.__version__,
    }
    write_sidecar(ckpt_best, meta)
    write_sidecar(ckpt_last, meta)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-dir",
        default="util/ml/congestion/data",
        help="Directory containing *_features.npz and *_thermal_labels.npz",
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
    ap.add_argument(
        "--arch",
        choices=ARCH_CHOICES,
        default="unet",
        help="Thermal model architecture (default: unet)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed torch/numpy and pin cuDNN to deterministic mode (default: unseeded)",
    )
    ap.add_argument(
        "--deterministic-algorithms",
        action="store_true",
        help=(
            "Call torch.use_deterministic_algorithms(True). Requires "
            "CUBLAS_WORKSPACE_CONFIG to be set."
        ),
    )
    train(ap.parse_args())


if __name__ == "__main__":
    main()
