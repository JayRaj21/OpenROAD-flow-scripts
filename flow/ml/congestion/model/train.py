"""
Train the CongestionUNet on placement grid -> congestion vector pairs.

Usage:
    python3 ml/congestion/model/train.py \
        --data ml/data \
        --epochs 100 \
        --batch 8 \
        --lr 1e-3 \
        --out ml/congestion/model/checkpoints

The congestion label for each design is a (10,) vector (one value per routing
layer). During training this vector is broadcast to a (10, H, W) target where
every spatial bin has the same per-layer value. This trains the model to
predict uniform congestion maps; once per-gcell labels become available the
dataset class can be extended to load them directly.
"""

import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from unet import CongestionUNet


class CongestionDataset(Dataset):
    def __init__(self, data_dir: str, grid_size: int = 64):
        self.samples = []
        placement_files = sorted(glob.glob(os.path.join(data_dir, "*_placement.npy")))
        for p_path in placement_files:
            tag = p_path.replace("_placement.npy", "")
            c_path = tag + "_congestion.npy"
            if os.path.exists(c_path):
                self.samples.append((p_path, c_path))
        self.grid_size = grid_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p_path, c_path = self.samples[idx]

        placement = np.load(p_path).astype(np.float32)
        congestion = np.load(c_path).astype(np.float32)  # shape (10,)

        # Resize placement grid if needed
        if placement.shape != (self.grid_size, self.grid_size):
            placement = np.array(
                torch.nn.functional.interpolate(
                    torch.tensor(placement).unsqueeze(0).unsqueeze(0),
                    size=(self.grid_size, self.grid_size),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .numpy()
            )

        # Broadcast congestion vector to 2D target: (10, H, W)
        target = congestion[:, None, None] * np.ones(
            (10, self.grid_size, self.grid_size), dtype=np.float32
        )

        x = torch.tensor(placement).unsqueeze(0)  # (1, H, W)
        y = torch.tensor(target)  # (10, H, W)
        return x, y


def train(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = CongestionDataset(args.data, grid_size=args.grid)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No training pairs found in {args.data}. "
            "Run batch_run.sh first to generate data."
        )
    print(f"Dataset size: {len(dataset)} samples")

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size}  Val: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    model = CongestionUNet(in_channels=1, out_channels=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False
        ):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
        val_loss /= val_size

        scheduler.step()

        print(
            f"Epoch {epoch:4d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = os.path.join(args.out, "best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"           -> saved checkpoint to {ckpt}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train CongestionUNet")
    parser.add_argument(
        "--data", default="ml/data", help="Directory with .npy training pairs"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grid", type=int, default=64, help="Grid resolution")
    parser.add_argument("--out", default="ml/congestion/model/checkpoints")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
