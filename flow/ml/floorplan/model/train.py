"""
Train the FloorplanGNN on netlist graph -> macro placement pairs.

Usage:
    python3 ml/floorplan/model/train.py \
        --data ml/data \
        --epochs 200 \
        --lr 1e-3 \
        --out ml/floorplan/model/checkpoints

Requires torch-geometric. Install with:
    pip install torch-geometric
"""

import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from tqdm import tqdm

try:
    from torch_geometric.data import Data, DataLoader

    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from gnn import FloorplanGNN


class FloorplanDataset(Dataset):
    def __init__(self, data_dir: str):
        self.samples = []
        graph_files = sorted(glob.glob(os.path.join(data_dir, "*_graph.npz")))
        for g_path in graph_files:
            tag = g_path.replace("_graph.npz", "")
            fp_path = tag + "_floorplan.npz"
            if os.path.exists(fp_path):
                fp = np.load(fp_path, allow_pickle=True)
                if len(fp["macro_positions"]) > 0:
                    self.samples.append((g_path, fp_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        g_path, fp_path = self.samples[idx]

        g = np.load(g_path, allow_pickle=True)
        fp = np.load(fp_path, allow_pickle=True)

        node_feat = torch.tensor(g["node_features"], dtype=torch.float32)
        edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
        edge_weight = torch.tensor(g["edge_weight"], dtype=torch.float32)
        node_names = g["node_names"]

        macro_positions = torch.tensor(
            fp["macro_positions"][:, :2], dtype=torch.float32
        )  # (M, 2) x,y only
        macro_names = fp["macro_names"]

        # Build macro mask: True for nodes that appear in macro_names
        macro_name_set = set(macro_names)
        macro_mask = torch.tensor(
            [name in macro_name_set for name in node_names], dtype=torch.bool
        )

        data = Data(
            x=node_feat,
            edge_index=edge_index,
            edge_attr=edge_weight.unsqueeze(1),
            macro_mask=macro_mask,
            y=macro_positions,
        )
        return data


def hpwl_loss(coords: torch.Tensor, macro_mask: torch.Tensor) -> torch.Tensor:
    """
    Half-perimeter wirelength auxiliary loss — encourages compact placement.
    Penalises the span of predicted macro positions.
    """
    if coords.shape[0] < 2:
        return torch.tensor(0.0, device=coords.device)
    span_x = coords[:, 0].max() - coords[:, 0].min()
    span_y = coords[:, 1].max() - coords[:, 1].min()
    return span_x + span_y


def train(args):
    if not HAS_PYG:
        raise ImportError("torch_geometric is required: pip install torch-geometric")

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = FloorplanDataset(args.data)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No training pairs found in {args.data}. "
            "Run ml/floorplan/data_collection/batch_run.sh first."
        )
    print(f"Dataset size: {len(dataset)} samples")

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size}  Val: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    model = FloorplanGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    l2_loss = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False
        ):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch, batch.macro_mask)
            coord_loss = l2_loss(pred, batch.y)
            aux_loss = hpwl_loss(pred, batch.macro_mask) * args.hpwl_weight
            loss = coord_loss + aux_loss
            loss.backward()
            optimizer.step()
            train_loss += coord_loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.batch, batch.macro_mask)
                val_loss += l2_loss(pred, batch.y).item()
        val_loss /= max(len(val_loader), 1)

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
    parser = argparse.ArgumentParser(description="Train FloorplanGNN")
    parser.add_argument("--data", default="ml/data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--hpwl-weight", type=float, default=0.1, help="Weight for HPWL auxiliary loss"
    )
    parser.add_argument("--out", default="ml/floorplan/model/checkpoints")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
