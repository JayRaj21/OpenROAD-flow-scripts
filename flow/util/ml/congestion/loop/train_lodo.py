"""
Train a leave-design-family-out thermal U-Net for the Stage B gate.

The gate asks whether the surrogate can rank placement-density variants of a
design it has never seen. A checkpoint must therefore exclude the design from
every PDK: sky130hd/riscv32i and gf180/riscv32i are the same netlist, so
holding out one PDK only would leak the other.

Family decision: `aes_lvt` (asap7 only) belongs to the `aes` family. It is the
same RTL netlist as `aes` on a different cell library, so it is the same kind
of leak as the same design on another PDK. `jpeg_lvt` would likewise belong to
`jpeg`. Every other design name is its own family. --holdout-design takes the
family name and excludes every base sample whose parsed design name is in that
family (matched exactly on the parsed name, never by substring).

The model, optimizer, scheduler, gradient clipping and loss are those of
train_thermal.py (the loss is imported). Training uses an augment=True
dataset; the internal validation split (3 samples drawn with the seed from the
training designs only, and removed from training) uses an augment=False
instance. The checkpoint with the best internal validation MSE is kept. Right
before training starts, the keys of the datasets actually handed to the data
loaders are checked to contain none of the excluded keys and no key of the
holdout family; the sidecar's train_keys and val_keys are read back from those
same datasets. A sidecar <out>.json records the excluded keys, the training
keys, epochs, seed, batch size, learning rate, Laplacian weight, a hash of the
data directory's key list and the sha256 of the .pt so gate_rank.py can verify
that a checkpoint was not trained on the design it scores. The sidecar is
written by this script and is not proof against a deliberate forger.

--verify-only trains nothing: it exits 0 if --out holds the checkpoint these
arguments would train (the reuse rule) and 3 otherwise.

Usage (from flow/):
  python3 util/ml/congestion/loop/train_lodo.py --data-dir util/ml/congestion/data \\
      --holdout-design riscv32i --out util/ml/congestion/checkpoints/thermal_lodo_riscv32i.pt \\
      --epochs 200 --seed 0 [--batch-size 4] [--laplacian-weight 0]
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "training"))

from laplacian_sweep import _parse_key  # noqa: E402
from thermal_dataset import ThermalDataset  # noqa: E402
from train_thermal import _loss  # noqa: E402
from unet import CongestionUNet  # noqa: E402

N_VAL = 3
FORBIDDEN_OUT_NAME = "thermal_best.pt"
FORBIDDEN_DIR = os.path.realpath(os.path.join(HERE, "..", "experiments", "thermal_loop"))


class LeakError(RuntimeError):
    pass


def family_of(design: str) -> str:
    return design[: -len("_lvt")] if design.endswith("_lvt") else design


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_dir_keys(data_dir: str) -> list[str]:
    """Base sample keys (those with both a features and a thermal label file)."""
    names = os.listdir(data_dir)
    feats = {n[: -len("_features.npz")] for n in names if n.endswith("_features.npz")}
    therm = {n[: -len("_thermal_labels.npz")] for n in names if n.endswith("_thermal_labels.npz")}
    return sorted(feats & therm)


def keys_sha256(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def reusable(ckpt: str, holdout: str, epochs: int, seed: int, batch_size: int,
             lr: float, laplacian_weight: float, data_dir: str) -> tuple[bool, str]:
    """Whether an existing checkpoint is the one these arguments would train."""
    sidecar_path = ckpt + ".json"
    if not os.path.isfile(ckpt) or not os.path.isfile(sidecar_path):
        return False, "no checkpoint or sidecar"
    with open(sidecar_path) as f:
        sc = json.load(f)
    if sc.get("pt_sha256") != sha256_file(ckpt):
        return False, "checkpoint hash does not match its sidecar"
    wanted = (
        ("holdout_design", holdout), ("epochs", epochs), ("seed", seed), ("batch_size", batch_size),
        ("lr", lr), ("laplacian_weight", laplacian_weight),
        ("data_keys_sha256", keys_sha256(data_dir_keys(data_dir))),
    )
    for name, want in wanted:
        if sc.get(name) != want:
            return False, f"sidecar {name} is {sc.get(name)!r}, requested {want!r}"
    return True, "matches"


def _under_forbidden_dir(path: str) -> bool:
    real = os.path.realpath(path)
    forbidden = os.path.realpath(FORBIDDEN_DIR)
    return real == forbidden or real.startswith(forbidden + os.sep)


def check_paths(data_dir: str, out: str) -> None:
    if os.path.basename(out) == FORBIDDEN_OUT_NAME:
        raise SystemExit(f"Refusing to write {out}: {FORBIDDEN_OUT_NAME} is the demo checkpoint")
    for label, path in (("--data-dir", data_dir), ("--out", out)):
        if _under_forbidden_dir(path):
            raise SystemExit(f"Refusing {label} {path}: nothing under {FORBIDDEN_DIR} may be read or written")


def split_keys(keys: list[str], holdout: str) -> tuple[list[str], list[str]]:
    """Return (excluded keys, training-pool keys) for the holdout family."""
    excluded, pool = [], []
    for key in keys:
        if not key.endswith("_base"):
            raise SystemExit(f"Unexpected non-base sample key in data dir: {key}")
        _, design = _parse_key(key)
        (excluded if family_of(design) == holdout else pool).append(key)
    return excluded, pool


def assert_no_leak(train_keys: list[str], val_keys: list[str], excluded: list[str], holdout: str) -> None:
    for label, keys in (("training", train_keys), ("validation", val_keys)):
        bad = sorted(k for k in keys if k in set(excluded) or family_of(_parse_key(k)[1]) == holdout)
        if bad:
            raise LeakError(f"{label} dataset contains excluded keys: {', '.join(bad)}")


def prepare_data(data_dir: str, holdout: str, seed: int) -> SimpleNamespace:
    train_ds = ThermalDataset(data_dir, augment=True)
    eval_ds = ThermalDataset(data_dir, augment=False)

    excluded, pool = split_keys(train_ds.keys, holdout)
    if not excluded:
        raise SystemExit(f"No sample in {data_dir} belongs to family '{holdout}'")
    if len(pool) <= N_VAL:
        raise SystemExit(f"Only {len(pool)} training-pool samples left; need more than {N_VAL}")

    rng = np.random.RandomState(seed)
    val_keys = sorted(rng.choice(pool, size=N_VAL, replace=False).tolist())
    planned = [k for k in pool if k not in val_keys]
    index = {k: i for i, k in enumerate(train_ds.keys)}

    train_set = Subset(train_ds, [index[k] for k in planned])
    val_set = Subset(eval_ds, [index[k] for k in val_keys])
    train_keys = [train_ds.keys[i] for i in train_set.indices]
    actual_val_keys = [eval_ds.keys[i] for i in val_set.indices]
    assert_no_leak(train_keys, actual_val_keys, excluded, holdout)
    return SimpleNamespace(train_set=train_set, val_set=val_set, excluded=excluded,
                           train_keys=train_keys, val_keys=actual_val_keys,
                           data_keys_sha256=keys_sha256(train_ds.keys))


class BestTracker:
    """Keeps the model state of the epoch with the lowest validation MSE."""

    def __init__(self):
        self.best_val = float("inf")
        self.best_epoch = 0
        self.best_state = None

    def update(self, epoch: int, val_mse: float, model: nn.Module) -> None:
        if val_mse < self.best_val:
            self.best_val = val_mse
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())


def train(args) -> dict:
    check_paths(args.data_dir, args.out)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = prepare_data(args.data_dir, args.holdout_design, args.seed)

    print(f"Device: {device}")
    print(f"Holdout family: {args.holdout_design}")
    print(f"Excluded keys ({len(data.excluded)}): {', '.join(data.excluded)}")
    print(f"Validation keys ({len(data.val_keys)}): {', '.join(data.val_keys)}")
    print(f"Training samples: {len(data.train_keys)}")

    train_loader = DataLoader(data.train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(data.val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CongestionUNet(in_channels=5, base_features=32, num_heatmap_layers=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best = BestTracker()
    train_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch["thermal"].to(device)
            loss = _loss(model(x).heatmap, target, args.laplacian_weight)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                target = batch["thermal"].to(device)
                val_mse += _loss(model(x).heatmap, target).item()
        val_mse /= len(val_loader)
        best.update(epoch, val_mse, model)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(best.best_state, args.out)
    sidecar = {
        "pt_sha256": sha256_file(args.out),
        "holdout_design": args.holdout_design,
        "excluded_keys": data.excluded,
        "val_keys": data.val_keys,
        "train_keys": data.train_keys,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "laplacian_weight": args.laplacian_weight,
        "data_keys_sha256": data.data_keys_sha256,
        "n_train": len(data.train_keys),
        "n_val": len(data.val_keys),
        "n_excluded": len(data.excluded),
        "best_val_mse": best.best_val,
        "best_epoch": best.best_epoch,
        "final_train_loss": train_loss,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Epochs: {args.epochs}")
    print(f"Final train loss: {train_loss:.5f}")
    print(f"Best validation MSE: {best.best_val:.5f} (epoch {best.best_epoch})")
    print(f"Excluded keys: {', '.join(data.excluded)}")
    print(f"Training samples: {len(data.train_keys)}")
    print(f"Saved {args.out} and {args.out}.json")
    return sidecar


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default="util/ml/congestion/data")
    ap.add_argument("--holdout-design", required=True, help="design family, e.g. riscv32i, aes, ibex")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--laplacian-weight", type=float, default=0.0)
    ap.add_argument("--reuse-if-valid", action="store_true",
                    help="skip training when --out already holds a checkpoint with the same family, epochs, seed, batch size, lr, Laplacian weight and data keys")
    ap.add_argument("--verify-only", action="store_true",
                    help="train nothing; exit 0 if --out is the checkpoint these arguments would train, else 3")
    args = ap.parse_args()
    if args.verify_only:
        ok, why = reusable(args.out, args.holdout_design, args.epochs, args.seed, args.batch_size,
                           args.lr, args.laplacian_weight, args.data_dir)
        print(f"[{'VERIFIED' if ok else 'MISMATCH'}] {args.out}: {why}")
        sys.exit(0 if ok else 3)
    if args.reuse_if_valid:
        ok, why = reusable(args.out, args.holdout_design, args.epochs, args.seed, args.batch_size,
                           args.lr, args.laplacian_weight, args.data_dir)
        print(f"[{'REUSE' if ok else 'RETRAIN'}] {args.out}: {why}")
        if ok:
            return
    train(args)


if __name__ == "__main__":
    main()
