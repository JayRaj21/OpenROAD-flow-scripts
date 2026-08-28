"""
Dataset loader for the IR-drop prediction track.

Pairs placement feature maps with PDNSim IR-drop labels:
  Input  (*_features.npz + *_irdrop_labels.npz):  x          (6, 64, 64)  float32
  Target (*_irdrop_labels.npz):                   irdrop_map (64, 64)    float32  [V]

Channel layout:
  0: cell_density          — from *_features.npz (same as thermal track)
  1: macro_density         — from *_features.npz
  2: pin_density            — from *_features.npz
  3: fanout_density         — from *_features.npz
  4: stripe_density         — from *_irdrop_labels.npz (PDN wire area/cell)
  5: via_density            — from *_irdrop_labels.npz (PDN via count/cell)

Unlike ThermalDataset's 5th channel (a Gaussian-blurred cell density used as
a lateral-diffusion proxy), IR drop is a DC resistive problem, not a
diffusion one, so no blur channel is used here — stripe_density and
via_density (actual PDN geometry) are the physically relevant additional
channels instead. current_density_proxy (also in *_irdrop_labels.npz) is
deliberately left out of the input: it's built from the same cell-type
area weighting as cell_density/macro_density and would be near-redundant
with them as a training input; it's kept in the label file purely as a
diagnostic/visualization channel.

File layout choice: stripe_density/via_density/current_density_proxy live in
*_irdrop_labels.npz (written by extract_irdrop_labels.py) rather than being
added as new keys to *_features.npz. extract_features.py's 4 channels are
placement-only (no PDN or power information available at that stage in
general; it's run as early as post-placement), while the new channels
require a routed ODB with a PDN, so they belong with the IR-drop label
extractor that already needs a routed ODB. This also matches the additive
constraint of leaving extract_features.py's existing output format
byte-for-byte unchanged.

A sample is included only when BOTH files exist for the same label prefix.

Normalisation strategy — per-sample:
  Each irdrop_map is independently min-max normalised to [0, 1] within that
  sample, identical rationale and mechanism to ThermalDataset: absolute
  IR-drop magnitude varies with die size, PDN density, and total switching
  current across designs, so the model is trained on the *relative*
  distribution within a design.

  Consequence: d_min / d_max on the dataset object are per-sample extremes
  stored as lists (mirrors ThermalDataset's t_min/t_max). Inference
  normalises the same way and produces a relative IR-drop map [0, 1].

Augmentation (optional, applied during training):
  - Random horizontal flip
  - Random vertical flip
  Both the input and target are flipped identically to preserve correspondence.
"""

import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split


class IRDropDataset(Dataset):

    def __init__(self, data_dir: str, augment: bool = False):
        self.augment = augment

        feat_files = {
            os.path.basename(p).replace("_features.npz", ""): p
            for p in glob.glob(os.path.join(data_dir, "*_features.npz"))
        }
        irdrop_files = {
            os.path.basename(p).replace("_irdrop_labels.npz", ""): p
            for p in glob.glob(os.path.join(data_dir, "*_irdrop_labels.npz"))
        }

        keys = sorted(feat_files.keys() & irdrop_files.keys())
        if not keys:
            raise RuntimeError(
                f"No matched feature+irdrop pairs found in {data_dir}.\n"
                f"  Feature files : {len(feat_files)}\n"
                f"  IR-drop files : {len(irdrop_files)}\n"
                "Run extract_features.py and extract_irdrop_labels.py first."
            )

        self.pairs = [(feat_files[k], irdrop_files[k]) for k in keys]
        self.keys = keys

        # Per-sample IR-drop ranges (volts) for reference / diagnostics.
        self.d_mins: list[float] = []
        self.d_maxs: list[float] = []
        for _, dp in self.pairs:
            d = np.load(dp)["irdrop_map"].astype(np.float32)
            self.d_mins.append(float(d.min()))
            self.d_maxs.append(float(d.max()))

        # Dataset-level properties kept for logging convenience.
        self.d_min = min(self.d_mins)
        self.d_max = max(self.d_maxs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        feat_path, irdrop_path = self.pairs[idx]

        feat = np.load(feat_path)
        irdrop_npz = np.load(irdrop_path)
        x = np.stack(
            [
                feat["cell_density"],
                feat["macro_density"],
                feat["pin_density"],
                feat["fanout_density"],
                irdrop_npz["stripe_density"],
                irdrop_npz["via_density"],
            ]
        ).astype(
            np.float32
        )  # (6, 64, 64)

        d = irdrop_npz["irdrop_map"].astype(np.float32)  # (64, 64)  V

        # Per-sample normalisation: each design is independently [0, 1].
        d_lo, d_hi = float(d.min()), float(d.max())
        denom = (d_hi - d_lo) if d_hi > d_lo else 1.0
        d_norm = np.clip((d - d_lo) / denom, 0.0, 1.0)

        x_t = torch.from_numpy(x)
        target_t = torch.from_numpy(d_norm).unsqueeze(0)  # (1, 64, 64)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                x_t = torch.flip(x_t, dims=[2])
                target_t = torch.flip(target_t, dims=[2])
            if torch.rand(1).item() > 0.5:
                x_t = torch.flip(x_t, dims=[1])
                target_t = torch.flip(target_t, dims=[1])

        return {"x": x_t, "irdrop": target_t, "d_min": d_lo, "d_max": d_hi}

    def denormalize(
        self, d_norm: torch.Tensor, d_min: float, d_max: float
    ) -> torch.Tensor:
        """Convert per-sample normalised [0,1] back to volts."""
        return d_norm * (d_max - d_min) + d_min


def split_irdrop_dataset(
    dataset: IRDropDataset,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    n = len(dataset)
    n_tr = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    n_te = max(0, n - n_tr - n_val)
    return random_split(
        dataset,
        [n_tr, n_val, n_te],
        generator=torch.Generator().manual_seed(seed),
    )
