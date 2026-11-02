"""
Pluggable thermal-model architecture: build either arch, dispatch the
forward pass to the right output shape, and read/write the checkpoint
sidecar metadata that records which architecture a `.pt` file is.

Only `unet` and `fno` are imported here — not anything under `loop/`
(`loop/train_lodo.py` imports `training/train_thermal.py`, so importing
`loop/` from here would be circular).
"""

import hashlib
import json
import os

import torch
import torch.nn as nn

from unet import CongestionUNet
from fno import FNO2d

ARCH_CHOICES = ("unet", "fno")

DEFAULT_ARCH_KWARGS = {
    "unet": {"base_features": 32},
    "fno": {"width": 32, "modes": 12, "n_layers": 4, "padding": 8},
}


def build_thermal_model(arch: str, in_channels: int = 5, **arch_kwargs) -> nn.Module:
    if arch == "unet":
        kwargs = {**DEFAULT_ARCH_KWARGS["unet"], **arch_kwargs}
        return CongestionUNet(
            in_channels, kwargs["base_features"], num_heatmap_layers=1
        )
    if arch == "fno":
        kwargs = {**DEFAULT_ARCH_KWARGS["fno"], **arch_kwargs}
        return FNO2d(in_channels, **kwargs)
    raise ValueError(f"Unknown thermal architecture: {arch!r} (expected one of {ARCH_CHOICES})")


def thermal_heatmap(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, CongestionUNet):
        return model(x).heatmap
    if isinstance(model, FNO2d):
        return model(x)
    raise TypeError(f"Unsupported thermal model type: {type(model)!r}")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def write_sidecar(ckpt: str, meta: dict) -> None:
    meta = dict(meta)
    meta["pt_sha256"] = _sha256_file(ckpt)
    with open(ckpt + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def load_thermal_model(
    ckpt: str,
    device,
    arch: str | None = None,
    base_features: int | None = None,
) -> tuple[nn.Module, dict]:
    sidecar_path = ckpt + ".json"
    if os.path.isfile(sidecar_path):
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        actual_sha = _sha256_file(ckpt)
        if sidecar.get("pt_sha256") != actual_sha:
            raise ValueError(
                f"{ckpt} does not match its sidecar {sidecar_path} "
                f"(sha256 {actual_sha[:12]} != {sidecar.get('pt_sha256', '')[:12]})"
            )

        # Older sidecars (e.g. loop/train_lodo.py's, which predate this
        # module) carry pt_sha256 but no "arch" key -- they are all U-Net,
        # from before any other architecture existed, so fall back to the
        # explicit --arch (if given) or "unet" rather than KeyError.
        resolved_arch = sidecar.get("arch", arch or "unet")
        if arch is not None and arch != resolved_arch:
            raise ValueError(
                f"--arch {arch!r} conflicts with sidecar arch {resolved_arch!r} for {ckpt}"
            )

        arch_kwargs = dict(sidecar.get("arch_kwargs", {}))
        if base_features is not None:
            sidecar_base_features = arch_kwargs.get("base_features")
            if sidecar_base_features is not None and base_features != sidecar_base_features:
                raise ValueError(
                    f"--base-features {base_features!r} conflicts with sidecar "
                    f"base_features {sidecar_base_features!r} for {ckpt}"
                )
            if resolved_arch == "unet":
                arch_kwargs["base_features"] = base_features

        in_channels = sidecar.get("in_channels", 5)
        resolved = {
            "arch": resolved_arch,
            "arch_kwargs": arch_kwargs,
            "in_channels": in_channels,
            "source": "sidecar",
        }
    else:
        resolved_arch = arch or "unet"
        arch_kwargs = {}
        if resolved_arch == "unet":
            arch_kwargs["base_features"] = base_features or 32
        in_channels = 5
        resolved = {
            "arch": resolved_arch,
            "arch_kwargs": arch_kwargs,
            "in_channels": in_channels,
            "source": "default",
        }

    model = build_thermal_model(
        resolved["arch"], resolved["in_channels"], **resolved["arch_kwargs"]
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, resolved
