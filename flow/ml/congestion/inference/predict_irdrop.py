"""
IR-drop prediction inference — run a trained U-Net on a routed design.

The model predicts a per-sample normalised IR-drop heatmap [0, 1] where 1 is
the worst-drop point in that specific design. Normalisation is per-sample
(not global), so no external norm JSON is needed.

Two usage modes:

  1. From pre-extracted feature/label files (no OpenROAD needed):
       python3 ml/congestion/inference/predict_irdrop.py \\
           --features   ml/congestion/data/<label>_features.npz \\
           --irdrop-features ml/congestion/data/<label>_irdrop_labels.npz \\
           --checkpoint ml/congestion/checkpoints/irdrop_best.pt \\
           --out        predicted_irdrop.npz

  2. From a routed ODB (requires openroad/orfs:latest and extracts features
     + PDN geometry automatically via docker_shell):
       python3 ml/congestion/inference/predict_irdrop.py \\
           --odb        /work/results/<platform>/<design>/<tag>/6_final.odb \\
           --spef       /work/results/<platform>/<design>/<tag>/6_final.spef \\
           --liberty    /work/platforms/<platform>/lib/<lib>.lib \\
           --checkpoint ml/congestion/checkpoints/irdrop_best.pt \\
           --out        predicted_irdrop.npz

Output (.npz):
  irdrop_pred_norm  (64, 64) float32  relative worst-drop map [0, 1]

Run from flow/ directory.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from unet import CongestionUNet


def _parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--features", help="Pre-extracted *_features.npz from extract_features.py"
    )
    group.add_argument(
        "--odb", help="Routed ODB (6_final.odb) — features extracted automatically"
    )

    ap.add_argument(
        "--irdrop-features",
        help="Pre-extracted *_irdrop_labels.npz (stripe_density/via_density) — "
        "required alongside --features; not needed with --odb",
    )
    ap.add_argument("--spef", help="SPEF for the routed ODB (used with --odb)")
    ap.add_argument(
        "--liberty", nargs="*", help="Liberty file(s) for the design (used with --odb)"
    )
    ap.add_argument("--net", default="VDD", help="Power net analyzed (default VDD)")
    ap.add_argument(
        "--voltage",
        type=float,
        default=1.1,
        help="Nominal supply voltage (default 1.1)",
    )

    ap.add_argument(
        "--checkpoint", required=True, help="Path to irdrop_best.pt checkpoint"
    )
    ap.add_argument(
        "--out", required=True, help="Output .npz path for predicted IR-drop map"
    )
    ap.add_argument(
        "--base-features",
        type=int,
        default=32,
        help="base_features used when training (default 32)",
    )
    ap.add_argument(
        "--grid",
        type=int,
        default=64,
        help="Grid size (must match training, default 64)",
    )
    return ap.parse_args()


def _extract_from_odb(args) -> tuple:
    """Run extract_features.py and extract_irdrop_labels.py inside docker_shell,
    return (features_npz_path, irdrop_labels_npz_path) on the host."""
    tmp_feat = tempfile.mktemp(suffix="_features.npz")
    tmp_irdrop = tempfile.mktemp(suffix="_irdrop_labels.npz")
    cont_feat_out = f"/work/{os.path.basename(tmp_feat)}"
    cont_irdrop_out = f"/work/{os.path.basename(tmp_irdrop)}"
    feat_script = "/work/ml/congestion/data_collection/extract_features.py"
    irdrop_script = "/work/ml/congestion/data_collection/extract_irdrop_labels.py"

    print(f"[predict] Extracting features from {args.odb} ...")
    cmd = [
        "util/docker_shell",
        "openroad",
        "-python",
        feat_script,
        "--odb",
        args.odb,
        "--out",
        cont_feat_out,
        "--grid",
        str(args.grid),
    ]
    result = subprocess.run(cmd, capture_output=False, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError("Feature extraction failed.")

    print(f"[predict] Running analyze_power_grid + PDN rasterization ...")
    cmd = [
        "util/docker_shell",
        "openroad",
        "-python",
        irdrop_script,
        "--odb",
        args.odb,
        "--out",
        cont_irdrop_out,
        "--grid",
        str(args.grid),
        "--net",
        args.net,
        "--voltage",
        str(args.voltage),
    ]
    if args.spef:
        cmd += ["--spef", args.spef]
    if args.liberty:
        cmd += ["--liberty", *args.liberty]
    result = subprocess.run(cmd, capture_output=False, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError("IR-drop label extraction failed.")

    host_feat = os.path.basename(tmp_feat)
    host_irdrop = os.path.basename(tmp_irdrop)
    if not os.path.exists(host_feat) or not os.path.exists(host_irdrop):
        raise RuntimeError("Expected output files not found after extraction.")
    return host_feat, host_irdrop


def load_features(features_path: str, irdrop_features_path: str) -> torch.Tensor:
    """Load *_features.npz + *_irdrop_labels.npz and return (1, 6, H, W) tensor."""
    feat = np.load(features_path)
    irdrop = np.load(irdrop_features_path)
    x = np.stack(
        [
            feat["cell_density"],
            feat["macro_density"],
            feat["pin_density"],
            feat["fanout_density"],
            irdrop["stripe_density"],
            irdrop["via_density"],
        ]
    ).astype(np.float32)
    return torch.from_numpy(x).unsqueeze(0)  # (1, 6, H, W)


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CongestionUNet(
        in_channels=6,
        base_features=args.base_features,
        num_heatmap_layers=1,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[predict] Loaded checkpoint: {args.checkpoint}")

    tmp_feat = None
    tmp_irdrop = None
    if args.features:
        if not args.irdrop_features:
            raise SystemExit("--irdrop-features is required alongside --features")
        feat_path, irdrop_path = args.features, args.irdrop_features
    else:
        feat_path, irdrop_path = _extract_from_odb(args)
        tmp_feat, tmp_irdrop = feat_path, irdrop_path

    x = load_features(feat_path, irdrop_path).to(device)

    with torch.no_grad():
        out = model(x)
    # Per-sample normalised output [0, 1]: 1 = predicted worst-drop point in design.
    irdrop_norm = out.heatmap[0, 0].cpu().numpy().astype(np.float32)

    print(
        f"[predict] Relative IR-drop map: "
        f"min={irdrop_norm.min():.3f}  max={irdrop_norm.max():.3f}  "
        f"mean={irdrop_norm.mean():.3f}"
    )

    np.savez(args.out, irdrop_pred_norm=irdrop_norm)
    print(f"[predict] Saved → {args.out}")

    for tmp in (tmp_feat, tmp_irdrop):
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    predict(_parse_args())
