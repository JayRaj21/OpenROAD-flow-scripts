"""
Tests for the pluggable thermal architecture (`models/thermal_arch.py`) and
the `--arch` plumbing through `train_thermal.py` / `predict_thermal.py`.

Run from the flow/ directory:
  python3 util/ml/congestion/tests/test_thermal_arch.py -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "..", "models")
_TRAINING = os.path.join(_HERE, "..", "training")
_INFERENCE = os.path.join(_HERE, "..", "inference")
sys.path.insert(0, _MODELS)
sys.path.insert(0, _TRAINING)
sys.path.insert(0, _INFERENCE)
sys.path.insert(0, _HERE)

from unet import CongestionUNet  # noqa: E402
from fno import FNO2d  # noqa: E402
from thermal_arch import (  # noqa: E402
    ARCH_CHOICES,
    build_thermal_model,
    load_thermal_model,
    thermal_heatmap,
    write_sidecar,
)
from thermal_dataset import ThermalDataset, split_thermal_dataset  # noqa: E402
from train_thermal import fit  # noqa: E402

from test_arch_sweep import _make_synthetic_dataset  # noqa: E402

FLOW_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
TRAIN_THERMAL_PATH = os.path.join(_TRAINING, "train_thermal.py")
PREDICT_THERMAL_PATH = os.path.join(_INFERENCE, "predict_thermal.py")

GRID = 64
DEVICE = torch.device("cpu")


class TestBuildThermalModel(unittest.TestCase):
    def test_unet_matches_congestion_unet(self):
        built = build_thermal_model("unet", in_channels=5)
        reference = CongestionUNet(5, 32, num_heatmap_layers=1)
        built_keys = {k: tuple(v.shape) for k, v in built.state_dict().items()}
        ref_keys = {k: tuple(v.shape) for k, v in reference.state_dict().items()}
        self.assertEqual(built_keys, ref_keys)

    def test_unknown_arch_raises(self):
        with self.assertRaises(ValueError):
            build_thermal_model("resnet")


class TestThermalHeatmap(unittest.TestCase):
    def _check(self, arch):
        model = build_thermal_model(arch, in_channels=5)
        x = torch.rand(2, 5, GRID, GRID)
        out = thermal_heatmap(model, x)
        self.assertEqual(tuple(out.shape), (2, 1, GRID, GRID))
        self.assertTrue(bool((out > 0).all()))
        self.assertTrue(bool((out < 1).all()))

    def test_unet_shape_and_range(self):
        self._check("unet")

    def test_fno_shape_and_range(self):
        self._check("fno")

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            thermal_heatmap(torch.nn.Linear(1, 1), torch.rand(1, 1))


class TestSidecarRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ckpt = os.path.join(self.tmpdir.name, "thermal_best.pt")
        model = build_thermal_model("unet", in_channels=5)
        torch.save(model.state_dict(), self.ckpt)
        write_sidecar(
            self.ckpt,
            {
                "arch": "unet",
                "arch_kwargs": {"base_features": 32},
                "in_channels": 5,
            },
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_round_trip(self):
        model, resolved = load_thermal_model(self.ckpt, DEVICE)
        self.assertEqual(resolved["arch"], "unet")
        self.assertEqual(resolved["source"], "sidecar")
        self.assertIsInstance(model, CongestionUNet)

    def test_conflicting_arch_raises(self):
        with self.assertRaises(ValueError):
            load_thermal_model(self.ckpt, DEVICE, arch="fno")

    def test_sha_mismatch_raises(self):
        with open(self.ckpt + ".json") as f:
            meta = json.load(f)
        meta["pt_sha256"] = "0" * 64
        with open(self.ckpt + ".json", "w") as f:
            json.dump(meta, f)
        with self.assertRaises(ValueError):
            load_thermal_model(self.ckpt, DEVICE)

    def test_no_sidecar_falls_back_to_unet(self):
        bare_ckpt = os.path.join(self.tmpdir.name, "bare.pt")
        model = build_thermal_model("unet", in_channels=5)
        torch.save(model.state_dict(), bare_ckpt)
        loaded, resolved = load_thermal_model(bare_ckpt, DEVICE)
        self.assertEqual(resolved["arch"], "unet")
        self.assertEqual(resolved["source"], "default")
        self.assertIsInstance(loaded, CongestionUNet)

    def test_fno_weights_into_unet_raises(self):
        fno_ckpt = os.path.join(self.tmpdir.name, "fno_bare.pt")
        fno_model = build_thermal_model("fno", in_channels=5)
        torch.save(fno_model.state_dict(), fno_ckpt)
        with self.assertRaises(RuntimeError):
            load_thermal_model(fno_ckpt, DEVICE, arch="unet")

    def test_sidecar_without_arch_key_falls_back_to_unet(self):
        # Reproduces loop/train_lodo.py's sidecar format, which predates
        # this module and has no "arch" key (every checkpoint it ever
        # wrote was U-Net) -- must not KeyError.
        legacy_ckpt = os.path.join(self.tmpdir.name, "legacy_lodo.pt")
        model = build_thermal_model("unet", in_channels=5)
        torch.save(model.state_dict(), legacy_ckpt)
        write_sidecar(legacy_ckpt, {"holdout_design": "aes", "seed": 0})
        loaded, resolved = load_thermal_model(legacy_ckpt, DEVICE)
        self.assertEqual(resolved["arch"], "unet")
        self.assertIsInstance(loaded, CongestionUNet)


class TestSeededFitDeterministic(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self.tmpdir.name, "data")
        os.makedirs(self.data_dir)
        _make_synthetic_dataset(self.data_dir)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run(self):
        torch.manual_seed(0)
        dataset = ThermalDataset(self.data_dir, augment=True)
        train_set, val_set, _ = split_thermal_dataset(dataset)
        from torch.utils.data import DataLoader

        train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0)
        torch.manual_seed(0)
        model = build_thermal_model("unet", in_channels=5)
        result = fit(
            model,
            train_loader,
            val_loader,
            epochs=2,
            lr=1e-3,
            laplacian_weight=0.0,
            device=DEVICE,
            log=lambda *a, **k: None,
        )
        return result["best_state"]

    def test_same_seed_same_state_dict(self):
        state1 = self._run()
        state2 = self._run()
        self.assertEqual(set(state1), set(state2))
        for k in state1:
            self.assertTrue(torch.equal(state1[k], state2[k]), f"mismatch at {k}")


class TestTrainPredictCLI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self.tmpdir.name, "data")
        os.makedirs(self.data_dir)
        _make_synthetic_dataset(self.data_dir)
        self.ckpt_dir = os.path.join(self.tmpdir.name, "ckpt")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_fno_train_then_predict(self):
        train_cmd = [
            sys.executable,
            TRAIN_THERMAL_PATH,
            "--data-dir",
            self.data_dir,
            "--checkpoint-dir",
            self.ckpt_dir,
            "--arch",
            "fno",
            "--epochs",
            "2",
            "--batch-size",
            "2",
        ]
        result = subprocess.run(train_cmd, cwd=FLOW_DIR, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

        ckpt = os.path.join(self.ckpt_dir, "thermal_best.pt")
        self.assertTrue(os.path.isfile(ckpt))
        self.assertTrue(os.path.isfile(ckpt + ".json"))

        features = next(
            f for f in os.listdir(self.data_dir) if f.endswith("_features.npz")
        )
        out_npz = os.path.join(self.tmpdir.name, "pred.npz")
        predict_cmd = [
            sys.executable,
            PREDICT_THERMAL_PATH,
            "--features",
            os.path.join(self.data_dir, features),
            "--checkpoint",
            ckpt,
            "--out",
            out_npz,
        ]
        result = subprocess.run(predict_cmd, cwd=FLOW_DIR, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("arch=fno", result.stdout)

        pred = np.load(out_npz)["thermal_pred_norm"]
        self.assertEqual(pred.shape, (GRID, GRID))


if __name__ == "__main__":
    unittest.main()
