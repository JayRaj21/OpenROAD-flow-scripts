"""
Tests for the thermal architecture-sweep harness (`training/arch_sweep.py`)
and the FNO model (`models/fno.py`).

Run from the flow/ directory:
  python3 util/ml/congestion/tests/test_arch_sweep.py -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "..", "models")
_TRAINING = os.path.join(_HERE, "..", "training")
sys.path.insert(0, _MODELS)
sys.path.insert(0, _TRAINING)

from fno import FNO2d  # noqa: E402

import arch_sweep  # noqa: E402

ARCH_SWEEP_PATH = os.path.join(_TRAINING, "arch_sweep.py")
FLOW_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))

GRID = 64

# (pdk, design) pairs: 6 keys across 3 families (aes/aes_lvt share a family).
SYNTH_DESIGNS = [
    ("fab1", "aes"),
    ("fab1", "aes_lvt"),
    ("fab1", "gcd"),
    ("fab2", "gcd"),
    ("fab1", "ibex"),
    ("fab2", "ibex"),
]


def _make_synthetic_dataset(out_dir: str, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    keys = []
    for pdk, design in SYNTH_DESIGNS:
        key = f"{pdk}_{design}_base"
        keys.append(key)

        cell = gaussian_filter(rng.random((GRID, GRID)).astype(np.float32), sigma=3)
        cell /= cell.max() + 1e-9
        macro = (rng.random((GRID, GRID)) > 0.9).astype(np.float32)
        pin = gaussian_filter(rng.random((GRID, GRID)).astype(np.float32), sigma=2)
        pin /= pin.max() + 1e-9
        fanout = gaussian_filter(rng.random((GRID, GRID)).astype(np.float32), sigma=4)
        fanout /= fanout.max() + 1e-9

        np.savez(
            os.path.join(out_dir, f"{key}_features.npz"),
            cell_density=cell,
            macro_density=macro,
            pin_density=pin,
            fanout_density=fanout,
        )

        thermal = gaussian_filter(cell, sigma=5) * 50.0 + 300.0
        thermal += rng.random((GRID, GRID)).astype(np.float32) * 0.5
        np.savez(
            os.path.join(out_dir, f"{key}_thermal_labels.npz"),
            thermal_map=thermal.astype(np.float32),
            power_grid=cell.astype(np.float32),
        )
    return keys


class TestFNO(unittest.TestCase):
    def test_shape_and_range(self):
        model = FNO2d(in_channels=5)
        x = torch.rand(2, 5, 64, 64)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (2, 1, 64, 64))
        self.assertTrue(bool((y > 0).all()))
        self.assertTrue(bool((y < 1).all()))


@unittest.skipUnless(arch_sweep.XGB_AVAILABLE, "xgboost not available on this host")
class TestPixelTree(unittest.TestCase):
    def test_pixel_features_shape_finite_equivariant(self):
        from pixel_tree import pixel_features

        rng = np.random.default_rng(0)
        x = rng.random((5, 64, 64)).astype(np.float32)
        feats = pixel_features(x)
        self.assertEqual(feats.shape, (4096, 13))
        self.assertTrue(np.isfinite(feats).all())

        flipped = x[:, ::-1, :].copy()
        feats_flipped = pixel_features(flipped)
        row_perm = feats.reshape(64, 64, 13)[::-1, :, :].reshape(4096, 13)
        np.testing.assert_allclose(feats_flipped, row_perm, atol=1e-5)


class TestFamilyFolds(unittest.TestCase):
    def setUp(self):
        self.keys = [f"{pdk}_{design}_base" for pdk, design in SYNTH_DESIGNS]

    def test_three_families_no_leak(self):
        folds = arch_sweep._make_family_folds(self.keys)
        self.assertEqual(len(folds), 3)
        names = sorted(f["name"] for f in folds)
        self.assertEqual(names, ["aes", "gcd", "ibex"])

        for fold in folds:
            held_out = set(fold["held_out"])
            for key in held_out:
                self.assertEqual(
                    arch_sweep.family_of(arch_sweep._parse_key(key)[1]), fold["name"]
                )
            for other_key in set(self.keys) - held_out:
                self.assertNotEqual(
                    arch_sweep.family_of(arch_sweep._parse_key(other_key)[1]), fold["name"]
                )

    def test_lvt_key_lands_in_base_family(self):
        folds = {f["name"]: f["held_out"] for f in arch_sweep._make_family_folds(self.keys)}
        self.assertIn("fab1_aes_lvt_base", folds["aes"])
        self.assertIn("fab1_aes_base", folds["aes"])


class TestScore(unittest.TestCase):
    def test_perfect_prediction(self):
        rng = np.random.default_rng(0)
        target = rng.random((64, 64)).astype(np.float32)
        scores = arch_sweep._score(target, target)
        self.assertAlmostEqual(scores["heldout_mse"], 0.0, places=10)
        self.assertAlmostEqual(scores["spatial_spearman"], 1.0, places=10)


class TestTrainCLI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self.tmpdir.name, "data")
        os.makedirs(self.data_dir)
        self.keys = _make_synthetic_dataset(self.data_dir)
        self.out = os.path.join(self.tmpdir.name, "arch_sweep_test.json")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run_cli(self, extra_args):
        cmd = [
            sys.executable,
            ARCH_SWEEP_PATH,
            "--data-dir",
            self.data_dir,
            "--out",
            self.out,
        ] + extra_args
        return subprocess.run(cmd, cwd=FLOW_DIR, capture_output=True, text=True)

    def test_unet8_two_epochs_one_record_per_design(self):
        result = self._run_cli(
            [
                "--archs",
                "unet8",
                "--seeds",
                "0",
                "--epochs",
                "2",
                "--batch-size",
                "2",
                "--folds",
                "ibex",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(self.out) as f:
            payload = json.load(f)
        runs = [r for r in payload["runs"] if r["arch"] == "unet8"]
        self.assertEqual(len(runs), 2)  # fab1_ibex_base, fab2_ibex_base
        designs = sorted(r["design"] for r in runs)
        self.assertEqual(designs, ["fab1_ibex_base", "fab2_ibex_base"])

    @unittest.skipUnless(arch_sweep.XGB_AVAILABLE, "xgboost not available on this host")
    def test_xgb_small_run(self):
        result = self._run_cli(
            ["--archs", "xgb", "--seeds", "0", "--folds", "ibex"]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(self.out) as f:
            payload = json.load(f)
        runs = [r for r in payload["runs"] if r["arch"] == "xgb"]
        self.assertEqual(len(runs), 2)

    def test_rerun_same_out_replaces_not_duplicates(self):
        args = ["--archs", "blur", "--seeds", "0", "--folds", "gcd"]
        result1 = self._run_cli(args)
        self.assertEqual(result1.returncode, 0, result1.stderr)
        with open(self.out) as f:
            n1 = len(json.load(f)["runs"])

        result2 = self._run_cli(args)
        self.assertEqual(result2.returncode, 0, result2.stderr)
        with open(self.out) as f:
            n2 = len(json.load(f)["runs"])
        self.assertEqual(n1, n2)


if __name__ == "__main__":
    unittest.main()
