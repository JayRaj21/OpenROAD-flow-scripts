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


def _make_synthetic_irdrop_dataset(out_dir: str, seed: int = 0) -> list[str]:
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

        stripe_density = gaussian_filter(
            rng.random((GRID, GRID)).astype(np.float32), sigma=2
        )
        via_density = gaussian_filter(
            rng.random((GRID, GRID)).astype(np.float32), sigma=2
        )
        irdrop_map = (gaussian_filter(cell, sigma=5) * 0.05).astype(np.float32)
        irdrop_map += rng.random((GRID, GRID)).astype(np.float32) * 0.005
        voltage_map = (0.8 - irdrop_map).astype(np.float32)
        current_density_proxy = cell.astype(np.float32)
        np.savez(
            os.path.join(out_dir, f"{key}_irdrop_labels.npz"),
            irdrop_map=irdrop_map,
            voltage_map=voltage_map,
            current_density_proxy=current_density_proxy,
            stripe_density=stripe_density.astype(np.float32),
            via_density=via_density.astype(np.float32),
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


class TestBlurPred(unittest.TestCase):
    def test_thermal_blur_equals_channel_4(self):
        rng = np.random.default_rng(0)
        cell = rng.random((GRID, GRID)).astype(np.float32)
        blurred = gaussian_filter(cell, sigma=3)
        blurred = blurred / blurred.max()
        x = np.stack(
            [cell, rng.random((GRID, GRID)), rng.random((GRID, GRID)), rng.random((GRID, GRID)), blurred]
        ).astype(np.float32)
        np.testing.assert_array_equal(arch_sweep._blur_cell_density(x), x[4])
        np.testing.assert_array_equal(arch_sweep._blur_pred("thermal", x), x[4])

    def test_irdrop_blur_is_smoothed_cell_density_not_channel_4(self):
        rng = np.random.default_rng(0)
        x = rng.random((6, GRID, GRID)).astype(np.float32)
        expected = gaussian_filter(x[0], 3)
        expected = expected / expected.max()
        pred = arch_sweep._blur_pred("irdrop", x)
        np.testing.assert_allclose(pred, expected, atol=1e-6)
        self.assertFalse(np.array_equal(pred, x[4]))


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


class TestIRDropTrainAndAnalyzeCLI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self.tmpdir.name, "data")
        os.makedirs(self.data_dir)
        self.keys = _make_synthetic_irdrop_dataset(self.data_dir)
        self.out = os.path.join(self.tmpdir.name, "irdrop_arch_sweep_test.json")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run_cli(self, extra_args):
        cmd = [
            sys.executable,
            ARCH_SWEEP_PATH,
            "--track",
            "irdrop",
            "--data-dir",
            self.data_dir,
            "--out",
            self.out,
        ] + extra_args
        return subprocess.run(cmd, cwd=FLOW_DIR, capture_output=True, text=True)

    def test_train_and_analyze_end_to_end(self):
        result = self._run_cli(
            [
                "--archs",
                "unet32,unet8,blur",
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

        result = self._run_cli(["--analyze"])
        self.assertEqual(result.returncode, 0, result.stderr)
        summary_path = os.path.splitext(self.out)[0] + "_summary.md"
        with open(summary_path) as f:
            summary = f.read()
        self.assertIn("fill_dominated", summary)
        self.assertIn("log10 worst_drop_mv", summary)

        report_data_path = os.path.splitext(self.out)[0] + "_report_data.json"
        with open(report_data_path) as f:
            report_data = json.load(f)
        self.assertEqual(report_data["track"], "irdrop")
        self.assertTrue(report_data["vs_unet32"])
        self.assertTrue(report_data["vs_blur"])
        for row in report_data["vs_unet32"] + report_data["vs_blur"]:
            self.assertIn(row["verdict"], ("Better", "Worse", "Indistinguishable"))

        html_out = os.path.join(self.tmpdir.name, "report.html")
        view_report_path = os.path.join(os.path.dirname(ARCH_SWEEP_PATH), "view_report.py")
        result = subprocess.run(
            [sys.executable, view_report_path, report_data_path, "--out", html_out],
            cwd=FLOW_DIR,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(html_out) as f:
            page = f.read()
        self.assertIn("IR-drop", page)
        self.assertIn("<svg", page)
        for cand_row in report_data["vs_unet32"]:
            self.assertIn(f"{cand_row['median_delta']:+.4f}", page)


class TestVerdictReturnsStats(unittest.TestCase):
    def test_verdict_stats_match_printed_values(self):
        deltas = {"a": -0.5, "b": -0.4, "c": 0.1}
        delta_rhos = {"a": 0.01, "b": 0.0, "c": -0.01}
        sds_c = {"a": 0.01, "b": 0.01, "c": 0.01}
        sds_b = {"a": 0.01, "b": 0.01, "c": 0.01}
        families = {"a": "fam1", "b": "fam2", "c": "fam3"}
        emitted = []
        verdict, stats_out = arch_sweep._verdict(
            deltas, delta_rhos, sds_c, sds_b, families, 1, 0.5, emitted.append, "test"
        )
        self.assertEqual(stats_out["verdict"], verdict)
        self.assertAlmostEqual(stats_out["median_delta"], -0.4)
        self.assertEqual(stats_out["wins"], 2)
        self.assertEqual(stats_out["losses"], 1)
        self.assertEqual(stats_out["n_families"], 3)
        self.assertEqual(len(emitted), 1)  # unchanged printed-line behavior


if __name__ == "__main__":
    unittest.main()
