"""
Tests for the FNO production-readiness harness (`training/readiness_eval.py`).

Run from the flow/ directory:
  python3 util/ml/congestion/tests/test_readiness_eval.py -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAINING = os.path.join(_HERE, "..", "training")
sys.path.insert(0, _TRAINING)
sys.path.insert(0, _HERE)

from test_arch_sweep import _make_synthetic_dataset  # noqa: E402

READINESS_EVAL_PATH = os.path.join(_TRAINING, "readiness_eval.py")
FLOW_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))


def _run_cli(extra_args, cwd=FLOW_DIR):
    cmd = [sys.executable, READINESS_EVAL_PATH] + extra_args
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


class TestTrainCLI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self.tmpdir.name, "data")
        os.makedirs(self.data_dir)
        self.keys = _make_synthetic_dataset(self.data_dir)
        self.out = os.path.join(self.tmpdir.name, "readiness_test.json")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_fno_two_epochs_schema(self):
        result = _run_cli(
            [
                "--data-dir", self.data_dir,
                "--out", self.out,
                "--archs", "fno",
                "--seeds", "0",
                "--epochs", "2",
                "--batch-size", "2",
                "--folds", "ibex",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(self.out) as f:
            payload = json.load(f)
        runs = [r for r in payload["runs"] if r["arch"] == "fno"]
        self.assertEqual(len(runs), 2)  # fab1_ibex_base, fab2_ibex_base
        for r in runs:
            self.assertEqual(r["track"], "thermal")
            self.assertEqual(r["protocol"], "lofo")
            self.assertEqual(r["recipe"], "production")
            self.assertEqual(r["lr"], 1e-3)
            self.assertIn("best_epoch", r)
            self.assertIn("best_val_mse", r)
            self.assertIsInstance(r["heldout_mse"], float)

    def test_two_folds_two_seeds(self):
        result = _run_cli(
            [
                "--data-dir", self.data_dir,
                "--out", self.out,
                "--archs", "unet32",
                "--seeds", "0,1",
                "--epochs", "2",
                "--batch-size", "2",
                "--folds", "ibex,gcd",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(self.out) as f:
            payload = json.load(f)
        runs = [r for r in payload["runs"] if r["arch"] == "unet32"]
        # ibex: 2 designs x 2 seeds = 4; gcd: 2 designs x 2 seeds = 4
        self.assertEqual(len(runs), 8)

    def test_blur_label_override_and_no_seed_duplication(self):
        result = _run_cli(
            [
                "--data-dir", self.data_dir,
                "--out", self.out,
                "--archs", "blur",
                "--seeds", "0,1,2",
                "--folds", "gcd",
                "--label", "blur_relabel",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(self.out) as f:
            payload = json.load(f)
        runs = [r for r in payload["runs"] if r["arch"] == "blur_relabel"]
        self.assertEqual(len(runs), 2)  # one run per held-out design, no seed loop

    def test_unknown_arch_rejected(self):
        result = _run_cli(
            [
                "--data-dir", self.data_dir,
                "--out", self.out,
                "--archs", "unet16",
                "--seeds", "0",
                "--folds", "ibex",
            ]
        )
        self.assertNotEqual(result.returncode, 0)

    def test_refuses_arch_sweep_json(self):
        forbidden = os.path.join(
            FLOW_DIR, "util", "ml", "congestion", "experiments", "arch_sweep.json"
        )
        result = _run_cli(
            [
                "--data-dir", self.data_dir,
                "--out", forbidden,
                "--archs", "blur",
                "--seeds", "0",
                "--folds", "ibex",
            ]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing", result.stderr)

    def test_refuses_irdrop_arch_sweep_json(self):
        forbidden = os.path.join(
            FLOW_DIR, "util", "ml", "congestion", "experiments", "irdrop_arch_sweep.json"
        )
        result = _run_cli(
            [
                "--data-dir", self.data_dir,
                "--out", forbidden,
                "--archs", "blur",
                "--seeds", "0",
                "--folds", "ibex",
            ]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing", result.stderr)

    def test_refuses_thermal_best_checkpoint_path(self):
        forbidden = os.path.join(self.tmpdir.name, "thermal_best.pt")
        result = _run_cli(
            [
                "--data-dir", self.data_dir,
                "--out", forbidden,
                "--archs", "blur",
                "--seeds", "0",
                "--folds", "ibex",
            ]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing", result.stderr)


class TestCheckIdentical(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self.tmpdir.name, "data")
        os.makedirs(self.data_dir)
        _make_synthetic_dataset(self.data_dir)
        self.out_a = os.path.join(self.tmpdir.name, "a.json")
        self.out_b = os.path.join(self.tmpdir.name, "b.json")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _train(self, out):
        result = _run_cli(
            [
                "--data-dir", self.data_dir,
                "--out", out,
                "--archs", "unet32",
                "--seeds", "0",
                "--epochs", "2",
                "--batch-size", "2",
                "--folds", "ibex",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_identical_reruns_match(self):
        self._train(self.out_a)
        self._train(self.out_b)
        result = _run_cli(
            [
                "--check-identical", self.out_a, self.out_b,
                "--archs", "unet32",
                "--seeds", "0",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0 mismatched", result.stdout)

    def test_deliberate_mismatch_reported_and_exit1(self):
        self._train(self.out_a)
        with open(self.out_a) as f:
            payload = json.load(f)
        for r in payload["runs"]:
            r["heldout_mse"] = r["heldout_mse"] + 1.0
        with open(self.out_b, "w") as f:
            json.dump(payload, f)

        result = _run_cli(
            [
                "--check-identical", self.out_a, self.out_b,
                "--archs", "unet32",
                "--seeds", "0",
            ]
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("MISMATCH", result.stdout)
        self.assertNotIn("0 mismatched", result.stdout)


class TestWireRCReport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.fixture = os.path.join(self.tmpdir.name, "fixture.json")

        e_key = "asap7_aes_base"  # in E
        l_key = "asap7_ibex_base"  # in L_asap7
        o_key = "nangate45_ibex_base"  # in O

        designs = [
            {"key": k, "design": d, "track": "thermal"}
            for k, d in [(e_key, "aes"), (l_key, "ibex"), (o_key, "ibex")]
        ]
        runs = []
        for key, unet_mse, fno_mse in [
            (e_key, 0.05, 0.05),
            (l_key, 0.05, 0.20),
            (o_key, 0.05, 0.04),
        ]:
            for arch, mse in [("unet32", unet_mse), ("fno", fno_mse)]:
                for seed in range(2):
                    runs.append(
                        {
                            "track": "thermal",
                            "protocol": "lofo",
                            "arch": arch,
                            "fold": "whatever",
                            "design": key,
                            "seed": seed,
                            "heldout_mse": mse,
                        }
                    )
        with open(self.fixture, "w") as f:
            json.dump({"designs": designs, "runs": runs}, f)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_groups_applied_correctly(self):
        result = _run_cli(["--wirerc-report", "--report-from", self.fixture])
        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.stdout
        self.assertIn("E (n=", out)
        self.assertIn("L_asap7 (n=", out)
        self.assertIn("L_sky (n=", out)
        self.assertIn("L (n=", out)
        self.assertIn("O (n=", out)
        self.assertIn("riscv32i", out)
        # E group: fno == unet32 (delta ~0); L_asap7: fno much worse.
        self.assertIn("E (n=1): fno-unet32 mean_delta=0.000000", out)
        self.assertIn("L_asap7 (n=1): fno-unet32 mean_delta=0.150000", out)
        # G3 is signed: a harmful (positive) delta that exceeds the noise
        # floor must print as a fail, and a favorable (negative) delta
        # must never print as a fail regardless of its magnitude -- this
        # is the comparison direction a prior version got backwards by
        # comparing abs(delta) instead of the signed value.
        for line in out.splitlines():
            if line.strip().startswith("L_asap7 (n=") and "mean_delta" in line:
                self.assertIn(">T (fail)", line)
            if line.strip().startswith("O (n=") and "mean_delta" in line:
                self.assertIn("<=T (pass)", line)


if __name__ == "__main__":
    unittest.main()
