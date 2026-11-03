"""
Tests for the symmetric re-evaluation harness (`training/symmetric_eval.py`).

Run from the flow/ directory:
  python3 util/ml/congestion/tests/test_symmetric_eval.py -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "training"))
sys.path.insert(0, _HERE)

import symmetric_eval as se  # noqa: E402
from test_arch_sweep import _make_synthetic_dataset, _make_synthetic_irdrop_dataset  # noqa: E402

SCRIPT = os.path.join(_HERE, "..", "training", "symmetric_eval.py")
FLOW_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))


class P1Test(unittest.TestCase):
    def test_constant_map_is_zeros(self):
        out = se.p1(np.full((8, 8), 0.37, dtype=np.float32))
        self.assertTrue(np.array_equal(out, np.zeros((8, 8))))

    def test_normal_map_spans_unit_interval_and_keeps_order(self):
        m = np.random.default_rng(0).normal(size=(8, 8)) * 3 + 5
        out = se.p1(m)
        self.assertAlmostEqual(out.min(), 0.0)
        self.assertAlmostEqual(out.max(), 1.0)
        self.assertTrue(np.array_equal(np.argsort(m.ravel()), np.argsort(out.ravel())))

    def test_uses_only_the_map(self):
        m = np.random.default_rng(1).random((8, 8))
        np.testing.assert_allclose(se.p1(m), se.p1(m * 10 + 4))


class AffineTest(unittest.TestCase):
    def test_recovers_known_affine(self):
        m = np.random.default_rng(2).random((4, 8, 8))
        a, b = se.fit_affine(m, 0.6 * m + 0.1)
        self.assertAlmostEqual(a, 0.6, places=9)
        self.assertAlmostEqual(b, 0.1, places=9)

    def test_apply_clips(self):
        out = se.apply_affine(np.array([0.0, 1.0, 2.0]), 1.0, -0.5)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0])

    def test_variant_maps_fit_on_training_designs_only(self):
        rng = np.random.default_rng(3)
        maps = rng.random((4, 8, 8))
        targets = 0.5 * np.stack([se.p1(m) for m in maps]) + 0.2
        train_idx = [0, 1, 2]
        changed = targets.copy()
        changed[3] = 1.0 - changed[3]
        v1 = se.variant_maps(maps, targets, train_idx)["P1+A"]
        v2 = se.variant_maps(maps, changed, train_idx)["P1+A"]
        np.testing.assert_allclose(v1, v2)
        np.testing.assert_allclose(v1, targets, atol=1e-9)


class BlurSelectTest(unittest.TestCase):
    def test_sigma_selected_on_training_designs_only(self):
        rng = np.random.default_rng(4)
        base = rng.random((6, 16, 16))
        maps = {s: np.stack([se.dt._max_normalise(gaussian_filter(b, s)) for b in base]) for s in (1.0, 4.0)}
        targets = maps[4.0].copy()
        train_idx = [0, 1, 2, 3]
        self.assertEqual(se.select_sigma(maps, targets, train_idx, "P0"), 4.0)
        poisoned = targets.copy()
        poisoned[4:] = maps[1.0][4:]
        self.assertEqual(se.select_sigma(maps, poisoned, train_idx, "P0"), 4.0)
        flipped = targets.copy()
        flipped[:4] = maps[1.0][:4]
        self.assertEqual(se.select_sigma(maps, flipped, train_idx, "P0"), 1.0)

    def test_tie_goes_to_smaller_sigma(self):
        maps = {1.0: np.zeros((2, 4, 4)), 2.0: np.zeros((2, 4, 4))}
        self.assertEqual(se.select_sigma(maps, np.zeros((2, 4, 4)), [0, 1], "P0"), 1.0)


class RidgeFeatureTest(unittest.TestCase):
    def test_mm_scaling_is_per_design_and_target_free(self):
        rng = np.random.default_rng(5)
        x_a = rng.random((4, 16, 16)) * 7 + 2
        x_b = rng.random((4, 16, 16)) * 0.1
        fa = se.ridge_design_features(x_a, (0, 1, 2, 3), True)
        self.assertEqual(fa.shape, (256, 20))
        np.testing.assert_allclose(fa.min(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(fa.max(axis=0), 1.0, atol=1e-12)
        fb = se.ridge_design_features(x_b, (0, 1, 2, 3), True)
        np.testing.assert_allclose(fb.max(axis=0), 1.0, atol=1e-12)
        np.testing.assert_allclose(se.ridge_design_features(x_a, (0, 1, 2, 3), True), fa)

    def test_raw_is_not_rescaled(self):
        x = np.random.default_rng(6).random((4, 16, 16)) * 7 + 2
        self.assertGreater(se.ridge_design_features(x, (0,), False).max(), 1.5)


class MergeTest(unittest.TestCase):
    def _rec(self, seed, v):
        return {"track": "thermal", "arch": "fno", "fold": "gcd", "seed": seed, "frac": 1.0, "v": v}

    def test_dedup_later_replaces_earlier(self):
        merged = se.merge_runs([self._rec(0, 1), self._rec(1, 1)], [self._rec(1, 2), self._rec(2, 2)])
        self.assertEqual(len(merged), 3)
        self.assertEqual({r["seed"]: r["v"] for r in merged}, {0: 1, 1: 2, 2: 2})

    def test_fraction_is_part_of_the_key(self):
        a = self._rec(0, 1)
        b = dict(a, frac=0.75)
        self.assertEqual(len(se.merge_runs([a], [b])), 2)

    def test_parse_seeds(self):
        self.assertEqual(se.parse_seeds("0-3,7"), [0, 1, 2, 3, 7])


def _series(values_by_design, rho=0.5):
    return {d: {"mse": list(v), "rho": [rho] * len(v), "mae": [0] * len(v), "top10": [0] * len(v)} for d, v in values_by_design.items()}


class VerdictWiringTest(unittest.TestCase):
    def setUp(self):
        self.designs = [f"p{i}_{fam}_base" for i, fam in enumerate(["aes", "gcd", "ibex", "jpeg", "ethmac", "riscv32i", "dynamic_node", "tinyRocket", "swerv_wrapper"] * 3)]
        self.fam = se.families_of(self.designs)
        self.lines = []

    def run_cmp(self, cand_vals, base_vals, sd=0.0):
        cand = _series({d: [v - sd, v + sd] for d, v in zip(self.designs, cand_vals)})
        base = _series({d: [v] for d, v in zip(self.designs, base_vals)})
        return se.run_comparisons([("c vs b", cand, base)], self.fam, 2, self.lines.append)[0]

    def test_clear_better(self):
        base = np.linspace(0.05, 0.06, 27)
        r = self.run_cmp(base * 0.5, base)
        self.assertEqual(r["verdict"], "Better")
        self.assertEqual(r["wins"], 27)

    def test_clear_worse(self):
        base = np.linspace(0.05, 0.06, 27)
        r = self.run_cmp(base * 2.0, base)
        self.assertEqual(r["verdict"], "Worse")

    def test_indistinguishable(self):
        rng = np.random.default_rng(7)
        base = np.linspace(0.05, 0.06, 27)
        r = self.run_cmp(base + rng.normal(scale=0.005, size=27), base)
        self.assertEqual(r["verdict"], "Indistinguishable")

    def test_holm_is_applied_across_the_family(self):
        base = np.linspace(0.05, 0.06, 27)
        cand = _series({d: [v] for d, v in zip(self.designs, base * 0.5)})
        b = _series({d: [v] for d, v in zip(self.designs, base)})
        out = se.run_comparisons([("a", cand, b), ("b", cand, b)], self.fam, 1, self.lines.append)
        self.assertAlmostEqual(out[0]["adjusted_p"], min(2 * out[0]["raw_p"], 1.0))

    def test_constant_comparator_has_nan_rank_so_strict_verdict_cannot_be_better(self):
        base = np.linspace(0.05, 0.06, 27)
        cand = _series({d: [v] for d, v in zip(self.designs, base * 0.5)})
        const = _series({d: [v] for d, v in zip(self.designs, base)}, rho=float("nan"))
        out = se.run_comparisons([("c vs const", cand, const)], self.fam, 1, self.lines.append)[0]
        self.assertEqual(out["verdict"], "Indistinguishable")
        self.assertEqual(out["verdict_rank_waived"], "Better")

    def test_all_zero_difference_gives_p_one(self):
        s = _series({d: [0.1] for d in self.designs})
        self.assertEqual(se.wilcoxon_p(s, s), 1.0)

    def test_s1_label_rules(self):
        better, worse, ind = {"verdict": "Better"}, {"verdict": "Worse"}, {"verdict": "Indistinguishable"}
        self.assertEqual(se.s1_label(better, better), "survives equal post-processing")
        self.assertEqual(se.s1_label(better, ind), "fno's win over unet32 is mostly output calibration")
        self.assertEqual(se.s1_label(better, worse), "reverses")

    def test_stage_b_label_rules(self):
        self.assertEqual(se.stage_b_label(0.05, 0.04, 0.01)[0], "data-limited")
        self.assertEqual(se.stage_b_label(0.05, 0.049, 0.5)[0], "saturated")
        self.assertEqual(se.stage_b_label(0.05, 0.04, 0.2)[0], "unresolved")
        self.assertEqual(se.stage_b_label(0.05, 0.049, 0.01)[0], "unresolved")


class CliTest(unittest.TestCase):
    def _run(self, track, maker):
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "data")
            os.makedirs(data)
            maker(data)
            out = os.path.join(tmp, "out.json")
            base = [sys.executable, SCRIPT, "--data-dir", data, "--out", out, "--track", track]
            r = subprocess.run(base + ["--stage", "baselines", "--tracks", track], cwd=FLOW_DIR, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            args = base + ["--stage", "train", "--archs", "unet32", "--seeds", "0", "--epochs", "1", "--folds", "gcd"]
            r = subprocess.run(args, cwd=FLOW_DIR, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            r = subprocess.run(args, cwd=FLOW_DIR, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out) as f:
                payload = json.load(f)
            self.assertEqual(len(payload["runs"]), 1)
            run = payload["runs"][0]
            self.assertEqual(sorted(run["designs"]), ["fab1_gcd_base", "fab2_gcd_base"])
            for v in run["designs"].values():
                self.assertEqual(set(v), {"P0", "P1"})
                self.assertIn("heldout_mse_torch", v["P0"])
            names = payload["baselines"][track]
            self.assertEqual(len(names), 16)
            for pred in ("const_mean", "mean_map", "blur3", "blur_sel", "ridge_raw", "ridge_mm"):
                self.assertIn(f"{pred}|P0", names)
                self.assertEqual(len(names[f"{pred}|P0"]), 6)
            self.assertIn("blur_sel|P1+A", names)
            self.assertEqual(len(payload["timing"]), 3)

    def test_thermal_end_to_end(self):
        self._run("thermal", _make_synthetic_dataset)

    def test_irdrop_end_to_end(self):
        self._run("irdrop", _make_synthetic_irdrop_dataset)


if __name__ == "__main__":
    unittest.main()
