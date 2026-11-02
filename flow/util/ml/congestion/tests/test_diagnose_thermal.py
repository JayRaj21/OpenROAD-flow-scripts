"""
Tests for the thermal diagnostics harness (`training/diagnose_thermal.py`).

Run from the flow/ directory:
  python3 util/ml/congestion/tests/test_diagnose_thermal.py -v
"""

import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "training"))

import diagnose_thermal as dt  # noqa: E402


class RidgeTest(unittest.TestCase):
    def test_recovers_known_linear_relation(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(5000, 4)) * np.array([1.0, 5.0, 0.2, 3.0]) + 2.0
        w0 = np.array([0.3, -0.1, 1.5, 0.05])
        y = X @ w0 + 0.7
        model = dt.ridge_fit(X, y, alpha=1e-9)
        pred = X @ w0 + 0.7
        unclipped = ((X - model["mu"]) / model["sd"]) @ model["w"] + model["b"]
        np.testing.assert_allclose(unclipped, pred, atol=1e-6)

    def test_constant_feature_is_harmless_and_clip_applies(self):
        rng = np.random.default_rng(1)
        X = np.concatenate([rng.normal(size=(200, 1)), np.ones((200, 1))], axis=1)
        y = 0.5 + 10.0 * X[:, 0]
        model = dt.ridge_fit(X, y, alpha=1.0)
        self.assertTrue(np.all(np.isfinite(model["w"])))
        pred = dt.ridge_predict(model, X)
        self.assertTrue(pred.min() >= 0.0 and pred.max() <= 1.0)

    def test_intercept_unpenalised(self):
        X = np.random.default_rng(2).normal(size=(100, 3))
        y = np.full(100, 0.4)
        model = dt.ridge_fit(X, y, alpha=1e6)
        self.assertAlmostEqual(model["b"], 0.4)


class CalibrationTest(unittest.TestCase):
    def test_fit_affine_recovers_a_b(self):
        rng = np.random.default_rng(3)
        m = rng.random((4, 8, 8))
        t = 0.6 * m + 0.1
        a, b = dt.fit_affine(m, t)
        self.assertAlmostEqual(a, 0.6, places=9)
        self.assertAlmostEqual(b, 0.1, places=9)

    def test_calibrated_variant_uses_training_designs_only(self):
        rng = np.random.default_rng(4)
        base = rng.random((3, 8, 8))
        mm = np.stack([dt._minmax_normalise(b) for b in base])
        targets = 0.5 * mm + 0.2
        targets[2] = 1.0 - mm[2]
        pred = dt.variant_predictions(base, targets, [0, 1], "calibrated")
        np.testing.assert_allclose(pred[:2], targets[:2], atol=1e-9)
        self.assertGreater(np.abs(pred[2] - targets[2]).max(), 0.1)

    def test_raw_max_and_minmax(self):
        b = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        np.testing.assert_allclose(dt.variant_predictions(b, b, [0], "raw_max")[0], b[0] / 4.0)
        np.testing.assert_allclose(dt.variant_predictions(b, b, [0], "minmax")[0], (b[0] - 1.0) / 3.0)
        flat = np.full((1, 2, 2), 2.0)
        self.assertEqual(dt.variant_predictions(flat, flat, [0], "minmax").max(), 0.0)


class SigmaSelectionTest(unittest.TestCase):
    def test_choose_sigma_ties_to_smaller(self):
        self.assertEqual(dt.choose_sigma({1.0: 0.2, 2.0: 0.1, 4.0: 0.1}), 2.0)

    def test_uses_training_best_not_heldout_best(self):
        rng = np.random.default_rng(5)
        a1 = rng.random((3, 64, 64))
        a8 = rng.random((3, 64, 64))
        maps = {1.0: a1, 8.0: a8}
        targets = np.stack(
            [
                dt._minmax_normalise(a1[0]),
                dt._minmax_normalise(a1[1]),
                dt._minmax_normalise(a8[2]),
            ]
        )
        keys = ["t0", "t1", "h"]
        folds = [{"name": "f", "held_out": ["h"]}]
        res = dt.run_blur_study(maps, targets, keys, folds)
        chosen = res["selected"]["minmax"]["h"]["sigma"]
        heldout_best = min(
            (1.0, 8.0), key=lambda s: res["sweep"]["minmax"][str(s)]["h"]["heldout_mse"]
        )
        self.assertEqual(heldout_best, 8.0)
        self.assertEqual(chosen, 1.0)
        self.assertEqual(
            res["selected"]["minmax"]["h"]["heldout_mse"],
            res["sweep"]["minmax"]["1.0"]["h"]["heldout_mse"],
        )


class NestingTest(unittest.TestCase):
    KEYS = [f"pdk{i}_d{i}_base" for i in range(23)]

    def test_nested_and_no_leak(self):
        held = {"pdk0_d0_base", "pdk5_d5_base", "pdk9_d9_base"}
        train = [k for k in self.KEYS if k not in held]
        for fold_index, seed in [(0, 0), (3, 1), (8, 2), (4, 0)]:
            s25 = dt.nested_train_keys(train, fold_index, seed, 0.25)
            s50 = dt.nested_train_keys(train, fold_index, seed, 0.5)
            s75 = dt.nested_train_keys(train, fold_index, seed, 0.75)
            s100 = dt.nested_train_keys(train, fold_index, seed, 1.0)
            self.assertLess(len(s25), len(s50))
            self.assertLess(len(s50), len(s75))
            self.assertTrue(set(s25) <= set(s50) <= set(s75) <= set(s100))
            self.assertEqual(set(s100), set(train))
            for s in (s25, s50, s75, s100):
                self.assertFalse(set(s) & held)
                self.assertEqual(len(s), len(set(s)))

    def test_deterministic_and_seed_dependent(self):
        a = dt.nested_train_keys(self.KEYS, 1, 0, 0.5)
        self.assertEqual(a, dt.nested_train_keys(self.KEYS, 1, 0, 0.5))
        self.assertNotEqual(a, dt.nested_train_keys(self.KEYS, 1, 1, 0.5))

    def test_restricted_dataset_view(self):
        class Base:
            keys = ["a", "b", "c", "d"]

            def __getitem__(self, i):
                return self.keys[i]

        view = dt.RestrictedDataset(Base(), ["d", "b"])
        self.assertEqual(view.keys, ["b", "d"])
        self.assertEqual(len(view), 2)
        self.assertEqual([view[i] for i in range(2)], ["b", "d"])


class ReadingRulesTest(unittest.TestCase):
    def rules(self, m_oracle=1.0, lc75=1.0, lc100=1.0, m_ridge=1.0, m_cal3=1.0, m_raw3=1.0):
        return dt.reading_rules(1.0, m_oracle, lc75, lc100, m_ridge, m_cal3, m_raw3)

    def test_r1(self):
        self.assertIn("bottleneck", self.rules(m_oracle=0.5)["R1"]["label"])
        self.assertIn("partially", self.rules(m_oracle=0.6)["R1"]["label"])
        self.assertIn("does not help", self.rules(m_oracle=0.8)["R1"]["label"])
        self.assertIn("does not help", self.rules(m_oracle=1.3)["R1"]["label"])
        self.assertIn("partially", self.rules(m_oracle=0.51)["R1"]["label"])
        self.assertIn("partially", self.rules(m_oracle=0.79)["R1"]["label"])

    def test_r2(self):
        self.assertIn("data-limited", self.rules(lc100=0.9)["R2"]["label"])
        self.assertIn("data-limited", self.rules(lc100=0.5)["R2"]["label"])
        self.assertIn("ambiguous", self.rules(lc100=0.91)["R2"]["label"])
        self.assertIn("ambiguous", self.rules(lc100=0.95)["R2"]["label"])
        self.assertIn("saturated", self.rules(lc100=0.951)["R2"]["label"])
        self.assertIn("saturated", self.rules(lc100=1.05)["R2"]["label"])

    def test_r3(self):
        self.assertIn("linear filter bank", self.rules(m_ridge=1.1)["R3"]["label"])
        self.assertIn("linear filter bank", self.rules(m_ridge=0.9)["R3"]["label"])
        self.assertIn("beats the linear", self.rules(m_ridge=1.11)["R3"]["label"])
        self.assertIn("linear filter bank beats", self.rules(m_ridge=0.89)["R3"]["label"])

    def test_r4(self):
        self.assertIn("calibrated blur beats unet32", self.rules(m_cal3=0.9)["R4"]["label"])
        self.assertIn("unet32 still beats", self.rules(m_cal3=1.1)["R4"]["label"])
        out = self.rules(m_cal3=0.9, m_raw3=1.2)["R4"]
        self.assertEqual((out["m_calibrated_sigma3"], out["m_raw_sigma3"]), (0.9, 1.2))


if __name__ == "__main__":
    unittest.main()
