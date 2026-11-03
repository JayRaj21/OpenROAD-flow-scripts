"""
Tests for the input-normalisation experiment (`training/input_norm_eval.py`).

Run from the flow/ directory:
  python3 util/ml/congestion/tests/test_input_norm_eval.py -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "training"))
sys.path.insert(0, _HERE)

import input_norm_eval as ine  # noqa: E402
import symmetric_eval as se  # noqa: E402
from irdrop_dataset import IRDropDataset  # noqa: E402
from test_arch_sweep import _make_synthetic_dataset, _make_synthetic_irdrop_dataset  # noqa: E402
from thermal_dataset import ThermalDataset  # noqa: E402

FAMILIES = ["aes", "dynamic_node", "ethmac", "gcd", "ibex", "jpeg", "riscv32i", "swerv", "tinyRocket"]


def _raw(c, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((c, 16, 16)) * rng.uniform(0.5, 300, size=(c, 1, 1))).astype(np.float32)


class MinMaxTest(unittest.TestCase):
    def test_each_channel_spans_unit_interval(self):
        out = ine.minmax_channels(_raw(5))
        self.assertTrue(np.allclose(out.min(axis=(1, 2)), 0.0))
        self.assertTrue(np.allclose(out.max(axis=(1, 2)), 1.0))

    def test_constant_channel_is_zeros(self):
        x = _raw(5)
        x[2] = 0.7
        out = ine.minmax_channels(x)
        self.assertTrue(np.array_equal(out[2], np.zeros((16, 16), dtype=np.float32)))
        self.assertTrue(np.allclose(out[3].max(), 1.0))

    def test_channels_are_independent(self):
        x = _raw(5)
        y = x.copy()
        y[1] = y[1] * 7 + 3
        a, b = ine.minmax_channels(x), ine.minmax_channels(y)
        for c in (0, 2, 3, 4):
            self.assertTrue(np.array_equal(a[c], b[c]))
        self.assertTrue(np.allclose(a[1], b[1], atol=1e-6))


class V2ChannelsTest(unittest.TestCase):
    def test_channel_counts(self):
        self.assertEqual(ine.in_channels("thermal", "V0"), 5)
        self.assertEqual(ine.in_channels("thermal", "V1"), 5)
        self.assertEqual(ine.in_channels("thermal", "V2"), 21)
        self.assertEqual(ine.in_channels("irdrop", "V0"), 6)
        self.assertEqual(ine.in_channels("irdrop", "V1"), 6)
        self.assertEqual(ine.in_channels("irdrop", "V2"), 30)

    def test_transform_shapes_match_counts(self):
        for track, c in (("thermal", 5), ("irdrop", 6)):
            x = _raw(c)
            for v in ine.VARIANTS:
                self.assertEqual(ine.transform_input(x, track, v).shape[0], ine.in_channels(track, v))

    def test_v2_does_not_blur_thermal_blur_channel(self):
        x = _raw(5)
        x[4] = np.random.default_rng(9).random((16, 16)) * 1e3
        y = x.copy()
        y[4] = 5.0
        self.assertTrue(np.array_equal(ine.transform_input(x, "thermal", "V2")[5:], ine.transform_input(y, "thermal", "V2")[5:]))

    def test_v0_is_identity(self):
        x = _raw(5)
        self.assertIs(ine.transform_input(x, "thermal", "V0"), x)


class FlipTest(unittest.TestCase):
    def test_v1_v2_commute_with_flips(self):
        for track, c in (("thermal", 5), ("irdrop", 6)):
            x = _raw(c, seed=3)
            for v in ("V1", "V2"):
                base = ine.transform_input(x, track, v)
                for axis in (1, 2):
                    flipped_first = ine.transform_input(np.flip(x, axis=axis).copy(), track, v)
                    self.assertLess(float(np.abs(flipped_first - np.flip(base, axis=axis)).max()), 1e-6)

    def test_transform_signature_reads_only_x(self):
        import inspect

        self.assertEqual(list(inspect.signature(ine.transform_input).parameters), ["x", "track", "variant"])


class DatasetWrapperTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _check(self, track, maker, cls, key):
        maker(self.tmp.name)
        base = cls(self.tmp.name, augment=False)
        for v in ine.VARIANTS:
            ds = ine.NormDataset(base, track, v, augment=False)
            self.assertEqual(ds.keys, base.keys)
            self.assertEqual(len(ds), len(base))
            item = ds[0]
            self.assertEqual(item["x"].shape[0], ine.in_channels(track, v))
            self.assertTrue(torch.equal(item[key], base[0][key]))
        v0 = ine.NormDataset(base, track, "V0", augment=False)
        self.assertTrue(torch.equal(v0[1]["x"], base[1]["x"]))

    def test_thermal(self):
        self._check("thermal", _make_synthetic_dataset, ThermalDataset, "thermal")

    def test_irdrop(self):
        self._check("irdrop", _make_synthetic_irdrop_dataset, IRDropDataset, "irdrop")

    def test_augmented_flips_x_and_target_together_and_matches_base_rng(self):
        _make_synthetic_dataset(self.tmp.name)
        base = ThermalDataset(self.tmp.name, augment=False)
        aug_base = ThermalDataset(self.tmp.name, augment=True)
        ds = ine.NormDataset(base, "thermal", "V0", augment=True)
        for i in range(len(base)):
            torch.manual_seed(i)
            a = aug_base[i]
            torch.manual_seed(i)
            b = ds[i]
            self.assertTrue(torch.equal(a["x"], b["x"]))
            self.assertTrue(torch.equal(a["thermal"], b["thermal"]))

    def test_target_is_not_read_by_inputs(self):
        _make_synthetic_dataset(self.tmp.name)
        base = ThermalDataset(self.tmp.name, augment=False)
        before = ine.NormDataset(base, "thermal", "V2", augment=False)[0]["x"].clone()
        base2 = ThermalDataset(self.tmp.name, augment=False)
        orig = base2.__getitem__

        def altered(i):
            item = dict(orig(i))
            item["thermal"] = item["thermal"] * 0 + 0.123
            return item

        base2.__getitem__ = altered
        after = ine.NormDataset(base2, "thermal", "V2", augment=False)[0]["x"]
        self.assertTrue(torch.equal(before, after))

    def test_rejects_augmenting_base(self):
        _make_synthetic_dataset(self.tmp.name)
        with self.assertRaises(ValueError):
            ine.NormDataset(ThermalDataset(self.tmp.name, augment=True), "thermal", "V1", True)

    def test_channels_override_restores(self):
        orig = se.TRACKS["thermal"]["in_channels"]
        with ine.channels_override("thermal", 21):
            self.assertEqual(se.TRACKS["thermal"]["in_channels"], 21)
        self.assertEqual(se.TRACKS["thermal"]["in_channels"], orig)


class MergeTest(unittest.TestCase):
    def rec(self, **kw):
        r = {"track": "thermal", "arch": "fno", "variant": "V1", "fold": "gcd", "seed": 0, "tag": "old"}
        r.update(kw)
        return r

    def test_dedup_keyed_by_variant(self):
        old = [self.rec(), self.rec(variant="V2")]
        merged = ine.merge_runs(old, [self.rec(tag="new"), self.rec(seed=1)])
        self.assertEqual(len(merged), 3)
        self.assertEqual([r["tag"] for r in merged if r["variant"] == "V1" and r["seed"] == 0], ["new"])

    def test_merge_idempotent(self):
        runs = [self.rec(), self.rec(seed=1)]
        self.assertEqual(len(ine.merge_runs(ine.merge_runs([], runs), runs)), 2)


def _series(values_by_design, rho):
    return {d: {"mse": [v] * 5, "rho": [rho] * 5, "mae": [v] * 5, "top10": [0.0] * 5} for d, v in values_by_design.items()}


def _designs():
    return {f"{pdk}_{fam}_base": 0.02 + 0.001 * i for i, (pdk, fam) in enumerate((p, f) for p in ("asap7", "nangate45", "sky130hd") for f in FAMILIES)}


class WiringTest(unittest.TestCase):
    def setUp(self):
        self.base = _designs()
        self.table = {}

    def get(self, track, arch, variant, pp):
        return self.table[(arch, variant, pp)]

    def free(self, track, name):
        return self.table[name]

    def test_n1_better_worse_indistinguishable(self):
        base = _series(self.base, 0.5)
        for arch in se.NEURAL:
            for pp in se.NEURAL_PPS:
                self.table[(arch, "V0", pp)] = base
        self.table[("unet32", "V1", "P0")] = _series({k: v * 0.5 for k, v in self.base.items()}, 0.6)
        self.table[("unet32", "V2", "P0")] = _series({k: v * 2.0 for k, v in self.base.items()}, 0.4)
        self.table[("fno", "V1", "P0")] = base
        self.table[("fno", "V2", "P0")] = base
        specs = ine.n1_specs(self.get, "thermal")
        self.assertEqual([s[0] for s in specs][:2], ["unet32 V1 vs unet32 V0 (P0)", "unet32 V2 vs unet32 V0 (P0)"])
        fam = se.families_of(base)
        out = se.run_comparisons(specs, fam, 5, lambda s: None)
        verdicts = {c["label"]: c["verdict"] for c in out}
        self.assertEqual(verdicts["unet32 V1 vs unet32 V0 (P0)"], "Better")
        self.assertEqual(verdicts["unet32 V2 vs unet32 V0 (P0)"], "Worse")
        self.assertEqual(verdicts["fno V1 vs fno V0 (P0)"], "Indistinguishable")
        self.assertEqual([ine.n1_label(v) for v in ("Better", "Worse", "Indistinguishable")], ["helps", "hurts", "no demonstrated effect"])
        self.assertEqual(out[0]["n_families"], 9)

    def test_n2_specs_and_labels(self):
        base = _series(self.base, 0.5)
        for arch in se.NEURAL:
            for v in ine.NEW_VARIANTS:
                self.table[(arch, v, "P1")] = base
                self.table[(arch, v, "P0")] = base
        for b in ("ridge_mm|P1", "blur_sel|P1", "ridge_raw|P0", "mean_map|P0"):
            self.table[b] = base
        th = ine.n2_specs(self.get, self.free, "thermal")
        ir = ine.n2_specs(self.get, self.free, "irdrop")
        self.assertEqual(len(th), 8)
        self.assertEqual(len(ir), 8)
        self.assertTrue(all("ridge_mm|P1" in s[0] or "blur_sel|P1" in s[0] for s in th))
        self.assertTrue(all("ridge_raw|P0" in s[0] or "mean_map|P0" in s[0] for s in ir))
        self.assertEqual([ine.n2_label(v) for v in ("Better", "Indistinguishable", "Worse")], ["matches or beats", "matches or beats", "still behind"])

    def test_n3_specs(self):
        for arch in se.NEURAL:
            for v in ine.VARIANTS:
                for pp in se.NEURAL_PPS:
                    self.table[(arch, v, pp)] = _series(self.base, 0.5)
        specs = ine.n3_specs(self.get)
        self.assertEqual(len(specs), 6)
        self.assertEqual(specs[0][0], "fno vs unet32 V0 (P0)")


if __name__ == "__main__":
    unittest.main()
