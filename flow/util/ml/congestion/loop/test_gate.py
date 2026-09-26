"""
Self-test for the Stage B gate tooling (plain Python, no pytest).

Run from flow/:  python3 util/ml/congestion/loop/test_gate.py [--keep]

Constructed cases feed gate_rank's pure scoring functions directly. The
end-to-end cases build synthetic variant files, train tiny 2-epoch
leave-family-out checkpoints on the real base data (read-only, a smoke test
only, trained lazily by the first case that needs one) and run gate_rank and
run_gate.sh on them, including runs whose ground truth is an affine function of
the model's own prediction (a real PASS) or of another variant's prediction
(a real FAIL). run_gate.sh is exercised in throwaway copies of the flow tree
with a stubbed trainer.

Every run uses a fresh temporary directory (honouring TMPDIR), removed at the
end unless --keep is given. Exit status: 0 if every case passes, 1 if any case
fails, 4 if a setup step (for example training a smoke checkpoint) crashed; a
setup crash prints "ERROR: setup failed: ..." so it is never mistaken for a
passing or a failing case.
"""

import atexit
import contextlib
import io
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gate_rank as gr  # noqa: E402
import train_lodo  # noqa: E402
from thermal_metrics import blur_proxy, shape_metrics  # noqa: E402
from train_lodo import sha256_file  # noqa: E402
from unet import CongestionUNet  # noqa: E402

KEEP = "--keep" in sys.argv[1:]
SCRATCH = tempfile.mkdtemp(prefix="gate_test_")
if KEEP:
    print(f"keeping scratch directory {SCRATCH}")
else:
    atexit.register(shutil.rmtree, SCRATCH, ignore_errors=True)

CONG = os.path.normpath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(CONG, "data")
TRAIN = os.path.join(HERE, "train_lodo.py")
GATE = os.path.join(HERE, "gate_rank.py")
RUN_GATE = os.path.join(HERE, "run_gate.sh")
GRID = [0.05, 0.15, 0.30, 0.45, 0.60, 0.75]
IBEX3 = "sky130hd/ibex,nangate45/ibex,asap7/ibex"

results = []


class SetupError(BaseException):
    """A setup step crashed; not a test failure and not a pass."""


def report():
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n      {msg}" if msg else ""))
    print(f"{sum(ok for _, ok, _ in results)}/{len(results)} cases passed")


def case(name):
    def deco(fn):
        try:
            fn()
            results.append((name, True, ""))
        except SetupError as e:
            report()
            print(f"ERROR: setup failed: {e}")
            sys.exit(4)
        except Exception as e:  # noqa: BLE001 - a failing case must be reported, not hidden
            results.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return deco


def make_rows(truth, ptp, pred, proxy, addons=GRID):
    return [
        {"tag": f"dn_{round(a * 100):03d}", "addon": a, "truth_top10": t, "truth_ptp_c": p,
         "pred_top10": q, "proxy_top10": x}
        for a, t, p, q, x in zip(addons, truth, ptp, pred, proxy)
    ]


TRUTH = [1.50, 1.55, 1.60, 1.70, 1.80, 1.90]
PTP = [40.0, 44.0, 48.0, 52.0, 56.0, 60.0]
PRED = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
PROXY_BAD = [1.3, 1.1, 1.5, 1.0, 1.4, 1.2]


def good(**over):
    kw = {"truth": TRUTH, "ptp": PTP, "pred": PRED, "proxy": PROXY_BAD}
    kw.update(over)
    return make_rows(kw["truth"], kw["ptp"], kw["pred"], kw["proxy"])


def four(rows):
    return {f"d{i}": gr.score_design(rows) for i in range(4)}


def three_bad_one_good(bad_rows):
    s = {f"d{i}": gr.score_design(bad_rows) for i in range(3)}
    s["d3"] = gr.score_design(good())
    return s


@case("constructed PASS")
def _():
    v = gr.evaluate_gate(four(good()))
    assert v["overall"] == "PASS" and (v["g1"], v["g2"], v["g3"]) == ("PASS",) * 3, v


@case("FAIL G1: spread too small (ptp)")
def _():
    v = gr.evaluate_gate(three_bad_one_good(good(ptp=[50.0, 50.1, 50.2, 50.3, 50.4, 50.5])))
    assert v["g1"] == "FAIL" and v["overall"] == "FAIL", v


@case("FAIL G1: spread too small (top10)")
def _():
    v = gr.evaluate_gate(three_bad_one_good(good(truth=[1.50, 1.51, 1.52, 1.53, 1.54, 1.53])))
    assert v["g1"] == "FAIL", v


@case("FAIL G1: non-monotonic (two adjacent swaps)")
def _():
    v = gr.evaluate_gate(three_bad_one_good(good(ptp=[44.0, 40.0, 48.0, 56.0, 52.0, 60.0])))
    assert v["g1"] == "FAIL", v


@case("G1 tolerates exactly one adjacent swap at 6 points")
def _():
    s = gr.score_design(good(ptp=[44.0, 40.0, 48.0, 52.0, 56.0, 60.0]))
    assert s["g1"]["c_monotonic"], s["g1"]


@case("FAIL G2: shape objective decoupled from ptp")
def _():
    v = gr.evaluate_gate(four(good(ptp=PTP[::-1])))
    assert v["g1"] == "PASS" and v["g2"] == "FAIL" and v["overall"] == "FAIL", v


@case("FAIL G3: low surrogate rho")
def _():
    v = gr.evaluate_gate(four(good(pred=PRED[::-1])))
    assert v["g3"] == "FAIL" and v["g1"] == "PASS" and v["g2"] == "PASS", v


@case("FAIL G3: proxy wins")
def _():
    pred = [1.0, 1.2, 1.1, 1.3, 1.4, 1.5]
    v = gr.evaluate_gate(four(good(pred=pred, proxy=TRUTH)))
    assert v["g3"] == "FAIL", v
    assert any("proxy" in n for n in v["notes"]), v["notes"]


@case("FAIL G3: top-1 pick wrong")
def _():
    pred = [1.2, 1.3, 1.0, 1.4, 1.5, 1.6]
    v = gr.evaluate_gate(four(good(pred=pred)))
    assert v["g3"] == "FAIL", v
    assert all(not s["g3_top1_ok"] for s in four(good(pred=pred)).values())
    assert any("top-1" in n for n in v["notes"]), v["notes"]


@case("G3 near threshold flags a multi-seed rerun")
def _():
    pred = [1.2, 1.4, 1.0, 1.3, 1.6, 1.5]
    v = gr.evaluate_gate(four(good(pred=pred)))
    assert v["detail"]["g3_rerun_multiseed_warranted"], v
    far = gr.evaluate_gate(four(good()))
    assert not far["detail"]["g3_rerun_multiseed_warranted"], far


@case("INCONCLUSIVE with only 2 usable designs")
def _():
    v = gr.evaluate_gate({"d0": gr.score_design(good()), "d1": gr.score_design(good())})
    assert v["overall"] == "INCONCLUSIVE" and v["g1"] == "INCONCLUSIVE", v


@case("exact Spearman rho with 4, 5 and 6 points")
def _():
    def rho(n, swaps):
        a = list(range(n))
        b = list(range(n))
        for i in swaps:
            b[i], b[i + 1] = b[i + 1], b[i]
        return gr.spearman(a, b)

    assert abs(rho(6, []) - 1.0) < 1e-12
    assert abs(rho(6, [0]) - 0.942857) < 1e-5
    assert abs(rho(6, [0, 3]) - 0.885714) < 1e-5
    assert abs(rho(5, []) - 1.0) < 1e-12
    assert abs(rho(5, [1]) - 0.9) < 1e-12
    assert abs(rho(5, [0, 3]) - 0.8) < 1e-12
    assert abs(rho(4, []) - 1.0) < 1e-12
    assert abs(rho(4, [1]) - 0.8) < 1e-12


@case("threshold 0.89: 6 points tolerate one swap, 4 points need a perfect order, 5 points tolerate one swap (0.9)")
def _():
    def passes(n, swaps):
        order = list(range(n))
        for i in swaps:
            order[i], order[i + 1] = order[i + 1], order[i]
        addons = [0.05 * (k + 1) for k in range(n)]
        rows = make_rows([1.5 + 0.1 * o for o in order], [40.0 + 5 * o for o in order],
                         [1.0] * n, [1.0] * n, addons=addons)
        return gr.score_design(rows)["g1"]["c_monotonic"]

    assert passes(6, [1]) and not passes(6, [0, 3])
    assert passes(5, []) and passes(5, [1])
    assert passes(4, []) and not passes(4, [1])


# ---- exact-at-threshold behaviour ----

def S(g1=True, g2=0.9, sur=0.9, prx=0.0, ok=True):
    return {"g1": {"pass": g1, "a_ptp_spread": True, "b_top10_spread": True, "c_monotonic": True},
            "g2_rho_top10_ptp": g2, "g3_rho_surrogate": sur, "g3_rho_proxy": prx, "g3_top1_ok": ok}


def ev(**per_design):
    """evaluate_gate over 4 designs; each kwarg is a value or a list of 4."""
    def pick(v, i):
        return v[i] if isinstance(v, list) else v
    return gr.evaluate_gate({f"d{i}": S(**{k: pick(v, i) for k, v in per_design.items()}) for i in range(4)})


def perm_with_rho(n, target):
    base = list(range(n))
    for p in itertools.permutations(base):
        if abs(gr.spearman(base, p) - target) < 1e-6:
            return [float(x) for x in p]
    raise AssertionError(f"no permutation of {n} with rho {target}")


@case("G1 needs 3 of 4 designs (exactly 3 passes, 2 fails)")
def _():
    assert ev(g1=[True, True, True, False])["g1"] == "PASS"
    assert ev(g1=[True, True, False, False])["g1"] == "FAIL"


@case("G2 median: exactly 0.7 and 0.7 - 1e-12 pass, 0.7 - 1e-6 and 0.5 fail")
def _():
    assert ev(g2=0.7)["g2"] == "PASS" and ev(g2=0.7 - 1e-12)["g2"] == "PASS"
    assert ev(g2=0.7 - 1e-6)["g2"] == "FAIL" and ev(g2=0.6)["g2"] == "FAIL"


@case("G2 uses the median, not the mean, across designs")
def _():
    assert ev(g2=[0.9, 0.9, 0.9, -1.0])["g2"] == "PASS"
    assert ev(g2=[0.1, 0.1, 0.1, 0.9])["g2"] == "FAIL"


@case("G3 median: exactly 0.6 and 0.6 - 1e-12 pass, 0.6 - 1e-6 fails")
def _():
    assert ev(sur=0.6)["g3"] == "PASS" and ev(sur=0.6 - 1e-12)["g3"] == "PASS"
    assert ev(sur=0.6 - 1e-6)["g3"] == "FAIL" and ev(sur=0.5)["g3"] == "FAIL"


@case("G3 uses the median, not the mean, of the surrogate and the proxy rho")
def _():
    assert ev(sur=[0.9, 0.9, 0.9, 0.35])["g3"] == "PASS"
    assert ev(sur=[0.55, 0.55, 0.55, 1.0])["g3"] == "FAIL"
    assert ev(sur=[0.9, 0.9, 0.9, 0.9], prx=[0.0, 0.0, 0.0, 1.0])["g3"] == "PASS"
    assert ev(sur=[0.9, 0.9, 0.9, 0.9], prx=[0.8, 0.8, 0.8, -1.0])["g3"] == "FAIL"


@case("G3 rho >= 0.3 sub-clause: median passes but only 2 of 4 designs reach 0.3")
def _():
    v = ev(sur=[0.2, 0.2, 1.0, 1.0])
    assert v["detail"]["g3_median_surrogate_rho"] >= 0.6 and v["detail"]["g3_designs_rho_ge_min"] == 2
    assert v["g3"] == "FAIL" and any("G3(1)" in n for n in v["notes"]), v["notes"]
    assert ev(sur=[0.2, 0.3, 1.0, 1.0])["g3"] == "PASS"
    assert ev(sur=[0.3, 0.3, 1.0, 1.0])["g3"] == "PASS"
    assert ev(sur=[0.3 - 1e-12, 0.3, 1.0, 1.0])["detail"]["g3_designs_rho_ge_min"] == 4
    assert ev(sur=[0.3 - 1e-6, 0.3, 1.0, 1.0])["detail"]["g3_designs_rho_ge_min"] == 3


@case("G3 proxy margin: exactly 0.15 passes (0.7 - 0.55), 1e-6 short fails, 0.1 fails")
def _():
    assert ev(sur=0.7, prx=0.55)["g3"] == "PASS"
    assert ev(sur=0.7, prx=0.55 + 1e-6)["g3"] == "FAIL"
    assert ev(sur=0.7, prx=0.6)["g3"] == "FAIL"


@case("G3 top-1: exactly 3 designs ok passes, 2 fails")
def _():
    assert ev(ok=[True, True, True, False])["g3"] == "PASS"
    assert ev(ok=[True, True, False, False])["g3"] == "FAIL"


@case("G3 rerun band: 0.7 and 0.5 are inside, 1e-6 beyond is outside")
def _():
    def flag(sur):
        return ev(sur=sur)["detail"]["g3_rerun_multiseed_warranted"]
    assert flag(0.7) and flag(0.5) and flag(0.6) and flag(0.69)
    assert not flag(0.7 + 1e-6) and not flag(0.5 - 1e-6) and not flag(0.75) and not flag(0.9)
    assert flag(gr.spearman(list(range(5)), perm_with_rho(5, 0.5))), "a real rho of 0.5 (0.49999999999999994) is on the band edge"


@case("floating-point ties: rho steps of 0.1 at 5 points land exactly on G2/G3 thresholds")
def _():
    n = 5
    truth = [1.0, 2.0, 3.0, 4.0, 5.0]

    def sur_design(rho, proxy_rho=None):
        rows = make_rows(truth, PTP[:n], perm_with_rho(n, rho), perm_with_rho(n, proxy_rho if proxy_rho is not None else 0.0),
                         addons=GRID[:n])
        return gr.score_design(rows)

    def g2_design(rho):
        rows = make_rows(truth, [40.0 + 5 * p for p in perm_with_rho(n, rho)], truth, truth, addons=GRID[:n])
        return gr.score_design(rows)

    g31 = gr.evaluate_gate({f"d{i}": sur_design(r) for i, r in enumerate([0.1, 0.3, 0.9, 1.0])})
    assert not any("G3(1)" in n_ for n_ in g31["notes"]), g31["notes"]
    g2 = gr.evaluate_gate({f"d{i}": g2_design(r) for i, r in enumerate([0.1, 0.5, 0.9, 1.0])})
    assert g2["g2"] == "PASS", g2["detail"]
    g32 = gr.evaluate_gate({f"d{i}": sur_design(0.7, p) for i, p in enumerate([0.5, 0.5, 0.6, 0.6])})
    assert not any("G3(2)" in n_ for n_ in g32["notes"]), (g32["notes"], g32["detail"])


@case("G1 ptp spread: exactly 20% of the median passes, just below fails")
def _():
    assert gr.score_design(good(ptp=[40.0, 42.0, 44.0, 46.0, 48.0, 49.0]))["g1"]["a_ptp_spread"]
    assert not gr.score_design(good(ptp=[40.0, 42.0, 44.0, 46.0, 48.0, 48.9]))["g1"]["a_ptp_spread"]


@case("G1(a) compares the spread with the median ptp, not the mean (they differ across the 20% line)")
def _():
    g1 = gr.score_design(good(ptp=[40.0, 40.0, 40.0, 40.0, 40.0, 48.0]))["g1"]
    assert g1["ptp_median_c"] == 40.0 and g1["a_ptp_spread"] is True, g1
    g1 = gr.score_design(good(ptp=[48.0, 48.0, 48.0, 48.0, 48.0, 40.0]))["g1"]
    assert g1["ptp_median_c"] == 48.0 and g1["a_ptp_spread"] is False, g1


@case("G1 top10 spread: exactly 0.05 and a float-error 0.05 pass, 0.0499 fails")
def _():
    def b(truth):
        return gr.score_design(good(truth=truth))["g1"]["b_top10_spread"]
    assert b([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    assert 0.6 - 0.55 < 0.05
    assert b([0.55, 0.56, 0.57, 0.58, 0.59, 0.6])
    assert not b([0.0, 0.01, 0.02, 0.03, 0.04, 0.0499])


@case("G1 rho: exactly 0.89 passes, 0.89 - 1e-6 fails, sign is ignored")
def _():
    real = gr.spearman
    try:
        for val, want in ((0.89, True), (0.89 - 1e-12, True), (0.89 - 1e-6, False), (-0.89, True), (-0.89 + 1e-6, False)):
            gr.spearman = lambda a, b, v=val: v
            assert gr.score_design(good())["g1"]["c_monotonic"] is want, val
    finally:
        gr.spearman = real


@case("spearman of a constant series is 0.0 in either argument")
def _():
    assert gr.spearman([1, 2, 3], [5, 5, 5]) == 0.0 and gr.spearman([5, 5, 5], [1, 2, 3]) == 0.0
    assert gr.spearman([1, 2, 3], [1, 2, 3]) == 1.0


@case("spearman is rank based, not linear (a monotone but non-linear series gives exactly 1.0)")
def _():
    assert abs(gr.spearman([1, 2, 3, 4, 5], [1, 10, 100, 1000, 100000]) - 1.0) < 1e-12


@case("proxy rho is against truth top10 (not ptp) and surrogate rho uses pred, not proxy")
def _():
    s = gr.score_design(good(ptp=PTP[::-1], proxy=TRUTH, pred=TRUTH[::-1]))
    assert s["g3_rho_proxy"] == 1.0 and s["g3_rho_surrogate"] == -1.0 and s["g2_rho_top10_ptp"] == -1.0, s


@case("top-1 truth rank: ties are not strictly better, rank 2 is ok, rank 3 is not")
def _():
    def rank(truth, pred):
        s = gr.score_design(good(truth=truth, pred=pred))
        return s["g3_top1_truth_rank"], s["g3_top1_ok"]
    assert rank([1.0, 1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 1.0, 3.0, 4.0, 5.0, 6.0]) == (1, True)
    assert rank([1.0, 1.0 + 1e-12, 2.0, 3.0, 4.0, 5.0], [2.0, 1.0, 3.0, 4.0, 5.0, 6.0]) == (1, True)
    assert rank([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 1.0, 3.0, 4.0, 5.0, 6.0]) == (2, True)
    assert rank([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3.0, 2.0, 1.0, 4.0, 5.0, 6.0]) == (3, False)
    assert gr.score_design(good(pred=[1.0] * 6))["g3_top1_tag"] == "dn_005"


# ---- where each metric comes from ----

@case("variant_metrics: pred_top10 is from the prediction, proxy_top10 from blur(cell density), truth from the thermal map")
def _():
    yy, xx = np.mgrid[0:64, 0:64]

    def blob(cx, cy, s):
        return np.exp(-(np.square(xx - cx) + np.square(yy - cy)) / (2.0 * s * s)).astype(np.float32)

    thermal, cell, pred = blob(20, 20, 4) + 0.1, blob(44, 40, 2) + 0.05, blob(30, 50, 12) + 0.2
    m = gr.variant_metrics(thermal, cell, pred)
    tt, pt = shape_metrics(thermal)["top10_ratio"], shape_metrics(pred)["top10_ratio"]
    px, px_pred = shape_metrics(blur_proxy(cell))["top10_ratio"], shape_metrics(blur_proxy(pred))["top10_ratio"]
    assert m["truth_top10"] == tt and m["pred_top10"] == pt and m["proxy_top10"] == px, m
    assert abs(tt - pt) > 0.05 and abs(px - px_pred) > 0.05 and abs(px - tt) > 0.05, (tt, pt, px, px_pred)


@case("load_model returns the model in eval mode (no batch-norm batch statistics at inference)")
def _():
    path = os.path.join(SCRATCH, "random_init.pt")
    torch.save(CongestionUNet(in_channels=5, base_features=32, num_heatmap_layers=1).state_dict(), path)
    model = gr.load_model(path, torch.device("cpu"))
    assert not model.training and all(not m.training for m in model.modules())


# ---- exit codes through main() ----

def main_with(run_gate_fn, extra=(), out_name="out.json"):
    """Run gr.main() with run_gate replaced; returns (exit code, out path, stderr text)."""
    out = os.path.join(SCRATCH, out_name)
    if os.path.exists(out):
        os.remove(out)
    real, argv = gr.run_gate, sys.argv
    gr.run_gate = run_gate_fn
    sys.argv = ["gate_rank.py", "--variant-dir", SCRATCH, "--designs", "sky130hd/ibex",
                "--checkpoint-map", "ibex=x", "--out", out, *extra]
    code, err = None, io.StringIO()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            gr.main()
    except SystemExit as e:
        code = e.code
    finally:
        gr.run_gate, sys.argv = real, argv
    return code, out, err.getvalue()


def run_main(res, out_name, extra=()):
    code, out, _ = main_with(lambda *a, **k: res, extra, out_name=out_name)
    return code, os.path.exists(out)


def fake_results(verdict, smoke=False):
    return {"designs": {}, "excluded": {}, "checkpoints": {}, "verdict": verdict, "smoke_test": smoke,
            "thresholds": gr.THRESHOLDS}


@case("exit codes: PASS 0, FAIL 1, INCONCLUSIVE 2 (all write the result)")
def _():
    assert run_main(fake_results(gr.evaluate_gate(four(good()))), "x0.json") == (0, True)
    assert run_main(fake_results(gr.evaluate_gate(four(good(pred=PRED[::-1])))), "x1.json") == (1, True)
    assert run_main(fake_results(gr.evaluate_gate({"d0": gr.score_design(good())})), "x2.json") == (2, True)


@case("the CLI default for --min-points is 4 and --min-points is passed through")
def _():
    seen = []

    def spy(designs, ckpt_map, variant_dir, min_points, *a, **k):
        seen.append(min_points)
        return fake_results(gr.evaluate_gate(four(good())))

    assert main_with(spy)[0] == 0
    assert main_with(spy, ["--min-points", "5"])[0] == 0
    assert seen == [4, 5], seen


@case("usage error and unknown family exit 3 and write nothing")
def _():
    p = subprocess.run([sys.executable, GATE, "--variant-dir", SCRATCH], capture_output=True, text=True)
    assert p.returncode == 3 and "ERROR" in p.stderr, (p.returncode, p.stderr)
    out = os.path.join(SCRATCH, "typo.json")
    if os.path.exists(out):
        os.remove(out)
    p = subprocess.run([sys.executable, GATE, "--variant-dir", SCRATCH, "--designs", "sky130hd/ibex",
                        "--checkpoint-map", "ibx=x", "--out", out], capture_output=True, text=True)
    assert p.returncode == 3 and not os.path.exists(out) and "ibex" in p.stderr, (p.returncode, p.stderr)


@case("an error leaves no stale --out or --markdown behind (a fresh ERROR never sits beside an old verdict)")
def _():
    md = os.path.join(SCRATCH, "stale.md")

    def boom(*a, **k):
        raise RuntimeError("simulated failure")

    out = os.path.join(SCRATCH, "stale.json")
    for path in (out, md):
        with open(path, "w") as f:
            f.write("old verdict")
    real, argv = gr.run_gate, sys.argv
    gr.run_gate = boom
    sys.argv = ["gate_rank.py", "--variant-dir", SCRATCH, "--designs", "sky130hd/ibex",
                "--checkpoint-map", "ibex=x", "--out", out, "--markdown", md]
    code, err = None, io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            gr.main()
    except SystemExit as e:
        code = e.code
    finally:
        gr.run_gate, sys.argv = real, argv
    assert code == 3 and not os.path.exists(out) and not os.path.exists(md), (code, err.getvalue())
    assert err.getvalue().startswith("ERROR"), err.getvalue()


@case("an output write error exits 3, not 1, and leaves neither file (nor a temp file) behind")
def _():
    out_dir = os.path.join(SCRATCH, "wr")
    os.makedirs(out_dir, exist_ok=True)
    for leftover in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, leftover))
    res = fake_results(gr.evaluate_gate(four(good())))
    out = os.path.join(out_dir, "gate.json")
    code, _o, err = main_with(lambda *a, **k: res, ["--markdown", os.path.join(SCRATCH, "no_such_dir", "gate.md")], out_name="wr/gate.json")
    assert code == 3 and err.startswith("ERROR"), (code, err)
    assert not os.path.exists(out) and os.listdir(out_dir) == [], os.listdir(out_dir)
    code, _o, err = main_with(lambda *a, **k: res, out_name="no_such_dir/gate.json")
    assert code == 3 and err.startswith("ERROR"), (code, err)


@case("a missing dependency at import time exits 3 with an ERROR line, not a traceback with exit 1")
def _():
    fake = os.path.join(SCRATCH, "fakemods", "scipy")
    os.makedirs(fake, exist_ok=True)
    with open(os.path.join(fake, "__init__.py"), "w") as f:
        f.write('raise ImportError("simulated missing scipy")\n')
    env = {**os.environ, "PYTHONPATH": os.path.join(SCRATCH, "fakemods")}
    p = subprocess.run([sys.executable, GATE, "--variant-dir", SCRATCH, "--designs", "sky130hd/ibex",
                        "--checkpoint-map", "ibex=x", "--out", os.path.join(SCRATCH, "imp.json")],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 3 and p.stderr.startswith("ERROR") and "Traceback" not in p.stderr, (p.returncode, p.stderr[-300:])


@case("smoke-test results read INCONCLUSIVE per criterion in the JSON and the markdown; the numbers are kept apart")
def _():
    scored = four(good())
    res = fake_results(gr.evaluate_gate(scored), smoke=True)
    res["verdict"]["overall"] = "INCONCLUSIVE"
    res["designs"] = {"d0": {"variants": good(), "skipped_variants": [], "score": scored["d0"]}}
    res["checkpoints"] = {"ibex": {"epochs": 2}}
    res["verdict"]["smoke_test_underlying"] = {k: res["verdict"][k] for k in ("g1", "g2", "g3")}
    for k in ("g1", "g2", "g3"):
        res["verdict"][k] = gr.SMOKE_LABEL
    md = gr.to_markdown(res)
    assert "G1 PASS" not in md and "| PASS |" not in md and "| FAIL |" not in md, md
    assert f"G1 {gr.SMOKE_LABEL}, G2 {gr.SMOKE_LABEL}, G3 {gr.SMOKE_LABEL}, overall INCONCLUSIVE" in md, md


# ---- train_lodo: data selection, best epoch, sidecar, protection ----

@case("train_lodo.prepare_data: the datasets exclude the family, validation is not augmented, training is, counts are 22/20/23")
def _():
    for fam, n_excl, n_train in (("riscv32i", 5, 22), ("aes", 7, 20), ("ibex", 4, 23)):
        for seed in (0, 1, 2):
            d = train_lodo.prepare_data(DATA_DIR, fam, seed)
            assert len(d.excluded) == n_excl and len(d.train_keys) == n_train and len(d.val_keys) == 3, (fam, seed)
            assert not set(d.excluded) & (set(d.train_keys) | set(d.val_keys)), (fam, seed)
            assert d.train_set.dataset.augment is True and d.val_set.dataset.augment is False
            assert d.train_keys == [d.train_set.dataset.keys[i] for i in d.train_set.indices]
            assert d.val_keys == [d.val_set.dataset.keys[i] for i in d.val_set.indices]
            assert not set(d.val_keys) & set(d.train_keys)
            assert d.data_keys_sha256 == train_lodo.keys_sha256(train_lodo.data_dir_keys(DATA_DIR))
    aes = train_lodo.prepare_data(DATA_DIR, "aes", 0)
    assert "asap7_aes_lvt_base" in aes.excluded and "sky130hs_aes_base" in aes.excluded


@case("train_lodo refuses to start when the training dataset contains an excluded key, and names it")
def _():
    real = train_lodo.Subset

    def leaky(ds, idx):
        if ds.augment:
            idx = list(idx) + [i for i, k in enumerate(ds.keys) if k == "sky130hd_ibex_base"]
        return real(ds, idx)

    train_lodo.Subset = leaky
    try:
        try:
            train_lodo.prepare_data(DATA_DIR, "ibex", 0)
        except train_lodo.LeakError as e:
            assert "sky130hd_ibex_base" in str(e), str(e)
            return
    finally:
        train_lodo.Subset = real
    raise AssertionError("a training set containing an excluded key was accepted")


@case("train_lodo.assert_no_leak: excluded keys and any key of the holdout family (also aes_lvt) are refused in either set")
def _():
    ok = ["sky130hd_riscv32i_base", "asap7_jpeg_base"]
    train_lodo.assert_no_leak(ok, ok[:1], ["nangate45_ibex_base"], "ibex")
    for bad, where in (("nangate45_ibex_base", 0), ("sky130hs_ibex_base", 1), ("asap7_aes_lvt_base", 0)):
        fam = "aes" if "aes" in bad else "ibex"
        args = [ok + [bad] if where == 0 else ok, ok + [bad] if where == 1 else ok, [], fam]
        try:
            train_lodo.assert_no_leak(*args)
        except train_lodo.LeakError:
            continue
        raise AssertionError(f"{bad} accepted in set {where}")
    try:
        train_lodo.assert_no_leak(ok, ok, [ok[0]], "ibex")
    except train_lodo.LeakError:
        return
    raise AssertionError("a key listed as excluded was accepted")


@case("train_lodo keeps the epoch with the best validation MSE, not the last (and not a later tie)")
def _():
    model = torch.nn.Linear(1, 1, bias=False)
    tracker = train_lodo.BestTracker()
    for epoch, val in enumerate([3.0, 1.0, 2.0, 1.0, 5.0], start=1):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        tracker.update(epoch, val, model)
    assert tracker.best_epoch == 2 and tracker.best_val == 1.0, (tracker.best_epoch, tracker.best_val)
    assert float(tracker.best_state["weight"]) == 2.0, tracker.best_state
    with torch.no_grad():
        model.weight.fill_(99.0)
    assert float(tracker.best_state["weight"]) == 2.0, "the kept state must be a copy"


@case("train_lodo refuses to overwrite thermal_best.pt anywhere, and to touch experiments/thermal_loop")
def _():
    for out in (os.path.join(SCRATCH, "thermal_best.pt"), "thermal_best.pt", os.path.join(CONG, "checkpoints", "thermal_best.pt")):
        try:
            train_lodo.check_paths(DATA_DIR, out)
        except SystemExit as e:
            assert "thermal_best.pt" in str(e), str(e)
            continue
        raise AssertionError(f"{out} was accepted")
    train_lodo.check_paths(DATA_DIR, os.path.join(SCRATCH, "thermal_lodo_x.pt"))


@case("train_lodo FORBIDDEN_DIR is absolute and enforced from another working directory")
def _():
    assert os.path.isabs(train_lodo.FORBIDDEN_DIR)
    cwd = os.getcwd()
    os.chdir("/")
    try:
        try:
            train_lodo.check_paths(DATA_DIR, os.path.join(train_lodo.FORBIDDEN_DIR, "x.pt"))
        except SystemExit:
            return
        raise AssertionError("a path under the forbidden dir was accepted")
    finally:
        os.chdir(cwd)


# ---- lazily trained smoke checkpoints and synthetic variant directories ----

def train_ckpt(family):
    out = os.path.join(SCRATCH, f"smoke_{family}.pt")
    p = subprocess.run([sys.executable, TRAIN, "--data-dir", DATA_DIR, "--holdout-design", family,
                        "--out", out, "--epochs", "2", "--seed", "0"], capture_output=True, text=True)
    if p.returncode != 0:
        tail = " | ".join(p.stderr.strip().splitlines()[-2:])
        raise SetupError(f"training the 2-epoch {family} smoke checkpoint exited {p.returncode}: {tail}")
    return out


class LazyCheckpoints(dict):
    def __missing__(self, family):
        self[family] = train_ckpt(family)
        return self[family]


CKPT = LazyCheckpoints()
DEVICE = torch.device("cpu")


def write_variant(dirpath, pdk, design, tag, sharp, flat=False):
    rng = np.random.RandomState(zlib.crc32(f"{design}{tag}".encode()))
    yy, xx = np.mgrid[0:64, 0:64]
    cell = np.exp(-(((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * (20 - 10 * sharp) ** 2))).astype(np.float32)
    feats = {"cell_density": cell,
             "macro_density": (rng.rand(64, 64) * 0.1).astype(np.float32),
             "pin_density": (rng.rand(64, 64) * 0.5).astype(np.float32),
             "fanout_density": (rng.rand(64, 64) * 0.5).astype(np.float32)}
    np.savez(os.path.join(dirpath, f"{pdk}_{design}_{tag}_features.npz"), **feats)
    thermal = (45 + 60 * sharp * cell + 5 * cell).astype(np.float32)
    if flat:
        thermal = np.full((64, 64), 60.0, dtype=np.float32)
    np.savez(os.path.join(dirpath, f"{pdk}_{design}_{tag}_thermal_labels.npz"),
             thermal_map=thermal, power_grid=cell)


def build_dir(name, layout):
    d = os.path.join(SCRATCH, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for (pdk, design), tags in layout.items():
        for tag, kw in tags.items():
            write_variant(d, pdk, design, tag, **kw)
    return d


def tags_for(values, flat_tag=None):
    return {f"dn_{round(a * 100):03d}": {"sharp": 0.2 + a, "flat": f"dn_{round(a * 100):03d}" == flat_tag}
            for a in values}


_E2E = []


def e2e_dir():
    if not _E2E:
        _E2E.append(build_dir("e2e", {
            ("sky130hd", "ibex"): {**tags_for(GRID), **tags_for([0.10, 0.70])},
            ("nangate45", "ibex"): tags_for(GRID),
            ("asap7", "ibex"): tags_for(GRID),
            ("sky130hd", "riscv32i"): tags_for(GRID[:3]),
        }))
    return _E2E[0]


def variant_ckpt(name, family="ibex", edit=None, pt=None, epochs=None):
    """Copy a smoke checkpoint plus its sidecar, optionally editing the sidecar or swapping the .pt."""
    out = os.path.join(SCRATCH, f"{name}.pt")
    shutil.copy(pt or CKPT[family], out)
    with open(CKPT[family] + ".json") as f:
        sc = json.load(f)
    if pt is None:
        sc["pt_sha256"] = sha256_file(out)
    if epochs is not None:
        sc["epochs"] = epochs
    if edit:
        edit(sc)
    with open(out + ".json", "w") as f:
        json.dump(sc, f)
    return out


def affine_variants(name, ckpt, designs, mismatch):
    """Variant files scored with real inference. The truth of variant k is an affine function of the model's own
    prediction for one of six feature maps chosen to be ordered by prediction and only weakly related to the
    blur proxy. mismatch=False gives variant k the features whose prediction produced its truth (the surrogate
    ranks perfectly); mismatch=True gives it the features of variant 5-k (the surrogate ranks in reverse)."""
    model = CongestionUNet(in_channels=5, base_features=32, num_heatmap_layers=1)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    outdir = os.path.join(SCRATCH, name)
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir)
    probe = os.path.join(outdir, "_probe.npz")
    yy, xx = np.mgrid[0:64, 0:64]
    for di, design in enumerate(designs):
        pdk, dname = design.split("/")
        rng = np.random.RandomState(1000 + di)
        cands = []
        for _ in range(24):
            cx, cy, s = rng.uniform(10, 54), rng.uniform(10, 54), rng.uniform(3, 20)
            cell = np.exp(-(np.square(xx - cx) + np.square(yy - cy)) / (2 * s * s)).astype(np.float32)
            cell += (rng.rand(64, 64) * rng.uniform(0, 0.6)).astype(np.float32)
            feats = {"cell_density": cell, "macro_density": (rng.rand(64, 64) * 0.1).astype(np.float32),
                     "pin_density": (rng.rand(64, 64) * 0.5).astype(np.float32),
                     "fanout_density": (rng.rand(64, 64) * 0.5).astype(np.float32)}
            np.savez(probe, **feats)
            pred = gr.predict_map(model, probe, DEVICE)
            cands.append((shape_metrics(pred)["top10_ratio"], shape_metrics(blur_proxy(cell))["top10_ratio"], feats, pred))
        cands.sort(key=lambda c: c[0])
        pr, px = np.array([c[0] for c in cands]), np.array([c[1] for c in cands])
        best = None
        for _ in range(1500):
            idx = np.sort(rng.choice(len(cands), 6, replace=False))
            if pr[idx][-1] - pr[idx][0] < 0.5:
                continue
            r = gr.spearman(pr[idx], px[idx])
            if best is None or r < best[0]:
                best = (r, idx)
        if best is None or best[0] > 0.5:
            raise SetupError(f"could not build an affine variant set for {design}: best proxy rho {None if best is None else best[0]}")
        chosen = [cands[i] for i in best[1]]
        for k in range(6):
            feats = chosen[5 - k if mismatch else k][2]
            pred_t = chosen[k][3]
            nrm = (pred_t - pred_t.min()) / (pred_t.max() - pred_t.min())
            thermal = (45 + (10 + 4 * k) * nrm).astype(np.float32)
            base = os.path.join(outdir, f"{pdk}_{dname}_{gr.GRID_TAGS[k]}")
            np.savez(base + "_features.npz", **feats)
            np.savez(base + "_thermal_labels.npz", thermal_map=thermal, power_grid=chosen[k][2]["cell_density"])
    os.remove(probe)
    return outdir


_AFFINE = {}


def affine_dir(mismatch):
    if mismatch not in _AFFINE:
        _AFFINE[mismatch] = affine_variants("affine_fail" if mismatch else "affine_pass", CKPT["ibex"], IBEX3.split(","), mismatch)
    return _AFFINE[mismatch]


def cli(ckpt, name, designs=IBEX3, fam="ibex", variant_dir=None, extra=()):
    out = os.path.join(SCRATCH, name)
    if os.path.exists(out):
        os.remove(out)
    p = subprocess.run([sys.executable, GATE, "--variant-dir", variant_dir or e2e_dir(), "--designs", designs,
                        "--checkpoint-map", f"{fam}={ckpt}", "--out", out, *extra], capture_output=True, text=True)
    return p, out


def load_json(path):
    with open(path) as f:
        return json.load(f)


@case("sidecar records the held-out family, the actual dataset keys, lr, Laplacian weight and the data-key hash")
def _():
    sc = load_json(CKPT["aes"] + ".json")
    assert sc["pt_sha256"] == sha256_file(CKPT["aes"]) and (sc["epochs"], sc["seed"], sc["batch_size"]) == (2, 0, 4)
    assert (sc["lr"], sc["laplacian_weight"]) == (1e-3, 0.0), sc
    assert sc["data_keys_sha256"] == train_lodo.keys_sha256(train_lodo.data_dir_keys(DATA_DIR))
    assert "asap7_aes_lvt_base" in sc["excluded_keys"] and "sky130hs_aes_base" in sc["excluded_keys"]
    assert not any("ibex" in k or "riscv" in k for k in sc["excluded_keys"])
    assert not set(sc["val_keys"]) & set(sc["excluded_keys"])
    assert not set(sc["train_keys"]) & set(sc["excluded_keys"])
    d = train_lodo.prepare_data(DATA_DIR, "aes", 0)
    assert sc["train_keys"] == d.train_keys and sc["val_keys"] == d.val_keys and sc["excluded_keys"] == d.excluded


@case("end to end: pilot tags ignored, <4 variants excluded, verdict is not INCONCLUSIVE with 3 usable designs")
def _():
    designs = ["sky130hd/ibex", "nangate45/ibex", "asap7/ibex", "sky130hd/riscv32i"]
    res = gr.run_gate(designs, {"ibex": CKPT["ibex"], "riscv32i": CKPT["riscv32i"]}, e2e_dir(), 4, False, DEVICE, DATA_DIR)
    e = res["designs"]["sky130hd/ibex"]
    tags = sorted(r["tag"] for r in e["variants"])
    assert tags == ["dn_005", "dn_015", "dn_030", "dn_045", "dn_060", "dn_075"], tags
    assert e["ignored_off_grid_tags"] == ["dn_010", "dn_070"], e["ignored_off_grid_tags"]
    assert "sky130hd/riscv32i" in res["excluded"], res["excluded"]
    assert res["verdict"]["g1"] == gr.SMOKE_LABEL, res["verdict"]
    assert res["verdict"]["smoke_test_underlying"]["g1"] in ("PASS", "FAIL"), res["verdict"]
    assert res["smoke_test"] and res["verdict"]["overall"] == "INCONCLUSIVE", res["verdict"]
    assert "score" not in res["designs"]["sky130hd/riscv32i"]
    gr.to_markdown(res)


@case("a design with exactly 4 usable variants is scored; with 3 it is excluded")
def _():
    d = build_dir("four", {("sky130hd", "ibex"): tags_for(GRID[:4]), ("nangate45", "ibex"): tags_for(GRID[:3])})
    res = gr.run_gate(["sky130hd/ibex", "nangate45/ibex"], {"ibex": CKPT["ibex"]}, d, 4, False, DEVICE, DATA_DIR)
    assert "score" in res["designs"]["sky130hd/ibex"] and res["designs"]["sky130hd/ibex"]["score"]["n_variants"] == 4
    assert "score" not in res["designs"]["nangate45/ibex"] and "nangate45/ibex" in res["excluded"], res["excluded"]


@case("--all-variants lets the pilot tags in")
def _():
    cands, ignored, _m = gr.discover_variants(e2e_dir(), "sky130hd/ibex", True)
    assert len(cands) == 8 and not ignored


@case("end to end: two usable designs is INCONCLUSIVE")
def _():
    res = gr.run_gate(["sky130hd/ibex", "nangate45/ibex"], {"ibex": CKPT["ibex"]}, e2e_dir(), 4, False, DEVICE, DATA_DIR)
    assert res["verdict"]["overall"] == "INCONCLUSIVE", res["verdict"]


@case("a variant with a flat (unusable) thermal file is skipped with its reason")
def _():
    d = build_dir("flat", {("sky130hd", "ibex"): tags_for(GRID, flat_tag="dn_030")})
    res = gr.run_gate(["sky130hd/ibex"], {"ibex": CKPT["ibex"]}, d, 4, False, DEVICE, DATA_DIR)
    e = res["designs"]["sky130hd/ibex"]
    assert len(e["variants"]) == 5
    assert e["skipped_variants"][0]["tag"] == "dn_030" and "flat map" in e["skipped_variants"][0]["reason"]


def refused(ckpt, needle, exc=gr.LeakageError, **kw):
    try:
        gr.run_gate(["sky130hd/ibex"], {"ibex": ckpt}, e2e_dir(), 4, False, DEVICE, DATA_DIR, **kw)
    except exc as e:
        assert needle in str(e), str(e)
        return
    raise AssertionError(f"checkpoint was accepted (expected refusal mentioning '{needle}')")


@case("leakage guard refuses a checkpoint from the wrong family")
def _():
    refused(CKPT["riscv32i"], "holdout family")


@case("leakage guard refuses a checkpoint trained on the design")
def _():
    trained_on = os.path.join(SCRATCH, "trained_on_ibex.pt")
    shutil.copy(CKPT["ibex"], trained_on)
    sc = load_json(CKPT["ibex"] + ".json")
    sc["excluded_keys"] = [k for k in sc["excluded_keys"] if k != "sky130hd_ibex_base"]
    with open(trained_on + ".json", "w") as f:
        json.dump(sc, f)
    refused(trained_on, "excluded keys")


@case("gate_rank CLI exits 3 on a leakage refusal and writes no result")
def _():
    p, out = cli(CKPT["riscv32i"], "cli_gate.json", designs="sky130hd/ibex")
    assert p.returncode == 3 and not os.path.exists(out), (p.returncode, p.stderr)


@case("gate_rank CLI exits 2 and writes the JSON and markdown for an INCONCLUSIVE verdict")
def _():
    md = os.path.join(SCRATCH, "cli_gate2.md")
    p, out = cli(CKPT["ibex"], "cli_gate2.json", designs="sky130hd/ibex,nangate45/ibex", extra=["--markdown", md])
    assert p.returncode == 2, (p.returncode, p.stderr)
    assert load_json(out)["verdict"]["overall"] == "INCONCLUSIVE" and os.path.isfile(md)


@case("leak guard: a different checkpoint next to a genuine sidecar (hash mismatch)")
def _():
    refused(variant_ckpt("swapped", pt=CKPT["riscv32i"]), "does not match its sidecar")


@case("leak guard: a checkpoint with no hash in its sidecar")
def _():
    refused(variant_ckpt("nohash", edit=lambda sc: sc.pop("pt_sha256")), "pt_sha256")


@case("leak guard: a family key in train_keys or val_keys (partial sidecar)")
def _():
    refused(variant_ckpt("cross_train", edit=lambda sc: sc["train_keys"].append("sky130hs_ibex_base")), "trained or validated")
    refused(variant_ckpt("cross_val", edit=lambda sc: sc["val_keys"].append("nangate45_ibex_base")), "trained or validated")


@case("leak guard: excluded_keys missing a family key that is in the data dir")
def _():
    refused(variant_ckpt("short_excl", edit=lambda sc: sc.update(
        excluded_keys=[k for k in sc["excluded_keys"] if k != "nangate45_ibex_base"])), "nangate45_ibex_base")


@case("leak guard: accepts the genuine smoke checkpoint")
def _():
    res = gr.run_gate(["sky130hd/ibex"], {"ibex": CKPT["ibex"]}, e2e_dir(), 4, False, DEVICE, DATA_DIR)
    c = res["checkpoints"]["ibex"]
    assert c["sha256"] == sha256_file(CKPT["ibex"]) and (c["epochs"], c["seed"], c["batch_size"]) == (2, 0, 4), c


@case("--expect-epochs / --expect-seed: a checkpoint with other epochs or seed is refused (exit 3, nothing written); a match is scored and recorded")
def _():
    refused(CKPT["ibex"], "epochs", gr.CheckpointMismatchError, expect_epochs=200)
    refused(CKPT["ibex"], "seed", gr.CheckpointMismatchError, expect_seed=1)
    p, out = cli(CKPT["ibex"], "exp_bad.json", extra=["--expect-epochs", "200", "--expect-seed", "0"])
    assert p.returncode == 3 and not os.path.exists(out) and "epochs" in p.stderr, (p.returncode, p.stderr)
    p, out = cli(CKPT["ibex"], "exp_bad2.json", extra=["--expect-epochs", "2", "--expect-seed", "7"])
    assert p.returncode == 3 and not os.path.exists(out) and "seed" in p.stderr, (p.returncode, p.stderr)
    p, out = cli(CKPT["ibex"], "exp_ok.json", extra=["--expect-epochs", "2", "--expect-seed", "0"])
    assert p.returncode == 2 and load_json(out)["expected"] == {"epochs": 2, "seed": 0}, (p.returncode, p.stderr)
    p, out = cli(CKPT["ibex"], "exp_none.json")
    assert load_json(out)["expected"] == {"epochs": None, "seed": None}


@case("smoke test rule: epochs < 100 gives INCONCLUSIVE with smoke_test recorded; epochs 100 does not; 99 does")
def _():
    designs = ["sky130hd/ibex", "nangate45/ibex", "asap7/ibex"]
    smoke = gr.run_gate(designs, {"ibex": CKPT["ibex"]}, e2e_dir(), 4, False, DEVICE, DATA_DIR)
    assert smoke["smoke_test"] is True and smoke["verdict"]["overall"] == "INCONCLUSIVE", smoke["verdict"]
    assert any("SMOKE TEST" in n for n in smoke["verdict"]["notes"])
    assert "SMOKE TEST" in gr.to_markdown(smoke)
    full = gr.run_gate(designs, {"ibex": variant_ckpt("e100", epochs=100)}, e2e_dir(), 4, False, DEVICE, DATA_DIR)
    assert full["smoke_test"] is False and full["verdict"]["overall"] in ("PASS", "FAIL"), full["verdict"]
    assert full["verdict"]["g1"] in ("PASS", "FAIL") and "smoke_test_underlying" not in full["verdict"]
    edge = gr.run_gate(designs, {"ibex": variant_ckpt("e99", epochs=99)}, e2e_dir(), 4, False, DEVICE, DATA_DIR)
    assert edge["smoke_test"] is True


@case("smoke test rule looks at every checkpoint: one long and one short is a smoke test, whichever comes first")
def _():
    long_ibex, short_riscv = variant_ckpt("long_ibex", epochs=100), CKPT["riscv32i"]
    for designs in (["sky130hd/ibex", "sky130hd/riscv32i"], ["sky130hd/riscv32i", "sky130hd/ibex"]):
        res = gr.run_gate(designs, {"ibex": long_ibex, "riscv32i": short_riscv}, e2e_dir(), 4, False, DEVICE, DATA_DIR)
        assert res["smoke_test"] is True and res["verdict"]["overall"] == "INCONCLUSIVE", designs
        assert any("riscv32i" in n and "SMOKE" in n for n in res["verdict"]["notes"]), res["verdict"]["notes"]


@case("CLI on synthetic data: smoke checkpoint exits 2 with INCONCLUSIVE criteria; 100-epoch checkpoint exits 0 or 1")
def _():
    p, out = cli(CKPT["ibex"], "smoke_cli.json", extra=["--markdown", os.path.join(SCRATCH, "smoke_cli.md")])
    assert p.returncode == 2, (p.returncode, p.stderr)
    d = load_json(out)
    assert d["smoke_test"] is True and d["verdict"]["overall"] == "INCONCLUSIVE"
    assert [d["verdict"][k] for k in ("g1", "g2", "g3")] == [gr.SMOKE_LABEL] * 3, d["verdict"]
    assert set(d["verdict"]["smoke_test_underlying"]) == {"g1", "g2", "g3"}
    with open(os.path.join(SCRATCH, "smoke_cli.md")) as f:
        md = f.read()
    assert "PASS" not in md.split("Verdict:")[1].split("\n")[0] and "SMOKE TEST" in md, md
    p, out = cli(variant_ckpt("e100b", epochs=100), "full_cli.json")
    overall = load_json(out)["verdict"]["overall"]
    assert (p.returncode, overall) in ((0, "PASS"), (1, "FAIL")), (p.returncode, overall, p.stderr)


@case("real inference, PASS: truth is an affine function of the model's own prediction, so the real CLI exits 0 with every criterion PASS")
def _():
    ck = variant_ckpt("e100_pass", epochs=100)
    p, out = cli(ck, "real_pass.json", variant_dir=affine_dir(False))
    assert p.returncode == 0, (p.returncode, p.stderr, p.stdout[-600:])
    d = load_json(out)
    assert d["verdict"]["overall"] == "PASS" and (d["verdict"]["g1"], d["verdict"]["g2"], d["verdict"]["g3"]) == ("PASS",) * 3
    for design, e in d["designs"].items():
        assert len(e["variants"]) == 6 and abs(e["score"]["g3_rho_surrogate"] - 1.0) < 1e-9, design
        for r in e["variants"]:
            assert abs(r["pred_top10"] - r["truth_top10"]) < 1e-3, (design, r)
            pdk, name = design.split("/")
            with np.load(os.path.join(affine_dir(False), f"{pdk}_{name}_{r['tag']}_features.npz")) as f:
                want = shape_metrics(blur_proxy(f["cell_density"]))["top10_ratio"]
            assert abs(r["proxy_top10"] - want) < 1e-6, (design, r["tag"])
        assert e["score"]["g3_rho_proxy"] < 0.5, e["score"]


@case("real inference, FAIL: the surrogate sees the wrong variants, so the real CLI exits 1 and only G3 fails")
def _():
    ck = variant_ckpt("e100_fail", epochs=100)
    p, out = cli(ck, "real_fail.json", variant_dir=affine_dir(True))
    assert p.returncode == 1, (p.returncode, p.stderr, p.stdout[-600:])
    d = load_json(out)
    v = d["verdict"]
    assert v["overall"] == "FAIL" and (v["g1"], v["g2"], v["g3"]) == ("PASS", "PASS", "FAIL"), v
    for design, e in d["designs"].items():
        assert abs(e["score"]["g3_rho_surrogate"] + 1.0) < 1e-9 and not e["score"]["g3_top1_ok"], design


@case("CLI errors exit 3 and write nothing: missing checkpoint, corrupt .pt, malformed sidecar, state-dict mismatch")
def _():
    def check(ckpt, name):
        p, out = cli(ckpt, name)
        assert p.returncode == 3 and not os.path.exists(out) and p.stderr.startswith("ERROR"), (name, p.returncode, p.stderr[-300:])

    check(os.path.join(SCRATCH, "does_not_exist.pt"), "e_missing.json")

    corrupt = variant_ckpt("corrupt")
    with open(corrupt, "wb") as f:
        f.write(b"not a checkpoint")
    sc = load_json(corrupt + ".json")
    sc["pt_sha256"] = sha256_file(corrupt)
    with open(corrupt + ".json", "w") as f:
        json.dump(sc, f)
    check(corrupt, "e_corrupt.json")

    bad_json = variant_ckpt("badjson")
    with open(bad_json + ".json", "w") as f:
        f.write("{not json")
    check(bad_json, "e_badjson.json")

    mismatch = variant_ckpt("mismatch")
    torch.save({"weight": torch.zeros(1)}, mismatch)
    sc = load_json(mismatch + ".json")
    sc["pt_sha256"] = sha256_file(mismatch)
    with open(mismatch + ".json", "w") as f:
        json.dump(sc, f)
    check(mismatch, "e_mismatch.json")


@case("a forward-pass failure exits 3 and writes nothing")
def _():
    real, real_run = gr.predict_map, gr.run_gate

    def boom(*a, **k):
        raise RuntimeError("simulated forward failure")

    gr.predict_map = boom
    try:
        p_code, out, err = main_with(
            lambda *a, **k: real_run(["sky130hd/ibex"], {"ibex": CKPT["ibex"]}, e2e_dir(), 4, False, DEVICE, DATA_DIR),
            out_name="e_forward.json")
    finally:
        gr.predict_map = real
    assert p_code == 3 and not os.path.exists(out) and err.startswith("ERROR"), (p_code, err)


@case("checkpoint reuse rule: same family, epochs, seed, batch size, lr, Laplacian weight, data keys and hash only")
def _():
    ck = CKPT["ibex"]

    def reuse(path=ck, fam="ibex", epochs=2, seed=0, bs=4, lr=1e-3, lap=0.0, data=DATA_DIR):
        return train_lodo.reusable(path, fam, epochs, seed, bs, lr, lap, data)[0]

    assert reuse()
    assert not reuse(epochs=200) and not reuse(seed=1) and not reuse(bs=8) and not reuse(fam="aes")
    assert not reuse(lr=1e-4) and not reuse(lap=0.1)
    assert not reuse(path=os.path.join(SCRATCH, "nope.pt"))
    assert not reuse(path=variant_ckpt("tampered", pt=CKPT["riscv32i"]))
    assert not reuse(path=variant_ckpt("nohash2", edit=lambda sc: sc.pop("pt_sha256")))
    keys = train_lodo.data_dir_keys(DATA_DIR)
    thin = os.path.join(SCRATCH, "thin_data")
    os.makedirs(thin, exist_ok=True)
    for k in keys[1:]:
        for suffix in ("_features.npz", "_thermal_labels.npz"):
            open(os.path.join(thin, k + suffix), "w").close()
    assert not reuse(data=thin), "a checkpoint made from another set of data files must be retrained"
    old = variant_ckpt("old_sidecar", edit=lambda sc: (sc.pop("lr"), sc.pop("data_keys_sha256")))
    assert not reuse(path=old)


@case("train_lodo --verify-only exits 0 for the matching checkpoint and 3 for any mismatch, training nothing")
def _():
    def verify(ck, **over):
        args = {"epochs": "2", "seed": "0"}
        args.update(over)
        return subprocess.run([sys.executable, TRAIN, "--data-dir", DATA_DIR, "--holdout-design", "ibex", "--out", ck,
                               "--epochs", args["epochs"], "--seed", args["seed"], "--verify-only"],
                              capture_output=True, text=True)

    before = sha256_file(CKPT["ibex"])
    assert verify(CKPT["ibex"]).returncode == 0
    p = verify(CKPT["ibex"], epochs="200")
    assert p.returncode == 3 and "epochs" in p.stdout, (p.returncode, p.stdout)
    assert verify(CKPT["ibex"], seed="1").returncode == 3
    assert verify(os.path.join(SCRATCH, "absent.pt")).returncode == 3
    assert sha256_file(CKPT["ibex"]) == before


# ---- run_gate.sh in throwaway flow trees ----

_STUB_MAIN = '''if __name__ == "__main__":
    if "--verify-only" in sys.argv:
        main()
    else:
        with open(os.environ["STUB_LOG"], "a") as log:
            log.write(" ".join(sys.argv[1:]) + "\\n")
        if os.environ.get("STUB_KILL"):
            os.kill(os.getppid(), 9)
        if os.environ.get("STUB_TOUCH"):
            with open(os.environ["STUB_TOUCH"], "a") as f:
                f.write("changed")
        sys.exit(int(os.environ.get("STUB_EXIT", "0")))
'''

_STUB_GATE = '''import json, os, sys
if os.environ.get("STUB_GATE_JSON"):
    out = sys.argv[sys.argv.index("--out") + 1]
    with open(out, "w") as f:
        json.dump({"verdict": {"overall": os.environ["STUB_GATE_JSON"]}, "designs": {}}, f)
sys.exit(int(os.environ.get("STUB_GATE_EXIT", "0")))
'''

_TREES = [0]


def fake_flow(variant_dir, sidecar_epochs, stub_gate=False):
    """A throwaway flow/ tree: real scripts, a trainer that only logs (its --verify-only stays real), the given
    variants and a checkpoint whose sidecar says sidecar_epochs. Returns (flow dir, checkpoint path)."""
    _TREES[0] += 1
    flow = os.path.join(SCRATCH, f"tree{_TREES[0]}", "flow")
    cong = os.path.join(flow, "util", "ml", "congestion")
    for sub in ("loop", "checkpoints", "experiments/thermal_loop"):
        os.makedirs(os.path.join(cong, sub))
    os.makedirs(os.path.join(flow, "results", "sky130hd", "ibex", "base"))
    with open(os.path.join(flow, "results", "sky130hd", "ibex", "base", "f.txt"), "w") as f:
        f.write("base")
    docker = os.path.join(flow, "util", "docker_shell")
    with open(docker, "w") as f:
        f.write("#!/bin/sh\nexit 0\n")
    os.chmod(docker, 0o755)
    for sub in ("models", "inference", "training", "data"):
        os.symlink(os.path.join(CONG, sub), os.path.join(cong, sub))
    for fn in os.listdir(HERE):
        if fn.endswith((".py", ".sh")):
            shutil.copy(os.path.join(HERE, fn), os.path.join(cong, "loop", fn))
    trainer = os.path.join(cong, "loop", "train_lodo.py")
    with open(trainer) as f:
        src = f.read()
    tail = 'if __name__ == "__main__":\n    main()\n'
    assert src.endswith(tail)
    with open(trainer, "w") as f:
        f.write(src[: -len(tail)] + _STUB_MAIN)
    if stub_gate:
        with open(os.path.join(cong, "loop", "gate_rank.py"), "w") as f:
            f.write(_STUB_GATE)
    shutil.copytree(variant_dir, os.path.join(cong, "experiments", "thermal_loop", "data"))
    ck = os.path.join(cong, "checkpoints", "thermal_lodo_ibex.pt")
    src_ck = variant_ckpt(f"tree{_TREES[0]}_ck", epochs=sidecar_epochs)
    shutil.copy(src_ck, ck)
    shutil.copy(src_ck + ".json", ck + ".json")
    return flow, ck


def run_flow(flow, extra_env=None, args=(), epochs="200"):
    env = {**os.environ, "STUB_LOG": os.path.join(flow, "..", "train_calls.log")}
    env.update(extra_env or {})
    exp = os.path.join(flow, "util", "ml", "congestion", "experiments", "thermal_loop")
    p = subprocess.run(["bash", os.path.join(flow, "util", "ml", "congestion", "loop", "run_gate.sh"),
                        "--designs", IBEX3, "--skip-place", "--skip-extract", "--epochs", epochs, *args],
                       capture_output=True, text=True, env=env)
    return p, exp


def write_stale(exp):
    for fn in ("gate.json", "gate.md"):
        with open(os.path.join(exp, fn), "w") as f:
            f.write("STALE")


def no_verdict(exp):
    return not os.path.exists(os.path.join(exp, "gate.json")) and not os.path.exists(os.path.join(exp, "gate.md"))


@case("run_gate.sh: a matching checkpoint gives a real PASS (exit 0) and a real FAIL (exit 1); trainer gets --seed 0 and the epochs; gate.json records the expectation")
def _():
    for mismatch, code, verdict, epochs in ((False, 0, "PASS", 150), (True, 1, "FAIL", 200)):
        flow, _ck = fake_flow(affine_dir(mismatch), epochs)
        p, exp = run_flow(flow, epochs=str(epochs))
        assert p.returncode == code, (p.returncode, p.stdout[-800:], p.stderr[-800:])
        d = load_json(os.path.join(exp, "gate.json"))
        assert d["verdict"]["overall"] == verdict and d["expected"] == {"epochs": epochs, "seed": 0}, d["expected"]
        assert os.path.isfile(os.path.join(exp, "gate.md"))
        assert f"Verdict: {verdict} (exit {code})" in p.stdout, p.stdout[-400:]
        with open(os.path.join(flow, "..", "train_calls.log")) as f:
            call = f.read()
        assert "--seed 0" in call and f"--epochs {epochs} " in call and "--reuse-if-valid" in call and "--holdout-design ibex" in call, call


@case("run_gate.sh: a crashed trainer exits 3 with no verdict, removes a stale gate.json/gate.md and never runs gate_rank, even though a valid matching checkpoint exists")
def _():
    flow, _ck = fake_flow(affine_dir(False), 200)
    exp = os.path.join(flow, "util", "ml", "congestion", "experiments", "thermal_loop")
    write_stale(exp)
    p, exp = run_flow(flow, {"STUB_EXIT": "1"})
    assert p.returncode == 3, (p.returncode, p.stdout[-600:], p.stderr[-600:])
    assert no_verdict(exp), os.listdir(exp)
    assert "START gate_rank" not in p.stdout and "Verdict:" not in p.stdout, p.stdout
    assert "ERROR, no verdict" in p.stderr and "train ibex" in p.stderr, p.stderr


@case("run_gate.sh: the reviewer's scenario (stale 150-epoch checkpoint, --epochs 200, trainer crashes) exits 3 with no verdict")
def _():
    flow, _ck = fake_flow(affine_dir(False), 150)
    p, exp = run_flow(flow, {"STUB_EXIT": "1"})
    assert p.returncode == 3 and no_verdict(exp), (p.returncode, p.stdout[-600:])
    assert "Verdict: PASS" not in p.stdout and "Verdict: FAIL" not in p.stdout


@case("run_gate.sh: a trainer that exits 0 but leaves a checkpoint of other epochs is caught by the verify step (exit 3, no verdict)")
def _():
    flow, _ck = fake_flow(affine_dir(False), 150)
    p, exp = run_flow(flow, {"STUB_EXIT": "0"})
    assert p.returncode == 3 and no_verdict(exp), (p.returncode, p.stdout[-600:], p.stderr[-600:])
    assert "verify ibex" in p.stderr and "START gate_rank" not in p.stdout


@case("run_gate.sh: --skip-train still verifies the checkpoint against --epochs")
def _():
    flow, _ck = fake_flow(affine_dir(False), 150)
    p, exp = run_flow(flow, args=["--skip-train"])
    assert p.returncode == 3 and no_verdict(exp), (p.returncode, p.stderr[-400:])
    assert not os.path.exists(os.path.join(flow, "..", "train_calls.log"))


@case("run_gate.sh: a changed results/*/base file during the run exits 3 and removes gate.json/gate.md")
def _():
    flow, _ck = fake_flow(affine_dir(False), 200)
    touched = os.path.join(flow, "results", "sky130hd", "ibex", "base", "f.txt")
    p, exp = run_flow(flow, {"STUB_TOUCH": touched})
    assert p.returncode == 3 and no_verdict(exp), (p.returncode, p.stdout[-400:], p.stderr[-400:])
    assert "changed during the run" in p.stderr, p.stderr


@case("run_gate.sh: gate_rank exiting 0 without a gate.json, or with a gate.json that disagrees, is an ERROR (exit 3, nothing left)")
def _():
    flow, _ck = fake_flow(affine_dir(False), 200, stub_gate=True)
    p, exp = run_flow(flow)
    assert p.returncode == 3 and no_verdict(exp), (p.returncode, p.stdout[-400:], p.stderr[-400:])
    assert "missing, unreadable" in p.stderr, p.stderr
    p, exp = run_flow(flow, {"STUB_GATE_JSON": "FAIL"})
    assert p.returncode == 3 and no_verdict(exp), (p.returncode, p.stderr[-400:])
    p, exp = run_flow(flow, {"STUB_GATE_JSON": "PASS"})
    assert p.returncode == 0 and os.path.isfile(os.path.join(exp, "gate.json")), (p.returncode, p.stderr[-400:])
    p, exp = run_flow(flow, {"STUB_GATE_JSON": "INCONCLUSIVE", "STUB_GATE_EXIT": "2"})
    assert p.returncode == 2, p.returncode
    p, exp = run_flow(flow, {"STUB_GATE_EXIT": "3"})
    assert p.returncode == 3 and no_verdict(exp)
    for odd in ("4", "139"):
        p, exp = run_flow(flow, {"STUB_GATE_EXIT": odd})
        assert p.returncode == 3 and no_verdict(exp), (odd, p.returncode)
    p, exp = run_flow(flow, {"STUB_GATE_EXIT": "1"})
    assert p.returncode == 3 and no_verdict(exp), "exit 1 without a gate.json must not read as FAIL"


@case("run_gate.sh: a run killed mid-way leaves no stale gate.json/gate.md from an earlier run")
def _():
    flow, _ck = fake_flow(affine_dir(False), 200)
    exp = os.path.join(flow, "util", "ml", "congestion", "experiments", "thermal_loop")
    write_stale(exp)
    p, exp = run_flow(flow, {"STUB_KILL": "1"})
    assert p.returncode != 0 and no_verdict(exp), (p.returncode, os.listdir(exp))


@case("run_gate.sh dry run: retrains through --reuse-if-valid, verifies, warns loudly on --epochs < 100, runs nothing")
def _():
    p = subprocess.run(["bash", RUN_GATE, "--dry-run", "--epochs", "2"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "--reuse-if-valid" in p.stdout and "--epochs 2 --seed 0" in p.stdout and "--verify-only" in p.stdout
    assert "--expect-epochs 2 --expect-seed 0" in p.stdout, p.stdout
    assert "SMOKE TEST" in p.stderr
    p = subprocess.run(["bash", RUN_GATE, "--dry-run", "--epochs", "200"], capture_output=True, text=True)
    assert p.returncode == 0 and "SMOKE TEST" not in p.stderr
    p = subprocess.run(["bash", RUN_GATE, "--dry-run", "--epochs", "abc"], capture_output=True, text=True)
    assert p.returncode == 3, p.returncode


report()
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
