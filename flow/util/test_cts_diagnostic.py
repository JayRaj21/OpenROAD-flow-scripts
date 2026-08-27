#!/usr/bin/env python3
"""Unit tests for cts_diagnostic.py — no OpenROAD, no filesystem outside tmp."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cts_diagnostic import (
    buffer_per_sink,
    check_cliff,
    gather,
    parse_cts_log,
    parse_cts_skew_json,
    parse_cts_skew_rpt,
    print_report,
)

# Fixture text mirrors the real format found in checked-in
# flow/logs/nangate45/ibex/base/4_1_cts.log (3 clock nets: clk_i, clk_i_regs,
# clk), each producing one final "Created N clock buffers." line and one
# "Sinks N" summary line, plus a separate "Total number of delay buffers" line.
CTS_LOG_FIXTURE = """\
[INFO CTS-0007] Net "clk_i" found for clock "core_clock".
[INFO CTS-0011]  Clock net "clk_i" for macros has 1 sinks.
[INFO CTS-0011]  Clock net "clk_i_regs" for registers has 995 sinks.
[INFO CTS-0010]  Clock net "clk" has 943 sinks.
[INFO CTS-0008] TritonCTS found 3 clock nets.
[INFO CTS-0018]     Created 2 clock buffers.
[INFO CTS-0012]     Minimum number of buffers in the clock path: 2.
[INFO CTS-0018]     Created 143 clock buffers.
[INFO CTS-0012]     Minimum number of buffers in the clock path: 3.
[INFO CTS-0018]     Created 157 clock buffers.
[INFO CTS-0124] Clock net "clk_i"
[INFO CTS-0125]  Sinks 1
[INFO CTS-0098] Clock net "clk_i_regs"
[INFO CTS-0099]  Sinks 1100
[INFO CTS-0100]  Leaf buffers 126
[INFO CTS-0098] Clock net "clk"
[INFO CTS-0099]  Sinks 1066
[INFO CTS-0100]  Leaf buffers 140
[INFO CTS-0033] Balancing latency for clock core_clock
[INFO CTS-0037] Total number of delay buffers: 2
"""

CTS_JSON_FIXTURE = {
    "cts__clock__skew__setup": 0.025187,
    "cts__clock__skew__hold": 0.0252836,
    "cts__timing__setup__ws": -0.0072,
}

CTS_RPT_SKEW_FIXTURE = """\
==========================================================================
cts final report_clock_skew
--------------------------------------------------------------------------
Clock core_clock
   0.30 source latency foo/CK ^
  -0.27 target latency bar/CK ^
   0.00 CRPR
--------------
   0.03 setup skew

"""


def _write(path, content):
    with open(path, "w") as f:
        f.write(content)


class TestParseCtsLog(unittest.TestCase):
    def test_extracts_buffer_and_sink_counts(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "4_1_cts.log")
            _write(log_path, CTS_LOG_FIXTURE)

            metrics = parse_cts_log(log_path)

            # 2 + 143 + 157 (per-net tree buffers) + 2 (delay buffers) = 304
            self.assertEqual(metrics["buffer_count"], 304)
            # 1 + 1100 + 1066 (post dummy-load-balancing sink totals)
            self.assertEqual(metrics["sink_count"], 2167)
            self.assertEqual(metrics["leaf_buffer_count"], 266)

    def test_missing_log_returns_empty(self):
        metrics = parse_cts_log("/nonexistent/4_1_cts.log")
        self.assertEqual(metrics, {})

    def test_ignores_unrelated_sink_mentions(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "4_1_cts.log")
            _write(
                log_path,
                "[INFO CTS-0028]  Total number of sinks: 995.\n"
                "[INFO CTS-0035]  Number of sinks covered: 126.\n"
                "[INFO CTS-0018]     Created 5 clock buffers.\n"
                "[INFO CTS-0099]  Sinks 10\n",
            )
            metrics = parse_cts_log(log_path)
            self.assertEqual(metrics["sink_count"], 10)
            self.assertEqual(metrics["buffer_count"], 5)


class TestBufferPerSink(unittest.TestCase):
    def test_computes_ratio(self):
        self.assertAlmostEqual(
            buffer_per_sink({"buffer_count": 304, "sink_count": 2167}),
            304 / 2167,
        )

    def test_missing_data_returns_none(self):
        self.assertIsNone(buffer_per_sink({"buffer_count": 304}))
        self.assertIsNone(buffer_per_sink({}))

    def test_zero_sinks_returns_none(self):
        self.assertIsNone(buffer_per_sink({"buffer_count": 5, "sink_count": 0}))


class TestSkewParsing(unittest.TestCase):
    def test_json_extracts_setup_and_hold(self):
        with tempfile.TemporaryDirectory() as d:
            json_path = os.path.join(d, "4_1_cts.json")
            with open(json_path, "w") as f:
                json.dump(CTS_JSON_FIXTURE, f)

            metrics = parse_cts_skew_json(json_path)
            self.assertAlmostEqual(metrics["setup_skew"], 0.025187)
            self.assertAlmostEqual(metrics["hold_skew"], 0.0252836)

    def test_json_missing_file_returns_empty(self):
        self.assertEqual(parse_cts_skew_json("/nonexistent/4_1_cts.json"), {})

    def test_json_malformed_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            json_path = os.path.join(d, "4_1_cts.json")
            _write(json_path, "{not valid json")
            self.assertEqual(parse_cts_skew_json(json_path), {})

    def test_rpt_fallback_extracts_setup_skew(self):
        with tempfile.TemporaryDirectory() as d:
            rpt_path = os.path.join(d, "4_cts_final.rpt")
            _write(rpt_path, CTS_RPT_SKEW_FIXTURE)

            metrics = parse_cts_skew_rpt(rpt_path)
            self.assertAlmostEqual(metrics["setup_skew"], 0.03)
            self.assertNotIn("hold_skew", metrics)


class TestCliffCheck(unittest.TestCase):
    def test_cliff_detected_when_drop_exceeds_threshold(self):
        stage_map = {
            "CTS": {"wns": -0.01},
            "Global route": {"wns": -0.20},
        }
        result = check_cliff(stage_map, threshold=0.05)
        self.assertTrue(result["detected"])
        self.assertAlmostEqual(result["drop"], 0.19)

    def test_no_cliff_when_drop_within_threshold(self):
        stage_map = {
            "CTS": {"wns": -0.05},
            "Global route": {"wns": -0.08},
        }
        result = check_cliff(stage_map, threshold=0.05)
        self.assertFalse(result["detected"])

    def test_no_cliff_when_grt_improves(self):
        stage_map = {
            "CTS": {"wns": -0.20},
            "Global route": {"wns": -0.05},
        }
        result = check_cliff(stage_map, threshold=0.05)
        self.assertFalse(result["detected"])
        self.assertLess(result["drop"], 0)

    def test_missing_stage_data_returns_none(self):
        self.assertIsNone(check_cliff({"CTS": {"wns": -0.01}}, threshold=0.05))
        self.assertIsNone(check_cliff({}, threshold=0.05))


class TestPrintReportExitSignals(unittest.TestCase):
    def _run(self, structural, cliff, buffer_ratio_threshold=0.5):
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            over_buffered, cliff_detected = print_report(
                structural, cliff, buffer_ratio_threshold, "unit-test"
            )
        return over_buffered, cliff_detected, buf.getvalue()

    def test_flags_over_buffering(self):
        structural = {"buffer_count": 60, "sink_count": 100}
        over_buffered, cliff_detected, out = self._run(structural, None)
        self.assertTrue(over_buffered)
        self.assertFalse(cliff_detected)
        self.assertIn("OVER-BUFFERING WARNING", out)

    def test_does_not_flag_normal_ratio(self):
        structural = {"buffer_count": 20, "sink_count": 100}
        over_buffered, cliff_detected, out = self._run(structural, None)
        self.assertFalse(over_buffered)
        self.assertNotIn("OVER-BUFFERING WARNING", out)

    def test_flags_cliff(self):
        cliff = {"cts_wns": -0.01, "grt_wns": -0.20, "drop": 0.19, "detected": True}
        over_buffered, cliff_detected, out = self._run({}, cliff)
        self.assertTrue(cliff_detected)
        self.assertIn("CLIFF DETECTED", out)

    def test_no_cliff_message_when_not_detected(self):
        cliff = {"cts_wns": -0.05, "grt_wns": -0.06, "drop": 0.01, "detected": False}
        over_buffered, cliff_detected, out = self._run({}, cliff)
        self.assertFalse(cliff_detected)
        self.assertIn("No CTS->GRT cliff detected", out)


class TestGatherIntegration(unittest.TestCase):
    def test_gather_combines_pr_metrics_and_structural(self):
        with tempfile.TemporaryDirectory() as d:
            reports_dir = os.path.join(d, "reports")
            logs_dir = os.path.join(d, "logs")
            os.makedirs(reports_dir)
            os.makedirs(logs_dir)

            _write(
                os.path.join(reports_dir, "4_cts_final.rpt"),
                "tns max -0.02\nwns max -0.01\nworst slack max -0.01\n",
            )
            _write(
                os.path.join(reports_dir, "5_global_route.rpt"),
                "tns max -1.20\nwns max -0.30\nworst slack max -0.30\n",
            )
            _write(os.path.join(logs_dir, "4_1_cts.log"), CTS_LOG_FIXTURE)
            with open(os.path.join(logs_dir, "4_1_cts.json"), "w") as f:
                json.dump(CTS_JSON_FIXTURE, f)

            rows, stage_map, structural = gather(reports_dir, logs_dir)

            self.assertEqual(stage_map["CTS"]["wns"], -0.01)
            self.assertEqual(stage_map["Global route"]["wns"], -0.30)
            self.assertEqual(structural["buffer_count"], 304)
            self.assertEqual(structural["sink_count"], 2167)
            self.assertAlmostEqual(structural["setup_skew"], 0.025187)

            cliff = check_cliff(stage_map, threshold=0.05)
            self.assertTrue(cliff["detected"])


if __name__ == "__main__":
    unittest.main()
