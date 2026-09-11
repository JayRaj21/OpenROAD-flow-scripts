#!/usr/bin/env python3
"""Unit tests for cts_diagnostic.py — no OpenROAD, no filesystem outside tmp."""

import io
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cts_diagnostic import (
    EXIT_CLEAN,
    EXIT_FINDING,
    EXIT_USAGE_ERROR,
    buffer_per_sink,
    check_cliff,
    derive_logs_dir,
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

    def test_json_null_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            json_path = os.path.join(d, "4_1_cts.json")
            _write(json_path, "null")
            self.assertEqual(parse_cts_skew_json(json_path), {})

    def test_json_list_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            json_path = os.path.join(d, "4_1_cts.json")
            _write(json_path, "[1, 2, 3]")
            self.assertEqual(parse_cts_skew_json(json_path), {})

    def test_json_scalar_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            json_path = os.path.join(d, "4_1_cts.json")
            _write(json_path, "3")
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

    def test_tns_cliff_detected_when_wns_looks_fine(self):
        # Real-data pattern (nangate45/swerv): dWNS is a tiny improvement
        # (+0.040, well within the WNS threshold) but TNS blows up 60%.
        stage_map = {
            "CTS": {"wns": -0.150, "tns": -306.65},
            "Global route": {"wns": -0.110, "tns": -492.21},
        }
        result = check_cliff(stage_map, threshold=0.05, tns_threshold_pct=20.0)
        self.assertFalse(result["wns_detected"])
        self.assertTrue(result["tns_detected"])
        self.assertTrue(result["detected"])
        self.assertAlmostEqual(result["tns_drop"], 185.56, places=2)
        self.assertGreater(result["tns_drop_pct"], 20.0)

    def test_no_tns_cliff_when_within_threshold(self):
        stage_map = {
            "CTS": {"wns": -0.01, "tns": -100.0},
            "Global route": {"wns": -0.02, "tns": -105.0},
        }
        result = check_cliff(stage_map, threshold=0.05, tns_threshold_pct=20.0)
        self.assertFalse(result["tns_detected"])
        self.assertFalse(result["detected"])

    def test_tns_missing_does_not_crash_or_falsely_detect(self):
        stage_map = {
            "CTS": {"wns": -0.01},
            "Global route": {"wns": -0.02},
        }
        result = check_cliff(stage_map, threshold=0.05, tns_threshold_pct=20.0)
        self.assertIsNone(result["tns_drop"])
        self.assertFalse(result["tns_detected"])
        self.assertFalse(result["detected"])

    def test_tns_zero_cts_tns_with_new_violations_detected(self):
        stage_map = {
            "CTS": {"wns": -0.01, "tns": 0.0},
            "Global route": {"wns": -0.02, "tns": -10.0},
        }
        result = check_cliff(stage_map, threshold=0.05, tns_threshold_pct=20.0)
        self.assertTrue(result["tns_detected"])
        self.assertTrue(result["detected"])


class TestPrintReportExitSignals(unittest.TestCase):
    def _run(self, structural, cliff, buffer_ratio_threshold=0.5):
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
        cliff = check_cliff(
            {"CTS": {"wns": -0.01}, "Global route": {"wns": -0.20}}, threshold=0.05
        )
        over_buffered, cliff_detected, out = self._run({}, cliff)
        self.assertTrue(cliff_detected)
        self.assertIn("CLIFF DETECTED", out)
        self.assertIn("WNS", out)

    def test_no_cliff_message_when_not_detected(self):
        cliff = check_cliff(
            {"CTS": {"wns": -0.05}, "Global route": {"wns": -0.06}}, threshold=0.05
        )
        over_buffered, cliff_detected, out = self._run({}, cliff)
        self.assertFalse(cliff_detected)
        self.assertIn("No CTS->GRT cliff detected", out)

    def test_flags_tns_cliff_and_shows_both_metrics(self):
        cliff = check_cliff(
            {
                "CTS": {"wns": -0.150, "tns": -306.65},
                "Global route": {"wns": -0.110, "tns": -492.21},
            },
            threshold=0.05,
            tns_threshold_pct=20.0,
        )
        over_buffered, cliff_detected, out = self._run({}, cliff)
        self.assertTrue(cliff_detected)
        self.assertIn("CLIFF DETECTED", out)
        self.assertIn("TNS", out)
        self.assertIn("CTS TNS", out)
        self.assertIn("GRT TNS", out)


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

    def test_gather_survives_non_dict_json_and_keeps_other_sections(self):
        with tempfile.TemporaryDirectory() as d:
            reports_dir = os.path.join(d, "reports")
            logs_dir = os.path.join(d, "logs")
            os.makedirs(reports_dir)
            os.makedirs(logs_dir)

            _write(
                os.path.join(reports_dir, "4_cts_final.rpt"),
                "tns max -0.02\nwns max -0.01\nworst slack max -0.01\n"
                + CTS_RPT_SKEW_FIXTURE,
            )
            _write(os.path.join(logs_dir, "4_1_cts.log"), CTS_LOG_FIXTURE)
            _write(os.path.join(logs_dir, "4_1_cts.json"), "null")

            rows, stage_map, structural = gather(reports_dir, logs_dir)

            self.assertEqual(structural["buffer_count"], 304)
            self.assertEqual(structural["sink_count"], 2167)
            self.assertAlmostEqual(structural["setup_skew"], 0.03)


class TestDeriveLogsDir(unittest.TestCase):
    def test_replaces_reports_path_component(self):
        logs_dir = derive_logs_dir("/repo/flow/reports/nangate45/ibex/base")
        self.assertEqual(logs_dir, "/repo/flow/logs/nangate45/ibex/base")

    def test_handles_relative_path_without_reports_substring(self):
        # Reproduces the real bug: a relative path given from within flow/
        # (e.g. "reports/nangate45/ibex/base") has no "/reports/" substring,
        # so a naive .replace("/reports/", "/logs/") is a silent no-op.
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as d:
                os.chdir(d)
                logs_dir = derive_logs_dir("reports/nangate45/ibex/base")
                self.assertTrue(
                    logs_dir.endswith(os.path.join("logs", "nangate45", "ibex", "base"))
                )
                self.assertNotIn("reports", logs_dir)
        finally:
            os.chdir(original_cwd)

    def test_falls_back_to_sibling_logs_dir_when_no_reports_component(self):
        logs_dir = derive_logs_dir("/some/other/layout/base")
        self.assertEqual(logs_dir, "/some/other/layout/logs")


class TestCliExitCodes(unittest.TestCase):
    SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cts_diagnostic.py")

    def _make_run(self, d, cliff_wns_drop=True):
        reports_dir = os.path.join(d, "reports")
        logs_dir = os.path.join(d, "logs")
        os.makedirs(reports_dir)
        os.makedirs(logs_dir)
        grt_wns = "-0.30" if cliff_wns_drop else "-0.02"
        grt_tns = "-1.20" if cliff_wns_drop else "-0.023"
        _write(
            os.path.join(reports_dir, "4_cts_final.rpt"),
            "tns max -0.02\nwns max -0.01\nworst slack max -0.01\n",
        )
        _write(
            os.path.join(reports_dir, "5_global_route.rpt"),
            f"tns max {grt_tns}\nwns max {grt_wns}\nworst slack max {grt_wns}\n",
        )
        return reports_dir, logs_dir

    def test_clean_run_exits_zero(self):
        with tempfile.TemporaryDirectory() as d:
            reports_dir, logs_dir = self._make_run(d, cliff_wns_drop=False)
            result = subprocess.run(
                [
                    sys.executable,
                    self.SCRIPT,
                    "--reports-dir",
                    reports_dir,
                    "--logs-dir",
                    logs_dir,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, EXIT_CLEAN)

    def test_cliff_detected_exits_one(self):
        with tempfile.TemporaryDirectory() as d:
            reports_dir, logs_dir = self._make_run(d, cliff_wns_drop=True)
            result = subprocess.run(
                [
                    sys.executable,
                    self.SCRIPT,
                    "--reports-dir",
                    reports_dir,
                    "--logs-dir",
                    logs_dir,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, EXIT_FINDING)

    def test_missing_reports_dir_exits_two(self):
        result = subprocess.run(
            [
                sys.executable,
                self.SCRIPT,
                "--reports-dir",
                "/nonexistent/reports/dir",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, EXIT_USAGE_ERROR)

    def test_missing_logs_dir_warns_but_still_reports(self):
        with tempfile.TemporaryDirectory() as d:
            reports_dir = os.path.join(d, "reports")
            os.makedirs(reports_dir)
            _write(
                os.path.join(reports_dir, "4_cts_final.rpt"),
                "tns max -0.02\nwns max -0.01\nworst slack max -0.01\n",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    self.SCRIPT,
                    "--reports-dir",
                    reports_dir,
                    "--logs-dir",
                    os.path.join(d, "nonexistent_logs"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertIn("WARNING", result.stderr)
            self.assertIn("logs directory not found", result.stderr)
            self.assertIn("CTS Quality Diagnostic", result.stdout)
            self.assertEqual(result.returncode, EXIT_CLEAN)


if __name__ == "__main__":
    unittest.main()
