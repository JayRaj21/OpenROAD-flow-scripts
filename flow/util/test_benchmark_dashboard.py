#!/usr/bin/env python3
"""Unit tests for benchmark_dashboard.py — no Docker, no API, no live ORFS run."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_dashboard as bd


def make_record(timestamp, sha, wns, fmax_mhz, hpwl, grt_overflow=0.0):
    return {
        "timestamp": timestamp,
        "git_sha": sha,
        "platform": "nangate45",
        "design": "ibex",
        "tag": "base",
        "stages": {
            "Finish": {
                "wns": wns,
                "tns": wns * 10 if wns is not None else None,
                "fmax_mhz": fmax_mhz,
                "hpwl": hpwl,
                "grt_overflow": grt_overflow,
            }
        },
    }


class TestAppendRecord(unittest.TestCase):
    def test_append_creates_dir_and_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "history.jsonl")
            bd.append_record(path, make_record("t0", "sha0", -0.1, 500.0, 100000))
            self.assertTrue(os.path.isfile(path))
            records = bd.load_records(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["git_sha"], "sha0")

    def test_append_is_append_only_across_multiple_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history.jsonl")
            bd.append_record(path, make_record("t0", "sha0", -0.1, 500.0, 100000))
            bd.append_record(path, make_record("t1", "sha1", -0.2, 490.0, 105000))
            bd.append_record(path, make_record("t2", "sha2", -0.05, 510.0, 98000))

            records = bd.load_records(path)
            self.assertEqual(len(records), 3)
            self.assertEqual([r["git_sha"] for r in records], ["sha0", "sha1", "sha2"])

            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 3)
            for line in lines:
                json.loads(line)

    def test_prior_lines_unmodified_after_new_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history.jsonl")
            bd.append_record(path, make_record("t0", "sha0", -0.1, 500.0, 100000))
            with open(path) as f:
                first_line_before = f.readlines()[0]
            bd.append_record(path, make_record("t1", "sha1", -0.2, 490.0, 105000))
            with open(path) as f:
                first_line_after = f.readlines()[0]
            self.assertEqual(first_line_before, first_line_after)


class TestGitSha(unittest.TestCase):
    def test_git_sha_returns_string_in_real_repo(self):
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sha = bd.git_sha(repo_dir)
        self.assertIsInstance(sha, str)
        self.assertEqual(len(sha), 40)

    def test_git_sha_returns_none_on_failure(self):
        with mock.patch(
            "benchmark_dashboard.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            self.assertIsNone(bd.git_sha("/nonexistent"))


class TestComputeDelta(unittest.TestCase):
    def test_delta_none_when_missing(self):
        self.assertIsNone(bd.compute_delta(None, {"wns": -0.1}, "wns"))
        self.assertIsNone(bd.compute_delta({"wns": -0.1}, {}, "wns"))

    def test_delta_computed(self):
        self.assertAlmostEqual(
            bd.compute_delta({"wns": -0.2}, {"wns": -0.1}, "wns"), 0.1
        )


class TestDetectRegressions(unittest.TestCase):
    def test_wns_regression_flagged(self):
        prev = {"wns": -0.10, "fmax_mhz": 500.0, "grt_overflow": 0.0}
        cur = {"wns": -0.15, "fmax_mhz": 500.0, "grt_overflow": 0.0}
        regs = bd.detect_regressions(prev, cur, 0.01, 1.0, 0.001)
        self.assertTrue(any("WNS" in r for r in regs))

    def test_wns_within_threshold_not_flagged(self):
        prev = {"wns": -0.10, "fmax_mhz": 500.0, "grt_overflow": 0.0}
        cur = {"wns": -0.105, "fmax_mhz": 500.0, "grt_overflow": 0.0}
        regs = bd.detect_regressions(prev, cur, 0.01, 1.0, 0.001)
        self.assertFalse(any("WNS" in r for r in regs))

    def test_fmax_regression_flagged(self):
        prev = {"wns": 0.0, "fmax_mhz": 500.0, "grt_overflow": 0.0}
        cur = {"wns": 0.0, "fmax_mhz": 480.0, "grt_overflow": 0.0}
        regs = bd.detect_regressions(prev, cur, 0.01, 1.0, 0.001)
        self.assertTrue(any("Fmax" in r for r in regs))

    def test_fmax_within_threshold_not_flagged(self):
        prev = {"wns": 0.0, "fmax_mhz": 500.0, "grt_overflow": 0.0}
        cur = {"wns": 0.0, "fmax_mhz": 498.0, "grt_overflow": 0.0}
        regs = bd.detect_regressions(prev, cur, 0.01, 1.0, 0.001)
        self.assertFalse(any("Fmax" in r for r in regs))

    def test_overflow_regression_flagged(self):
        prev = {"wns": 0.0, "fmax_mhz": 500.0, "grt_overflow": 0.0}
        cur = {"wns": 0.0, "fmax_mhz": 500.0, "grt_overflow": 0.01}
        regs = bd.detect_regressions(prev, cur, 0.01, 1.0, 0.001)
        self.assertTrue(any("overflow" in r for r in regs))

    def test_improvement_not_flagged(self):
        prev = {"wns": -0.2, "fmax_mhz": 480.0, "grt_overflow": 0.01}
        cur = {"wns": -0.1, "fmax_mhz": 500.0, "grt_overflow": 0.0}
        regs = bd.detect_regressions(prev, cur, 0.01, 1.0, 0.001)
        self.assertEqual(regs, [])

    def test_no_prev_no_regressions(self):
        cur = {"wns": -0.1, "fmax_mhz": 500.0, "grt_overflow": 0.0}
        regs = bd.detect_regressions(None, cur, 0.01, 1.0, 0.001)
        self.assertEqual(regs, [])


class TestBestEver(unittest.TestCase):
    def test_best_ever_lower(self):
        records = [
            make_record("t0", "s0", -0.1, 500, 100),
            make_record("t1", "s1", -0.1, 500, 90),
            make_record("t2", "s2", -0.1, 500, 120),
        ]
        self.assertEqual(bd.best_ever(records, "Finish", "hpwl", "lower"), 90)

    def test_best_ever_higher(self):
        records = [
            make_record("t0", "s0", -0.2, 480, 100),
            make_record("t1", "s1", -0.05, 510, 100),
        ]
        self.assertEqual(bd.best_ever(records, "Finish", "wns", "higher"), -0.05)

    def test_best_ever_empty(self):
        self.assertIsNone(bd.best_ever([], "Finish", "wns", "higher"))


class TestBuildReportRows(unittest.TestCase):
    def test_single_record_no_regressions(self):
        records = [make_record("t0", "s0", -0.1, 500, 100000)]
        table_rows, latest = bd.build_report_rows(records, "Finish", 0.01, 1.0, 0.001)
        self.assertEqual(len(table_rows), 1)
        self.assertEqual(latest, [])
        self.assertIsNone(table_rows[0]["wns_delta"])

    def test_regression_detected_on_latest_record(self):
        records = [
            make_record("t0", "s0", -0.10, 500.0, 100000),
            make_record("t1", "s1", -0.30, 500.0, 100000),
        ]
        table_rows, latest = bd.build_report_rows(records, "Finish", 0.01, 1.0, 0.001)
        self.assertEqual(len(table_rows), 2)
        self.assertTrue(any("WNS" in r for r in latest))
        self.assertAlmostEqual(table_rows[1]["wns_delta"], -0.20)

    def test_flags_worse_than_best(self):
        records = [
            make_record("t0", "s0", -0.05, 510.0, 90000),
            make_record("t1", "s1", -0.05, 510.0, 90000),
            make_record("t2", "s2", -0.20, 480.0, 150000),
        ]
        table_rows, _ = bd.build_report_rows(records, "Finish", 0.01, 1.0, 0.001)
        flags = table_rows[2]["flags"]
        self.assertIn("worse-than-best-WNS", flags)
        self.assertIn("worse-than-best-Fmax", flags)
        self.assertIn("worse-than-best-HPWL", flags)

    def test_no_regression_when_improving(self):
        records = [
            make_record("t0", "s0", -0.30, 480.0, 150000),
            make_record("t1", "s1", -0.05, 510.0, 90000),
        ]
        _, latest = bd.build_report_rows(records, "Finish", 0.01, 1.0, 0.001)
        self.assertEqual(latest, [])


class TestCliRecordAndReport(unittest.TestCase):
    def _run(self, args, cwd):
        return subprocess.run(
            [sys.executable, "benchmark_dashboard.py"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
        )

    def _util_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

    def _make_fake_flow(self, tmp):
        reports = os.path.join(tmp, "flow", "reports", "nangate45", "ibex", "base")
        logs = os.path.join(tmp, "flow", "logs", "nangate45", "ibex", "base")
        os.makedirs(reports)
        os.makedirs(logs)
        with open(os.path.join(reports, "6_finish.rpt"), "w") as f:
            f.write(
                "tns max -1.0\nwns max -0.10\nworst slack max -0.10\nfmax = 500.0\n"
            )
        return os.path.join(tmp, "flow")

    def test_record_cli_writes_history_file(self):
        util_dir = self._util_dir()
        with tempfile.TemporaryDirectory() as tmp:
            flow_dir = self._make_fake_flow(tmp)
            history_file = os.path.join(
                util_dir, "benchmark_history", "nangate45__ibex__base.jsonl"
            )
            if os.path.isfile(history_file):
                os.remove(history_file)
            try:
                proc = self._run(
                    [
                        "record",
                        "--platform",
                        "nangate45",
                        "--design",
                        "ibex",
                        "--tag",
                        "base",
                        "--flow-dir",
                        flow_dir,
                    ],
                    util_dir,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertTrue(os.path.isfile(history_file))
                records = bd.load_records(history_file)
                self.assertEqual(len(records), 1)
                self.assertAlmostEqual(records[0]["stages"]["Finish"]["wns"], -0.10)
            finally:
                if os.path.isfile(history_file):
                    os.remove(history_file)

    def test_report_exit_code_regression_vs_clean(self):
        util_dir = self._util_dir()
        with tempfile.TemporaryDirectory() as tmp:
            history_file = os.path.join(tmp, "clean.jsonl")
            bd.append_record(history_file, make_record("t0", "s0", -0.05, 510.0, 90000))
            bd.append_record(history_file, make_record("t1", "s1", -0.06, 508.0, 91000))

            records = bd.load_records(history_file)
            _, latest = bd.build_report_rows(records, "Finish", 0.01, 1.0, 0.001)
            self.assertEqual(latest, [])

            history_file2 = os.path.join(tmp, "regressed.jsonl")
            bd.append_record(
                history_file2, make_record("t0", "s0", -0.05, 510.0, 90000)
            )
            bd.append_record(
                history_file2, make_record("t1", "s1", -0.30, 480.0, 90000)
            )
            records2 = bd.load_records(history_file2)
            _, latest2 = bd.build_report_rows(records2, "Finish", 0.01, 1.0, 0.001)
            self.assertTrue(len(latest2) > 0)


class TestHtmlOutput(unittest.TestCase):
    def test_html_written_and_non_empty_for_two_records(self):
        records = [
            make_record("2026-08-01T00:00:00+00:00", "s0", -0.10, 500.0, 100000),
            make_record("2026-08-02T00:00:00+00:00", "s1", -0.05, 510.0, 90000),
        ]
        table_rows, _ = bd.build_report_rows(records, "Finish", 0.01, 1.0, 0.001)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.html")
            bd.render_html(
                records, "Finish", table_rows, "nangate45/ibex/base", out_path
            )
            self.assertTrue(os.path.isfile(out_path))
            with open(out_path) as f:
                content = f.read()
            self.assertGreater(len(content), 0)
            self.assertIn("<svg", content)
            self.assertIn("<table>", content)
            self.assertIn("s0", content)
            self.assertIn("s1", content)
            self.assertNotIn("http://", content)
            self.assertNotIn("https://", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
