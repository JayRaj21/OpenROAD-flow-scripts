#!/usr/bin/env python3
"""Unit tests for benchmark_dashboard.py — no Docker, no API, no live ORFS run."""

import argparse
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
            records, dropped_last_line = bd.load_records(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["git_sha"], "sha0")
            self.assertFalse(dropped_last_line)

    def test_append_is_append_only_across_multiple_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history.jsonl")
            bd.append_record(path, make_record("t0", "sha0", -0.1, 500.0, 100000))
            bd.append_record(path, make_record("t1", "sha1", -0.2, 490.0, 105000))
            bd.append_record(path, make_record("t2", "sha2", -0.05, 510.0, 98000))

            records, _ = bd.load_records(path)
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


class TestComputeDeltaNonNumeric(unittest.TestCase):
    def test_delta_none_when_both_non_numeric(self):
        self.assertIsNone(bd.compute_delta({"wns": "n/a"}, {"wns": "n/a"}, "wns"))

    def test_report_survives_both_non_numeric_history_records(self):
        records = [
            {
                "timestamp": "t0",
                "git_sha": "s0",
                "platform": "nangate45",
                "design": "ibex",
                "tag": "base",
                "stages": {"Finish": {"wns": "n/a", "fmax_mhz": "n/a", "hpwl": "n/a"}},
            },
            {
                "timestamp": "t1",
                "git_sha": "s1",
                "platform": "nangate45",
                "design": "ibex",
                "tag": "base",
                "stages": {"Finish": {"wns": "n/a", "fmax_mhz": "n/a", "hpwl": "n/a"}},
            },
        ]
        table_rows, latest = bd.build_report_rows(records, "Finish", 0.01, 1.0, 0.001)
        self.assertIsNone(table_rows[1]["wns_delta"])
        self.assertEqual(latest, [])
        bd.print_report("Finish", table_rows, "nangate45/ibex/base")


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

    def test_empty_latest_stage_metrics_flagged_as_regression(self):
        records = [
            make_record("t0", "s0", -0.05, 510.0, 90000),
            {
                "timestamp": "t1",
                "git_sha": "s1",
                "platform": "nangate45",
                "design": "ibex",
                "tag": "base",
                "stages": {},
            },
        ]
        table_rows, latest = bd.build_report_rows(records, "Finish", 0.01, 1.0, 0.001)
        self.assertTrue(any("no metrics" in r for r in latest))

    def test_empty_latest_stage_metrics_after_previous_success(self):
        regs = bd.detect_regressions(
            {"wns": -0.05, "fmax_mhz": 510.0}, {}, 0.01, 1.0, 0.001
        )
        self.assertTrue(any("no metrics" in r for r in regs))

    def test_no_regression_when_both_prev_and_cur_empty(self):
        regs = bd.detect_regressions(None, {}, 0.01, 1.0, 0.001)
        self.assertEqual(regs, [])


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
                records, _ = bd.load_records(history_file)
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

            records, _ = bd.load_records(history_file)
            _, latest = bd.build_report_rows(records, "Finish", 0.01, 1.0, 0.001)
            self.assertEqual(latest, [])

            history_file2 = os.path.join(tmp, "regressed.jsonl")
            bd.append_record(
                history_file2, make_record("t0", "s0", -0.05, 510.0, 90000)
            )
            bd.append_record(
                history_file2, make_record("t1", "s1", -0.30, 480.0, 90000)
            )
            records2, _ = bd.load_records(history_file2)
            _, latest2 = bd.build_report_rows(records2, "Finish", 0.01, 1.0, 0.001)
            self.assertTrue(len(latest2) > 0)


class TestLoadRecordsCorruptLines(unittest.TestCase):
    def test_corrupt_line_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history.jsonl")
            good0 = json.dumps(make_record("t0", "s0", -0.05, 510.0, 90000))
            good1 = json.dumps(make_record("t1", "s1", -0.06, 508.0, 91000))
            with open(path, "w") as f:
                f.write(good0 + "\n")
                f.write("{not valid json truncated mid-rec\n")
                f.write(good1 + "\n")

            with mock.patch("sys.stderr") as mock_stderr:
                records, dropped_last_line = bd.load_records(path)

            self.assertEqual(len(records), 2)
            self.assertEqual([r["git_sha"] for r in records], ["s0", "s1"])
            self.assertFalse(dropped_last_line)
            written = "".join(c.args[0] for c in mock_stderr.write.call_args_list)
            self.assertIn("line 2", written)

    def test_all_corrupt_lines_returns_empty_not_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history.jsonl")
            with open(path, "w") as f:
                f.write("{{{not json\n")
                f.write("also not json\n")
            with mock.patch("sys.stderr"):
                records, dropped_last_line = bd.load_records(path)
            self.assertEqual(records, [])
            self.assertTrue(dropped_last_line)

    def test_report_cli_survives_corrupt_history_line(self):
        util_dir = os.path.dirname(os.path.abspath(__file__))
        tag = "corrupt-line-test"
        history_file = os.path.join(
            util_dir, "benchmark_history", f"nangate45__ibex__{tag}.jsonl"
        )
        try:
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            with open(history_file, "w") as f:
                f.write(json.dumps(make_record("t0", "s0", -0.05, 510.0, 90000)) + "\n")
                f.write("not valid json\n")
                f.write(json.dumps(make_record("t1", "s1", -0.06, 508.0, 91000)) + "\n")

            proc = subprocess.run(
                [
                    sys.executable,
                    "benchmark_dashboard.py",
                    "report",
                    "--platform",
                    "nangate45",
                    "--design",
                    "ibex",
                    "--tag",
                    tag,
                ],
                cwd=util_dir,
                capture_output=True,
                text=True,
            )
            self.assertIn(proc.returncode, (0, 1))
            self.assertIn("WARNING", proc.stderr)
            self.assertIn("s0", proc.stdout)
            self.assertIn("s1", proc.stdout)
        finally:
            if os.path.isfile(history_file):
                os.remove(history_file)

    def test_report_cli_fails_when_last_line_is_corrupt(self):
        util_dir = os.path.dirname(os.path.abspath(__file__))
        tag = "corrupt-last-line-test"
        history_file = os.path.join(
            util_dir, "benchmark_history", f"nangate45__ibex__{tag}.jsonl"
        )
        try:
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            with open(history_file, "w") as f:
                f.write(json.dumps(make_record("t0", "s0", -0.05, 510.0, 90000)) + "\n")
                f.write(json.dumps(make_record("t1", "s1", -0.06, 508.0, 91000)) + "\n")
                f.write('{"truncated": tr\n')

            proc = subprocess.run(
                [
                    sys.executable,
                    "benchmark_dashboard.py",
                    "report",
                    "--platform",
                    "nangate45",
                    "--design",
                    "ibex",
                    "--tag",
                    tag,
                ],
                cwd=util_dir,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("corrupt/truncated record", proc.stderr)
        finally:
            if os.path.isfile(history_file):
                os.remove(history_file)

    def test_load_records_flags_dropped_last_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps(make_record("t0", "s0", -0.05, 510.0, 90000)) + "\n")
                f.write("not valid json at all\n")
            with mock.patch("sys.stderr"):
                records, dropped_last_line = bd.load_records(path)
            self.assertEqual(len(records), 1)
            self.assertTrue(dropped_last_line)

    def test_non_dict_json_line_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history.jsonl")
            with open(path, "w") as f:
                f.write("null\n")
                f.write(json.dumps([1, 2, 3]) + "\n")
                f.write(json.dumps(make_record("t0", "s0", -0.05, 510.0, 90000)) + "\n")
            with mock.patch("sys.stderr"):
                records, dropped_last_line = bd.load_records(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["git_sha"], "s0")
            self.assertFalse(dropped_last_line)

    def test_null_nested_stage_value_treated_as_valid_empty_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history.jsonl")
            rec = make_record("t0", "s0", -0.05, 510.0, 90000)
            rec["stages"]["Global route"] = None
            with open(path, "w") as f:
                f.write(json.dumps(rec) + "\n")
            records, dropped_last_line = bd.load_records(path)
            self.assertEqual(len(records), 1)
            self.assertFalse(dropped_last_line)
            table_rows, _ = bd.build_report_rows(
                records, "Global route", 0.01, 1.0, 0.001
            )
            self.assertEqual(table_rows[0]["metrics"], {})

    def test_load_records_flags_dropped_last_line_with_trailing_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps(make_record("t0", "s0", -0.05, 510.0, 90000)) + "\n")
                f.write("not valid json at all\n")
                f.write("\n")
            with mock.patch("sys.stderr"):
                records, dropped_last_line = bd.load_records(path)
            self.assertEqual(len(records), 1)
            self.assertTrue(dropped_last_line)

    def test_report_cli_fails_on_all_garbage_history(self):
        util_dir = os.path.dirname(os.path.abspath(__file__))
        tag = "all-garbage-test"
        history_file = os.path.join(
            util_dir, "benchmark_history", f"nangate45__ibex__{tag}.jsonl"
        )
        try:
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            with open(history_file, "w") as f:
                f.write("not valid json\n")
                f.write("also not valid json\n")

            proc = subprocess.run(
                [
                    sys.executable,
                    "benchmark_dashboard.py",
                    "report",
                    "--platform",
                    "nangate45",
                    "--design",
                    "ibex",
                    "--tag",
                    tag,
                ],
                cwd=util_dir,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("corrupt/truncated record", proc.stderr)
        finally:
            if os.path.isfile(history_file):
                os.remove(history_file)

    def test_load_records_takes_shared_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history.jsonl")
            bd.append_record(path, make_record("t0", "s0", -0.05, 510.0, 90000))

            flock_calls = []
            real_flock = bd.fcntl.flock

            def spy_flock(fd, op):
                flock_calls.append(op)
                return real_flock(fd, op)

            with mock.patch("benchmark_dashboard.fcntl.flock", side_effect=spy_flock):
                records, _ = bd.load_records(path)
            self.assertEqual(len(records), 1)
            self.assertIn(bd.fcntl.LOCK_SH, flock_calls)
            self.assertIn(bd.fcntl.LOCK_UN, flock_calls)


class TestHtmlEscaping(unittest.TestCase):
    def test_malicious_label_and_stage_are_escaped(self):
        malicious = "</title></head><body><script>alert(1)</script>"
        records = [
            make_record("t0", "s0", -0.10, 500.0, 100000),
            make_record("t1", "s1", -0.05, 510.0, 90000),
        ]
        table_rows, _ = bd.build_report_rows(records, "Finish", 0.01, 1.0, 0.001)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.html")
            bd.render_html(records, malicious, table_rows, malicious, out_path)
            with open(out_path) as f:
                content = f.read()
            self.assertNotIn("<script>", content)
            self.assertIn("&lt;script&gt;", content)

    def test_malicious_git_sha_is_escaped(self):
        records = [
            make_record("t0", "<img src=x onerror=alert(1)>", -0.10, 500.0, 100000),
            make_record("t1", "s1", -0.05, 510.0, 90000),
        ]
        table_rows, _ = bd.build_report_rows(records, "Finish", 0.01, 1.0, 0.001)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.html")
            bd.render_html(records, "Finish", table_rows, "label", out_path)
            with open(out_path) as f:
                content = f.read()
            self.assertNotIn("<img src=x", content)
            self.assertIn("&lt;img", content)


class TestLastWindowKeepsFullHistoryBest(unittest.TestCase):
    def test_last_does_not_narrow_best_ever_scope(self):
        records = [
            make_record("t0", "s0", -0.05, 510.0, 90000),
            make_record("t1", "s1", -0.06, 508.0, 91000),
            make_record("t2", "s2", -0.06, 508.0, 91000),
            make_record("t3", "s3", -0.06, 508.0, 91000),
            make_record("t4", "s4", -0.055, 505.0, 92000),
        ]
        table_rows, latest = bd.build_report_rows(records, "Finish", 0.01, 1.0, 0.001)
        display_rows = table_rows[-2:]

        self.assertEqual(len(display_rows), 2)
        self.assertIn("worse-than-best-WNS", display_rows[-1]["flags"])
        self.assertIn("worse-than-best-Fmax", display_rows[-1]["flags"])
        self.assertIn("worse-than-best-HPWL", display_rows[-1]["flags"])
        # confirm the "true best" record (t0) is outside the 2-record window
        self.assertNotIn(records[0], [row["record"] for row in display_rows])

    def test_report_cli_last_flag_reports_true_best_ever(self):
        util_dir = os.path.dirname(os.path.abspath(__file__))
        tag = "last-window-test"
        history_file = os.path.join(
            util_dir, "benchmark_history", f"nangate45__ibex__{tag}.jsonl"
        )
        try:
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            with open(history_file, "w") as f:
                for r in [
                    make_record("t0", "s0", -0.05, 510.0, 90000),
                    make_record("t1", "s1", -0.06, 508.0, 91000),
                    make_record("t2", "s2", -0.06, 508.0, 91000),
                    make_record("t3", "s3", -0.06, 508.0, 91000),
                    make_record("t4", "s4", -0.055, 505.0, 92000),
                ]:
                    f.write(json.dumps(r) + "\n")

            proc = subprocess.run(
                [
                    sys.executable,
                    "benchmark_dashboard.py",
                    "report",
                    "--platform",
                    "nangate45",
                    "--design",
                    "ibex",
                    "--tag",
                    tag,
                    "--last",
                    "2",
                ],
                cwd=util_dir,
                capture_output=True,
                text=True,
            )
            self.assertIn("worse-than-best-WNS", proc.stdout)
            self.assertNotIn("s0", proc.stdout)
        finally:
            if os.path.isfile(history_file):
                os.remove(history_file)


class TestResolveDirsValidation(unittest.TestCase):
    def test_raises_clear_error_when_platform_design_undeterminable(self):
        args = argparse.Namespace(
            platform=None,
            design=None,
            tag="base",
            reports_dir="/tmp/x",
            logs_dir=None,
            flow_dir="/tmp",
        )
        with self.assertRaises(SystemExit) as ctx:
            bd.resolve_dirs(args)
        self.assertIn("could not determine", str(ctx.exception))

    def test_reports_dir_derives_tag_from_path_when_not_passed(self):
        args = argparse.Namespace(
            platform=None,
            design=None,
            tag=None,
            reports_dir="/tmp/x/reports/nangate45/ibex/hardened",
            logs_dir=None,
            flow_dir="/tmp",
        )
        bd.resolve_dirs(args)
        self.assertEqual(args.platform, "nangate45")
        self.assertEqual(args.design, "ibex")
        self.assertEqual(args.tag, "hardened")

    def test_reports_dir_explicit_tag_overrides_path_derivation(self):
        args = argparse.Namespace(
            platform=None,
            design=None,
            tag="explicit-tag",
            reports_dir="/tmp/x/reports/nangate45/ibex/hardened",
            logs_dir=None,
            flow_dir="/tmp",
        )
        bd.resolve_dirs(args)
        self.assertEqual(args.tag, "explicit-tag")

    def test_raises_clear_error_when_reports_dir_one_level_too_high(self):
        args = argparse.Namespace(
            platform=None,
            design=None,
            tag=None,
            reports_dir="/tmp/flow/reports/nangate45/ibex",
            logs_dir=None,
            flow_dir="/tmp",
        )
        with self.assertRaises(SystemExit) as ctx:
            bd.resolve_dirs(args)
        self.assertIn("does not look like", str(ctx.exception))

    def test_reports_dir_with_no_reports_component_still_works(self):
        args = argparse.Namespace(
            platform=None,
            design=None,
            tag=None,
            reports_dir="/tmp/artifacts/nangate45/ibex/base",
            logs_dir=None,
            flow_dir="/tmp",
        )
        bd.resolve_dirs(args)
        self.assertEqual(args.platform, "nangate45")
        self.assertEqual(args.design, "ibex")
        self.assertEqual(args.tag, "base")

    def test_relative_three_component_path_still_works(self):
        args = argparse.Namespace(
            platform=None,
            design=None,
            tag=None,
            reports_dir="nangate45/ibex/base",
            logs_dir=None,
            flow_dir="/tmp",
        )
        bd.resolve_dirs(args)
        self.assertEqual(args.platform, "nangate45")
        self.assertEqual(args.design, "ibex")
        self.assertEqual(args.tag, "base")


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


class TestFmtNonNumeric(unittest.TestCase):
    def test_fmt_returns_missing_for_non_numeric_value(self):
        self.assertEqual(bd.fmt("n/a", "{:+.3f}"), "—")

    def test_fmt_returns_missing_for_none(self):
        self.assertEqual(bd.fmt(None, "{:+.3f}"), "—")

    def test_fmt_formats_numeric_value(self):
        self.assertEqual(bd.fmt(-0.1, "{:+.3f}"), "-0.100")

    def test_fmt_delta_returns_missing_for_non_numeric_value(self):
        self.assertEqual(bd.fmt_delta("n/a", "{:+.3f}"), "—")

    def test_fmt_delta_formats_numeric_value(self):
        self.assertEqual(bd.fmt_delta(0.1, "{:+.3f}"), "+0.100")

    def test_fmt_returns_missing_for_bool_value(self):
        self.assertEqual(bd.fmt(True, "{:+.3f}"), "—")
        self.assertEqual(bd.fmt(False, "{:+.3f}"), "—")

    def test_fmt_delta_returns_missing_for_bool_value(self):
        self.assertEqual(bd.fmt_delta(True, "{:+.3f}"), "—")

    def test_print_report_survives_non_numeric_metric(self):
        records = [
            {
                "timestamp": "t0",
                "git_sha": "s0",
                "platform": "nangate45",
                "design": "ibex",
                "tag": "base",
                "stages": {"Finish": {"wns": "n/a", "fmax_mhz": 500.0, "hpwl": 100000}},
            }
        ]
        table_rows, _ = bd.build_report_rows(records, "Finish", 0.01, 1.0, 0.001)
        bd.print_report("Finish", table_rows, "nangate45/ibex/base")


class TestCmdRecordCollectFailure(unittest.TestCase):
    def test_cmd_record_reports_clear_error_on_collect_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = os.path.join(tmp, "reports")
            logs_dir = os.path.join(tmp, "logs")
            os.makedirs(reports_dir)
            os.makedirs(logs_dir)
            args = argparse.Namespace(platform="nangate45", design="ibex", tag="base")
            flow_util_dir = os.path.dirname(os.path.abspath(__file__))

            with mock.patch(
                "benchmark_dashboard.collect", side_effect=ValueError("bad report")
            ):
                with self.assertRaises(SystemExit) as ctx:
                    bd.cmd_record(
                        args, tmp, flow_util_dir, reports_dir, logs_dir, "label"
                    )
            self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
