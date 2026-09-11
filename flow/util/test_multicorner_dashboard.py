#!/usr/bin/env python3
"""Unit tests for multicorner_dashboard.py — no Docker, no OpenSTA, synthetic fixtures only."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multicorner_dashboard import (
    build_table,
    collect_per_corner,
    default_stage,
    find_multicorner_reports,
    parse_clock_skew,
    worst_corner,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _rpt_text(corner, tns, wns, worst_slack, skews=None):
    """Build a fixture matching report_multicorner_timing.tcl's real output:
    tns/wns/worst_slack lines it writes itself, plus report_clock_skew's own
    native per-clock "<value> setup|hold skew" lines (search/ClkSkew.cc),
    not an invented "Worst skew" summary line.
    """
    lines = [
        "==========================================================================",
        f"Corner: {corner}",
        f"cts final report_tns (corner {corner})",
        "--------------------------------------------------------------------------",
        f"tns max {tns}",
        "",
        "==========================================================================",
        f"cts final report_wns (corner {corner})",
        "--------------------------------------------------------------------------",
        f"wns max {wns}",
        "",
        "==========================================================================",
        f"cts final report_worst_slack (corner {corner})",
        "--------------------------------------------------------------------------",
        f"worst slack max {worst_slack}",
        "",
    ]
    if skews is not None:
        lines += [
            "==========================================================================",
            f"cts final report_clock_skew -corner {corner}",
            "--------------------------------------------------------------------------",
        ]
        for clk_name, value, setup_hold in skews:
            lines.append(f"Clock {clk_name}")
            lines.append(f"  {value} {setup_hold} skew")
        lines.append("")
    return "\n".join(lines)


class TestParsing(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, name, text):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            f.write(text)
        return path

    def test_find_multicorner_reports_matches_and_extracts_corner(self):
        self._write("4_cts_final_multicorner_tt.rpt", _rpt_text("tt", -1.0, -0.5, -0.5))
        self._write("4_cts_final_multicorner_ff.rpt", _rpt_text("ff", -2.0, -1.5, -1.5))
        self._write("6_finish_multicorner_tt.rpt", _rpt_text("tt", -0.1, -0.1, -0.1))

        found = find_multicorner_reports(self.tmpdir, "4_cts_final")
        self.assertEqual(set(found.keys()), {"tt", "ff"})

    def test_find_multicorner_reports_ignores_non_multicorner_files(self):
        self._write("4_cts_final.rpt", _rpt_text("merged", -1.0, -0.5, -0.5))
        self._write("4_cts_final_multicorner_tt.rpt", _rpt_text("tt", -1.0, -0.5, -0.5))
        found = find_multicorner_reports(self.tmpdir, "4_cts_final")
        self.assertEqual(set(found.keys()), {"tt"})

    def test_parse_clock_skew_picks_largest_magnitude_across_clocks(self):
        path = self._write(
            "4_cts_final_multicorner_tt.rpt",
            _rpt_text(
                "tt",
                -1.0,
                -0.5,
                -0.5,
                skews=[("clk1", "0.050", "setup"), ("clk2", "-0.234", "hold")],
            ),
        )
        self.assertAlmostEqual(parse_clock_skew(path), -0.234)

    def test_parse_clock_skew_absent(self):
        path = self._write(
            "4_cts_final_multicorner_tt.rpt", _rpt_text("tt", -1.0, -0.5, -0.5)
        )
        self.assertIsNone(parse_clock_skew(path))

    def test_find_multicorner_reports_handles_underscore_corner_names(self):
        self._write(
            "4_cts_final_multicorner_ss_0p9v_125c.rpt",
            _rpt_text("ss_0p9v_125c", -1.0, -0.5, -0.5),
        )
        self._write(
            "4_cts_final_multicorner_ff_1p1v_n40c.rpt",
            _rpt_text("ff_1p1v_n40c", -2.0, -1.5, -1.5),
        )
        found = find_multicorner_reports(self.tmpdir, "4_cts_final")
        self.assertEqual(set(found.keys()), {"ss_0p9v_125c", "ff_1p1v_n40c"})

    def test_default_stage_picks_highest_numeric_prefix(self):
        self._write("4_cts_final_multicorner_tt.rpt", _rpt_text("tt", -1.0, -0.5, -0.5))
        self._write("6_finish_multicorner_tt.rpt", _rpt_text("tt", -0.1, -0.1, -0.1))
        self.assertEqual(default_stage(self.tmpdir), "6_finish")

    def test_default_stage_none_when_no_reports(self):
        self.assertIsNone(default_stage(self.tmpdir))

    def test_default_stage_tie_on_numeric_prefix_is_deterministic(self):
        """Two stage labels sharing the numeric prefix "4" used to be stored
        in a `set` and broken by hash-randomized iteration order (varies with
        PYTHONHASHSEED). default_stage() must now pick the same answer every
        time regardless of hash seed -- verified here both by running the
        selection many times in-process (proving it does not depend on
        Python's per-run randomized set/str hash salt within calls that
        rebuild the set each time) and by re-invoking this exact test in a
        subprocess under several different PYTHONHASHSEED values.
        """
        # "4_cts_final" is written after "4_cts_pre-repair-timing", so by the
        # mtime tie-break it should always win.
        self._write(
            "4_cts_pre-repair-timing_multicorner_tt.rpt",
            _rpt_text("tt", -1.0, -0.5, -0.5),
        )
        self._write("4_cts_final_multicorner_tt.rpt", _rpt_text("tt", -1.0, -0.5, -0.5))

        results = {default_stage(self.tmpdir) for _ in range(50)}
        self.assertEqual(
            results,
            {"4_cts_final"},
            "default_stage() returned different answers across repeated calls "
            "with identical inputs",
        )

    def test_default_stage_tie_deterministic_across_pythonhashseed(self):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multicorner_dashboard.py")
        picks = set()
        for seed in ("0", "1", "42"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; sys.path.insert(0, %r); "
                        "from multicorner_dashboard import default_stage; "
                        "print(default_stage(%r))"
                    )
                    % (os.path.dirname(script), self.tmpdir),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            picks.add(proc.stdout.strip())
        self.assertEqual(
            len(picks),
            1,
            f"default_stage() picked different stages across PYTHONHASHSEED values: {picks}",
        )

    def test_default_stage_tie_warns_on_stderr(self):
        self._write(
            "4_cts_pre-repair-timing_multicorner_tt.rpt",
            _rpt_text("tt", -1.0, -0.5, -0.5),
        )
        self._write("4_cts_final_multicorner_tt.rpt", _rpt_text("tt", -1.0, -0.5, -0.5))

        old_stderr = sys.stderr
        sys.stderr = captured = __import__("io").StringIO()
        try:
            default_stage(self.tmpdir)
        finally:
            sys.stderr = old_stderr
        self.assertIn("Warning", captured.getvalue())

    def test_collect_per_corner_uses_pr_metrics_parse_rpt(self):
        self._write(
            "4_cts_final_multicorner_tt.rpt",
            _rpt_text(
                "tt",
                tns=-5.0,
                wns=-1.2,
                worst_slack=-1.2,
                skews=[("clk", "-0.100", "setup")],
            ),
        )
        self._write(
            "4_cts_final_multicorner_ss.rpt",
            _rpt_text(
                "ss",
                tns=-9.0,
                wns=-2.5,
                worst_slack=-2.5,
                skews=[("clk", "-0.300", "setup")],
            ),
        )
        data = collect_per_corner(self.tmpdir, "4_cts_final")
        self.assertEqual(set(data.keys()), {"tt", "ss"})
        self.assertAlmostEqual(data["tt"]["wns"], -1.2)
        self.assertAlmostEqual(data["tt"]["tns"], -5.0)
        self.assertAlmostEqual(data["tt"]["worst_slack"], -1.2)
        self.assertAlmostEqual(data["tt"]["clock_skew"], -0.1)
        self.assertAlmostEqual(data["ss"]["wns"], -2.5)


class TestWorstCorner(unittest.TestCase):
    def test_worst_corner_wns_is_most_negative(self):
        data = {"tt": {"wns": -0.5}, "ss": {"wns": -2.1}, "ff": {"wns": -0.1}}
        self.assertEqual(worst_corner(data, "wns"), "ss")

    def test_worst_corner_tns_is_most_negative(self):
        data = {"tt": {"tns": -5.0}, "ss": {"tns": -20.0}}
        self.assertEqual(worst_corner(data, "tns"), "ss")

    def test_worst_corner_missing_metric_returns_none(self):
        data = {"tt": {"wns": -0.5}}
        self.assertIsNone(worst_corner(data, "clock_skew"))

    def test_worst_corner_skips_corners_missing_the_metric(self):
        data = {"tt": {"wns": -0.5}, "ss": {}}
        self.assertEqual(worst_corner(data, "wns"), "tt")

    def test_worst_corner_clock_skew_is_largest_magnitude_either_sign(self):
        # a large positive skew is worse than a smaller negative one
        data = {"tt": {"clock_skew": -0.05}, "ss": {"clock_skew": 0.3}}
        self.assertEqual(worst_corner(data, "clock_skew"), "ss")


class TestBuildTable(unittest.TestCase):
    def test_table_marks_worst_corner(self):
        data = {
            "tt": {"wns": -0.5, "tns": -5.0, "worst_slack": -0.5},
            "ss": {"wns": -2.1, "tns": -20.0, "worst_slack": -2.1},
        }
        table = build_table(data, "nangate45/aes/base / 4_cts_final")
        self.assertIn("ss", table)
        self.assertIn("tt", table)
        ss_wns_line = [l for l in table.splitlines() if l.startswith("WNS")][0]
        self.assertIn("(worst)", ss_wns_line)
        # the worst value (-2.100) must carry the marker, not the better one
        worst_idx = ss_wns_line.index("(worst)")
        self.assertIn("-2.100", ss_wns_line[:worst_idx])

    def test_table_handles_missing_clock_skew(self):
        data = {"tt": {"wns": -0.5, "tns": -5.0, "worst_slack": -0.5}}
        table = build_table(data, "stage")
        self.assertNotIn("Clock skew", table)


class TestTclSyntax(unittest.TestCase):
    """report_multicorner_timing.tcl parses as valid Tcl (behavior, not just text)."""

    def test_tcl_script_is_syntactically_valid(self):
        tclsh = shutil.which("tclsh")
        if not tclsh:
            self.skipTest("tclsh not available")

        script = os.path.join(
            REPO_ROOT, "flow", "scripts", "report_multicorner_timing.tcl"
        )
        self.assertTrue(os.path.isfile(script))

        proc = subprocess.run(
            [tclsh],
            input=f'source "{script}"\n',
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Undefined ORFS procs/env vars (env_var_exists_and_non_empty, REPORTS_DIR,
        # etc.) are expected to error at runtime — that is not a syntax error.
        # A syntax error surfaces as "syntax error in expression" / "unmatched"
        # / "extra characters after close-brace" from the Tcl parser itself.
        syntax_markers = ("syntax error", "unmatched", "extra characters after")
        lowered = proc.stderr.lower()
        for marker in syntax_markers:
            self.assertNotIn(
                marker,
                lowered,
                f"Tcl syntax error detected: {proc.stderr}",
            )

    def test_proc_report_multicorner_timing_drives_two_corner_branch(self):
        """Exercise the actual 2+-corner code path (the one Finding 1 found
        crashing) by stubbing the real, verified OpenSTA SWIG commands the
        proc calls -- sta::find_scene, sta::total_negative_slack_scene_cmd,
        sta::worst_slack_scene, sta::format_time -- plus report_clock_skew,
        and asserting real files are written with parseable content."""
        tclsh = shutil.which("tclsh")
        if not tclsh:
            self.skipTest("tclsh not available")

        script = os.path.join(
            REPO_ROOT, "flow", "scripts", "report_multicorner_timing.tcl"
        )
        tmpdir = tempfile.mkdtemp()
        try:
            tcl_input = f"""
proc env_var_exists_and_non_empty {{env_var}} {{
  return [expr {{[info exists ::env($env_var)] && $::env($env_var) ne ""}}]
}}

namespace eval sta {{
  array set ::TNS_BY_CORNER {{tt -5.0 ss -20.0}}
  array set ::WS_BY_CORNER {{tt -1.2 ss -3.5}}

  proc find_scene {{name}} {{
    if {{[info exists ::TNS_BY_CORNER($name)]}} {{
      return $name
    }}
    return "NULL"
  }}
  proc total_negative_slack_scene_cmd {{scene min_max}} {{
    return $::TNS_BY_CORNER($scene)
  }}
  proc worst_slack_scene {{scene min_max}} {{
    return $::WS_BY_CORNER($scene)
  }}
  proc format_time {{val digits}} {{
    return [format "%.4f" $val]
  }}
}}

array set ::SKEW_BY_CORNER {{tt -0.100 ss -0.300}}
proc report_clock_skew {{args}} {{
  set n [llength $args]
  set redirect_file ""
  if {{ $n >= 2 && [lindex $args [expr {{$n - 2}}]] eq ">>" }} {{
    set redirect_file [lindex $args [expr {{$n - 1}}]]
    set args [lrange $args 0 [expr {{$n - 3}}]]
  }}
  set corner [lindex $args 1]
  set text "Clock clk\\n  $::SKEW_BY_CORNER($corner) setup skew\\n"
  if {{ $redirect_file ne "" }} {{
    set fid [open $redirect_file a]
    puts -nonewline $fid $text
    close $fid
  }}
}}

set ::env(CORNERS) {{tt ss}}
set ::env(REPORTS_DIR) {{{tmpdir}}}
set ::env(REPORT_CLOCK_SKEW) 1
source "{script}"
set ::env(REPORT_MULTICORNER_TIMING) 1
report_multicorner_timing 4 "cts final"
puts "OK"
"""
            proc = subprocess.run(
                [tclsh], input=tcl_input, capture_output=True, text=True, timeout=10
            )
            self.assertIn("OK", proc.stdout, msg=f"stderr: {proc.stderr}")
            self.assertEqual(proc.stderr.strip(), "", msg=proc.stderr)

            tt_path = os.path.join(tmpdir, "4_cts_final_multicorner_tt.rpt")
            ss_path = os.path.join(tmpdir, "4_cts_final_multicorner_ss.rpt")
            self.assertTrue(os.path.isfile(tt_path))
            self.assertTrue(os.path.isfile(ss_path))

            data = collect_per_corner(tmpdir, "4_cts_final")
            self.assertEqual(set(data.keys()), {"tt", "ss"})
            self.assertAlmostEqual(data["tt"]["tns"], -5.0)
            self.assertAlmostEqual(data["tt"]["worst_slack"], -1.2)
            self.assertAlmostEqual(data["tt"]["wns"], -1.2)
            self.assertAlmostEqual(data["ss"]["tns"], -20.0)
            self.assertAlmostEqual(data["ss"]["worst_slack"], -3.5)
            self.assertAlmostEqual(data["ss"]["wns"], -3.5)
            self.assertAlmostEqual(data["tt"]["clock_skew"], -0.100)
            self.assertAlmostEqual(data["ss"]["clock_skew"], -0.300)
            self.assertEqual(worst_corner(data, "wns"), "ss")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_two_hook_sourcings_in_one_session_do_not_cross_contaminate(self):
        """POST_CTS_TCL and POST_GLOBAL_ROUTE_TCL can be wired to this same
        file with REPORT_MULTICORNER_STAGE/WHEN left unset (both are
        process-global env vars, so a single `export` cannot label the two
        hook points differently). Source the script twice in the same
        interpreter, as those two hooks would, with the underlying timing
        data changed in between, and confirm the first (CTS) invocation's
        report file survives untouched and unmislabeled once the second
        (global-route) invocation runs, instead of being truncated /
        overwritten under the same label."""
        tclsh = shutil.which("tclsh")
        if not tclsh:
            self.skipTest("tclsh not available")

        script = os.path.join(
            REPO_ROOT, "flow", "scripts", "report_multicorner_timing.tcl"
        )
        tmpdir = tempfile.mkdtemp()
        try:
            tcl_input = f"""
proc env_var_exists_and_non_empty {{env_var}} {{
  return [expr {{[info exists ::env($env_var)] && $::env($env_var) ne ""}}]
}}

namespace eval sta {{
  proc find_scene {{name}} {{
    if {{[info exists ::TNS_BY_CORNER($name)]}} {{
      return $name
    }}
    return "NULL"
  }}
  proc total_negative_slack_scene_cmd {{scene min_max}} {{
    return $::TNS_BY_CORNER($scene)
  }}
  proc worst_slack_scene {{scene min_max}} {{
    return $::WS_BY_CORNER($scene)
  }}
  proc format_time {{val digits}} {{
    return [format "%.4f" $val]
  }}
}}

set ::env(CORNERS) {{tt ss}}
set ::env(REPORTS_DIR) {{{tmpdir}}}
set ::env(REPORT_MULTICORNER_TIMING) 1

array set ::TNS_BY_CORNER {{tt -5.0 ss -20.0}}
array set ::WS_BY_CORNER {{tt -1.2 ss -3.5}}
source "{script}"

array set ::TNS_BY_CORNER {{tt -1.0 ss -2.0}}
array set ::WS_BY_CORNER {{tt -0.5 ss -0.6}}
source "{script}"

puts "OK"
"""
            proc = subprocess.run(
                [tclsh], input=tcl_input, capture_output=True, text=True, timeout=10
            )
            self.assertIn("OK", proc.stdout, msg=f"stderr: {proc.stderr}")

            cts_path = os.path.join(tmpdir, "4_cts_final_multicorner_tt.rpt")
            grt_path = os.path.join(tmpdir, "5_global_route_multicorner_tt.rpt")
            self.assertTrue(os.path.isfile(cts_path), os.listdir(tmpdir))
            self.assertTrue(os.path.isfile(grt_path), os.listdir(tmpdir))

            cts_data = collect_per_corner(tmpdir, "4_cts_final")
            grt_data = collect_per_corner(tmpdir, "5_global_route")
            self.assertAlmostEqual(cts_data["tt"]["tns"], -5.0)
            self.assertAlmostEqual(grt_data["tt"]["tns"], -1.0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_proc_report_multicorner_timing_is_noop_for_single_corner(self):
        tclsh = shutil.which("tclsh")
        if not tclsh:
            self.skipTest("tclsh not available")

        script = os.path.join(
            REPO_ROOT, "flow", "scripts", "report_multicorner_timing.tcl"
        )
        tmpdir = tempfile.mkdtemp()
        try:
            tcl_input = f"""
proc env_var_exists_and_non_empty {{env_var}} {{
  return [expr {{[info exists ::env($env_var)] && $::env($env_var) ne ""}}]
}}
set ::env(REPORT_MULTICORNER_TIMING) 1
set ::env(CORNERS) {{tt}}
set ::env(REPORTS_DIR) {{{tmpdir}}}
source "{script}"
report_multicorner_timing 4 "cts final"
puts "OK"
"""
            proc = subprocess.run(
                [tclsh], input=tcl_input, capture_output=True, text=True, timeout=10
            )
            self.assertIn("OK", proc.stdout)
            self.assertEqual(os.listdir(tmpdir), [])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
