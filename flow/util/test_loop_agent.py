#!/usr/bin/env python3
"""Unit tests for loop_agent.py — no Docker, no API, no filesystem side-effects."""

import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

# Make loop_agent importable without triggering argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_agent import (
    CONFIG_HOOK_PATHS,
    ECO_STAGE_ODB,
    ECO_TOLERANCES,
    HOOK_PATHS,
    PARAM_ALLOWLIST,
    STAGE_STALE_FILES,
    _format_eco_result,
    impl_eco_fix,
    impl_set_config_param,
    write_config_params,
)


class TestAllowlist(unittest.TestCase):
    """set_config_param rejects params not in PARAM_ALLOWLIST."""

    def _call(self, param, value):
        pending, log = {}, []
        result = impl_set_config_param(param, value, pending, log)
        return result, pending, log

    def test_rejects_unknown_param(self):
        result, pending, log = self._call("MAX_FANOUT", "16")
        self.assertIn("ERROR", result)
        self.assertIn("not allowlisted", result)
        self.assertEqual(pending, {})
        self.assertEqual(log, [])

    def test_accepts_all_allowlisted_params(self):
        for param in PARAM_ALLOWLIST:
            with self.subTest(param=param):
                value = "enabled" if "TCL" in param else "0.03"
                result, pending, _ = self._call(param, value)
                self.assertNotIn("ERROR", result)
                self.assertIn(param, pending)

    def test_rejects_empty_string_param(self):
        result, _, _ = self._call("", "0.03")
        self.assertIn("ERROR", result)

    def test_rejects_injected_param(self):
        result, _, _ = self._call("SETUP_SLACK_MARGIN; rm -rf /", "0.03")
        self.assertIn("ERROR", result)

    def test_rejects_injected_value_dollar_paren(self):
        result, pending, log = self._call("SETUP_SLACK_MARGIN", "$(shell rm -rf /)")
        self.assertIn("ERROR", result)
        self.assertEqual(pending, {})
        self.assertEqual(log, [])

    def test_rejects_injected_value_dollar_brace(self):
        result, pending, log = self._call("SETUP_SLACK_MARGIN", "${shell rm -rf /}")
        self.assertIn("ERROR", result)
        self.assertEqual(pending, {})
        self.assertEqual(log, [])


class TestHookTranslation(unittest.TestCase):
    """'enabled' sentinel is translated to the Docker /work/scripts/ path."""

    def _call(self, param, value):
        pending, log = {}, []
        impl_set_config_param(param, value, pending, log)
        return pending

    def test_post_cts_enabled_translates(self):
        pending = self._call("POST_CTS_TCL", "enabled")
        self.assertEqual(pending["POST_CTS_TCL"], HOOK_PATHS["POST_CTS_TCL"])
        self.assertTrue(pending["POST_CTS_TCL"].startswith("/work/scripts/"))

    def test_post_grt_enabled_translates(self):
        pending = self._call("POST_GLOBAL_ROUTE_TCL", "enabled")
        self.assertEqual(
            pending["POST_GLOBAL_ROUTE_TCL"], HOOK_PATHS["POST_GLOBAL_ROUTE_TCL"]
        )

    def test_enabled_case_insensitive(self):
        pending = self._call("POST_CTS_TCL", "ENABLED")
        self.assertEqual(pending["POST_CTS_TCL"], HOOK_PATHS["POST_CTS_TCL"])

    def test_numeric_param_not_translated(self):
        pending = self._call("SETUP_SLACK_MARGIN", "0.03")
        self.assertEqual(pending["SETUP_SLACK_MARGIN"], "0.03")

    def test_explicit_path_not_double_translated(self):
        explicit = "/work/scripts/post_cts_timing_repair.tcl"
        pending = self._call("POST_CTS_TCL", explicit)
        self.assertEqual(pending["POST_CTS_TCL"], explicit)


class TestStaleFilePaths(unittest.TestCase):
    """STAGE_STALE_FILES covers the right files for each stage."""

    def _paths(self, stage):
        return [
            p.format(p="nangate45", d="aes", t="base") for p in STAGE_STALE_FILES[stage]
        ]

    def test_cts_stale_files(self):
        paths = self._paths("cts")
        self.assertTrue(any("4_1_cts.odb" in p for p in paths))
        self.assertTrue(any("4_cts.odb" in p for p in paths))

    def test_grt_stale_files(self):
        paths = self._paths("grt")
        self.assertTrue(any("5_1_grt.odb" in p for p in paths))

    def test_place_stale_files_include_global_place(self):
        # Must include 3_3_place_gp.odb — earliest file PLACE_DENSITY_LB_ADDON affects
        paths = self._paths("place")
        self.assertTrue(any("3_3_place_gp.odb" in p for p in paths))
        self.assertTrue(any("3_5_place_dp.odb" in p for p in paths))
        self.assertTrue(any("3_place.odb" in p for p in paths))

    def test_finish_stale_files(self):
        paths = self._paths("finish")
        self.assertTrue(any("5_2_route.odb" in p for p in paths))

    def test_place_stage_subset_of_cts(self):
        # place stale files must NOT include CTS outputs (4_1_cts.odb etc.)
        place_paths = self._paths("place")
        self.assertFalse(any("4_1_cts" in p or "4_cts" in p for p in place_paths))

    def test_all_stages_present(self):
        for stage in ("place", "cts", "grt", "finish"):
            self.assertIn(stage, STAGE_STALE_FILES)
            self.assertTrue(len(STAGE_STALE_FILES[stage]) > 0)


class TestWriteConfigParams(unittest.TestCase):
    """write_config_params updates in-place and appends new params correctly."""

    def _make_config(self, content):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix="config.mk", delete=False)
        tmp.write(textwrap.dedent(content))
        tmp.flush()
        return tmp.name

    def _write(self, config_text, params):
        """Write params into a temp config.mk and return (result_str, final_text)."""
        cfg = self._make_config(config_text)
        # Build minimal directory tree: designs/<platform>/<design>/config.mk
        with tempfile.TemporaryDirectory() as tmpdir:
            platform, design = "nangate45", "aes"
            design_dir = os.path.join(tmpdir, "designs", platform, design)
            os.makedirs(design_dir)
            config_path = os.path.join(design_dir, "config.mk")
            with open(cfg) as f:
                content = f.read()
            with open(config_path, "w") as f:
                f.write(content)
            os.unlink(cfg)
            result = write_config_params(params, platform, design, tmpdir)
            with open(config_path) as f:
                final = f.read()
        return result, final

    def test_updates_existing_param_in_place(self):
        config = """\
            export PLACE_DENSITY_LB_ADDON = 0.20
            export TNS_END_PERCENT        = 100
        """
        _, final = self._write(config, {"PLACE_DENSITY_LB_ADDON": "0.25"})
        self.assertIn("export PLACE_DENSITY_LB_ADDON = 0.25", final)
        self.assertNotIn("0.20", final)

    def test_appends_new_param(self):
        config = "export DESIGN_NICKNAME = aes\n"
        _, final = self._write(config, {"SETUP_SLACK_MARGIN": "0.03"})
        self.assertIn("export SETUP_SLACK_MARGIN = 0.03", final)

    def test_hook_path_translated_for_writeback(self):
        config = "export DESIGN_NICKNAME = aes\n"
        docker_path = HOOK_PATHS["POST_CTS_TCL"]  # /work/scripts/...
        _, final = self._write(config, {"POST_CTS_TCL": docker_path})
        # Must NOT write the /work/scripts path; must write $(SCRIPTS_DIR)/...
        self.assertNotIn("/work/scripts/", final)
        self.assertIn(CONFIG_HOOK_PATHS["POST_CTS_TCL"], final)

    def test_non_hook_path_not_translated(self):
        config = "export DESIGN_NICKNAME = aes\n"
        _, final = self._write(config, {"SETUP_SLACK_MARGIN": "0.03"})
        self.assertIn("export SETUP_SLACK_MARGIN = 0.03", final)

    def test_multiple_params_all_written(self):
        config = "export PLACE_DENSITY_LB_ADDON = 0.20\n"
        params = {
            "PLACE_DENSITY_LB_ADDON": "0.25",
            "SETUP_SLACK_MARGIN": "0.03",
            "TNS_END_PERCENT": "100",
        }
        _, final = self._write(config, params)
        self.assertIn("export PLACE_DENSITY_LB_ADDON = 0.25", final)
        self.assertIn("export SETUP_SLACK_MARGIN = 0.03", final)
        self.assertIn("export TNS_END_PERCENT = 100", final)

    def test_existing_line_not_duplicated(self):
        config = "export SETUP_SLACK_MARGIN = 0.00\n"
        _, final = self._write(config, {"SETUP_SLACK_MARGIN": "0.03"})
        count = final.count("export SETUP_SLACK_MARGIN")
        self.assertEqual(count, 1)

    def test_returns_error_for_missing_config(self):
        result = write_config_params(
            {"SETUP_SLACK_MARGIN": "0.03"},
            "nangate45",
            "missing_design",
            "/nonexistent",
        )
        self.assertIn("ERROR", result)

    def test_loop_agent_comment_added_with_new_params(self):
        config = "export DESIGN_NICKNAME = aes\n"
        _, final = self._write(config, {"SETUP_SLACK_MARGIN": "0.03"})
        self.assertIn("loop_agent.py", final)

    def test_loop_agent_comment_absent_when_only_updating(self):
        config = "export SETUP_SLACK_MARGIN = 0.00\n"
        _, final = self._write(config, {"SETUP_SLACK_MARGIN": "0.03"})
        # No new params → comment block should NOT be added
        self.assertNotIn("loop_agent.py", final)

    def test_refuses_to_write_dollar_paren_injection(self):
        config = "export SETUP_SLACK_MARGIN = 0.00\n"
        result, final = self._write(config, {"SETUP_SLACK_MARGIN": "$(shell rm -rf /)"})
        self.assertIn("ERROR", result)
        self.assertNotIn("$(shell", final)
        self.assertIn("export SETUP_SLACK_MARGIN = 0.00", final)

    def test_refuses_to_write_dollar_brace_injection(self):
        config = "export SETUP_SLACK_MARGIN = 0.00\n"
        result, final = self._write(config, {"SETUP_SLACK_MARGIN": "${shell rm -rf /}"})
        self.assertIn("ERROR", result)
        self.assertNotIn("${shell", final)
        self.assertIn("export SETUP_SLACK_MARGIN = 0.00", final)


class TestEcoFixValidation(unittest.TestCase):
    """impl_eco_fix rejects bad input before ever touching a subprocess."""

    def _call(self, flow_dir, fix_type="resize_up", target="_412_", stage="cts", cell=""):
        change_log = []
        eco_counter = itertools.count(1)
        result = impl_eco_fix(
            fix_type,
            target,
            stage,
            cell,
            "nangate45",
            "aes",
            "base",
            flow_dir,
            change_log,
            eco_counter,
        )
        return result, change_log

    def test_rejects_unknown_fix_type(self):
        with mock.patch("loop_agent.subprocess.run") as run:
            result, _ = self._call("/nonexistent", fix_type="frobnicate")
            run.assert_not_called()
        self.assertIn("ERROR", result)
        self.assertIn("fix_type", result)

    def test_rejects_target_with_semicolon(self):
        with mock.patch("loop_agent.subprocess.run") as run:
            result, _ = self._call("/nonexistent", target="_412_; rm -rf /")
            run.assert_not_called()
        self.assertIn("ERROR", result)

    def test_rejects_target_with_dollar_paren(self):
        with mock.patch("loop_agent.subprocess.run") as run:
            result, _ = self._call("/nonexistent", target="$(shell rm -rf /)")
            run.assert_not_called()
        self.assertIn("ERROR", result)

    def test_rejects_target_with_newline(self):
        with mock.patch("loop_agent.subprocess.run") as run:
            result, _ = self._call("/nonexistent", target="_412_\nsource evil.tcl")
            run.assert_not_called()
        self.assertIn("ERROR", result)

    def test_rejects_stage_not_in_eco_stage_odb(self):
        with mock.patch("loop_agent.subprocess.run") as run:
            result, _ = self._call("/nonexistent", stage="finish")
            run.assert_not_called()
        self.assertIn("ERROR", result)
        self.assertIn("finish", result)

    def test_missing_odb_returns_error_without_subprocess(self):
        with tempfile.TemporaryDirectory() as flow_dir:
            with mock.patch("loop_agent.subprocess.run") as run:
                result, _ = self._call(flow_dir)
                run.assert_not_called()
            self.assertIn("ERROR", result)
            self.assertIn("not found", result)

    def test_rejects_insert_buffer_fix_type(self):
        # insert_buffer was removed from ECO_FIX_TYPES: it segfaults
        # OpenROAD on every tested input (see PR_EXTENSION_DEV_LOG.md).
        with mock.patch("loop_agent.subprocess.run") as run:
            result, _ = self._call("/nonexistent", fix_type="insert_buffer")
            run.assert_not_called()
        self.assertIn("ERROR", result)
        self.assertIn("fix_type", result)

    def test_rejects_cell_with_semicolon(self):
        with mock.patch("loop_agent.subprocess.run") as run:
            result, _ = self._call("/nonexistent", cell="AND2_X1; rm -rf /")
            run.assert_not_called()
        self.assertIn("ERROR", result)
        self.assertIn("cell", result)

    def test_rejects_fix_hold_target_without_slash(self):
        with mock.patch("loop_agent.subprocess.run") as run:
            result, _ = self._call("/nonexistent", fix_type="fix_hold", target="_412_")
            run.assert_not_called()
        self.assertIn("ERROR", result)
        self.assertIn("pin name", result)

    def test_rejects_target_with_odd_trailing_backslashes(self):
        with mock.patch("loop_agent.subprocess.run") as run:
            result, _ = self._call("/nonexistent", target="_412_\\")
            run.assert_not_called()
        self.assertIn("ERROR", result)
        self.assertIn("backslash", result)

    def test_accepts_target_with_even_trailing_backslashes(self):
        # Not a Tcl-injection boundary issue by itself — must fail later
        # (missing odb), not at the backslash-parity check.
        with tempfile.TemporaryDirectory() as flow_dir:
            with mock.patch("loop_agent.subprocess.run") as run:
                result, _ = self._call(flow_dir, target="_412_\\\\")
                run.assert_not_called()
            self.assertIn("ERROR", result)
            self.assertIn("not found", result)

    def test_rejects_cell_with_odd_trailing_backslashes(self):
        with mock.patch("loop_agent.subprocess.run") as run:
            result, _ = self._call("/nonexistent", cell="AND2_X1\\")
            run.assert_not_called()
        self.assertIn("ERROR", result)
        self.assertIn("backslash", result)

    def test_json_parse_failure_returns_error_with_output_tail(self):
        with tempfile.TemporaryDirectory() as flow_dir:
            platform, design, tag = "nangate45", "aes", "base"
            odb_rel = ECO_STAGE_ODB["cts"].format(p=platform, d=design, t=tag)
            odb_path = os.path.join(flow_dir, odb_rel)
            os.makedirs(os.path.dirname(odb_path))
            with open(odb_path, "w") as f:
                f.write("")

            with mock.patch(
                "loop_agent.subprocess.run",
                return_value=mock.Mock(stdout="some make output", stderr=""),
            ):
                result, change_log = self._call(flow_dir)

            self.assertIn("ERROR", result)
            self.assertIn("no result", result)
            self.assertIn("some make output", result)
            self.assertEqual(change_log, [])


class TestEcoFixGeneratedTcl(unittest.TestCase):
    """impl_eco_fix writes a generated Tcl script invoking trepair::eco_run."""

    def test_generated_tcl_contains_target_and_eco_run(self):
        with tempfile.TemporaryDirectory() as flow_dir:
            platform, design, tag = "nangate45", "aes", "base"
            odb_rel = ECO_STAGE_ODB["cts"].format(p=platform, d=design, t=tag)
            odb_path = os.path.join(flow_dir, odb_rel)
            os.makedirs(os.path.dirname(odb_path))
            with open(odb_path, "w") as f:
                f.write("")

            fake_json = {
                "id": "eco1",
                "status": "applied",
                "msg": "",
                "fix": {"kind": "resize", "inst": "_412_", "from": "AND2_X1", "to": "AND2_X2"},
                "before": {
                    "wns": -0.031,
                    "tns": -1.204,
                    "worst_hold_slack": 0.021,
                    "setup_viol_count": 37,
                    "hold_viol_count": 0,
                },
                "after": {
                    "wns": -0.019,
                    "tns": -0.864,
                    "worst_hold_slack": 0.020,
                    "setup_viol_count": 30,
                    "hold_viol_count": 0,
                },
                "delta": {"wns": 0.012, "tns": 0.340, "worst_hold_slack": -0.001},
                "verdict": {
                    "accepted": True,
                    "target_metric": "wns",
                    "reason": "improved wns",
                },
                "targets": [{"pin": "_412_/ZN", "inst": "_412_", "cell": "AND2_X1"}],
                "odb_written": True,
            }

            def fake_run(cmd, cwd, capture_output, text, timeout):
                json_path = os.path.join(
                    flow_dir, "objects", platform, design, tag, "eco", "eco1.json"
                )
                with open(json_path, "w") as f:
                    json.dump(fake_json, f)
                return mock.Mock(stdout="", stderr="")

            change_log = []
            eco_counter = itertools.count(1)
            with mock.patch("loop_agent.subprocess.run", side_effect=fake_run):
                result = impl_eco_fix(
                    "resize_up",
                    "_412_",
                    "cts",
                    "",
                    platform,
                    design,
                    tag,
                    flow_dir,
                    change_log,
                    eco_counter,
                )

            tcl_path = os.path.join(
                flow_dir, "objects", platform, design, tag, "eco", "eco1.tcl"
            )
            with open(tcl_path) as f:
                tcl_text = f.read()

            self.assertIn("trepair::eco_run", tcl_text)
            self.assertIn("_412_", tcl_text)
            self.assertIn("ACCEPTED", result)
            self.assertEqual(len(change_log), 1)
            self.assertEqual(change_log[0]["action"], "eco_fix")


class TestEcoFixJsonTransport(unittest.TestCase):
    """Round-trips a raw JSON *string* through the real json.load() call in
    impl_eco_fix, simulating actual Tcl output post eco_json_str's
    backslash-escaping fix. A hand-written Python dict fixture (as used
    elsewhere in this file) cannot catch a Tcl-side escaping bug by
    construction — this test exercises the transport itself.
    """

    def test_backslash_in_pin_name_round_trips_through_real_json_parse(self):
        with tempfile.TemporaryDirectory() as flow_dir:
            platform, design, tag = "nangate45", "gcd", "base"
            odb_rel = ECO_STAGE_ODB["cts"].format(p=platform, d=design, t=tag)
            odb_path = os.path.join(flow_dir, odb_rel)
            os.makedirs(os.path.dirname(odb_path))
            with open(odb_path, "w") as f:
                f.write("")

            # Mirrors what eco_json_str now emits for a real escaped ODB/
            # yosys name (e.g. ctrl.state.out\[0\]$_DFF_P_): backslash
            # escaped to \\ before the JSON string is written.
            raw_json = (
                r'{"id":"eco1","status":"applied","msg":"",'
                r'"fix":{"kind":"fix_hold","inst":"",'
                r'"from":"ctrl.state.out\\[0\\]$_DFF_P_\/D",'
                r'"to":"ctrl.state.out\\[0\\]$_DFF_P_\/D"},'
                r'"before":{"wns":-0.031,"tns":-1.204,"worst_hold_slack":-0.005,'
                r'"setup_viol_count":37,"hold_viol_count":2},'
                r'"after":{"wns":-0.032,"tns":-1.204,"worst_hold_slack":0.001,'
                r'"setup_viol_count":37,"hold_viol_count":0},'
                r'"delta":{"wns":-0.001,"tns":0.0,"worst_hold_slack":0.006},'
                r'"verdict":{"accepted":true,"target_metric":"worst_hold_slack",'
                r'"reason":"improved worst_hold_slack by 0.006 and no regression"},'
                r'"targets":[],"odb_written":true}'
            )

            def fake_run(cmd, cwd, capture_output, text, timeout):
                json_path = os.path.join(
                    flow_dir, "objects", platform, design, tag, "eco", "eco1.json"
                )
                with open(json_path, "w") as f:
                    f.write(raw_json)
                return mock.Mock(stdout="", stderr="")

            change_log = []
            eco_counter = itertools.count(1)
            with mock.patch("loop_agent.subprocess.run", side_effect=fake_run):
                result = impl_eco_fix(
                    "fix_hold",
                    r"ctrl.state.out\[0\]$_DFF_P_/D",
                    "cts",
                    "",
                    platform,
                    design,
                    tag,
                    flow_dir,
                    change_log,
                    eco_counter,
                )

            self.assertIn("ACCEPTED", result)
            self.assertNotIn("ERROR", result)
            self.assertEqual(len(change_log), 1)
            parsed = change_log[0]["result"]
            self.assertIn("\\[0\\]", parsed["fix"]["from"])


class TestEcoResultFormatter(unittest.TestCase):
    """_format_eco_result renders accept/reject verdicts for all four fix types."""

    def _result(self, accepted, target_metric, reason="", odb_written=None):
        if odb_written is None:
            odb_written = accepted
        return {
            "status": "applied" if accepted else "rejected",
            "msg": "",
            "fix": {"kind": "resize", "inst": "_412_", "from": "AND2_X1", "to": "AND2_X2"},
            "before": {
                "wns": -0.031,
                "tns": -1.204,
                "worst_hold_slack": 0.021,
                "setup_viol_count": 37,
                "hold_viol_count": 0,
            },
            "after": {
                "wns": -0.019,
                "tns": -0.864,
                "worst_hold_slack": 0.020,
                "setup_viol_count": 30,
                "hold_viol_count": 0,
            },
            "delta": {"wns": 0.012, "tns": 0.340, "worst_hold_slack": -0.001},
            "verdict": {
                "accepted": accepted,
                "target_metric": target_metric,
                "reason": reason,
            },
            "targets": [{"pin": "_412_/ZN", "inst": "_412_", "cell": "AND2_X1"}],
            "odb_written": odb_written,
        }

    def test_resize_up_accepted(self):
        result = self._result(True, "wns", "improved wns")
        text = _format_eco_result("resize_up", "_412_", "cts", result)
        self.assertIn("ACCEPTED", text)
        self.assertIn("odb_written: True", text)

    def test_resize_down_rejected(self):
        result = self._result(False, "wns", "tns regressed")
        text = _format_eco_result("resize_down", "_412_", "cts", result)
        self.assertIn("REJECTED", text)
        self.assertIn("odb_written: False", text)

    def test_insert_buffer_accepted(self):
        result = self._result(True, "wns", "improved wns")
        text = _format_eco_result("insert_buffer", "net1", "grt", result)
        self.assertIn("ACCEPTED", text)

    def test_fix_hold_rejected(self):
        result = self._result(False, "worst_hold_slack", "no improvement in worst_hold_slack")
        text = _format_eco_result("fix_hold", "_412_/D", "cts", result)
        self.assertIn("REJECTED", text)
        self.assertIn("worst_hold_slack", text)

    def test_tolerances_exposed(self):
        self.assertIn("wns", ECO_TOLERANCES)
        self.assertIn("tns", ECO_TOLERANCES)
        self.assertIn("hold", ECO_TOLERANCES)
        self.assertIn("wns_hold", ECO_TOLERANCES)
        # wns_hold must be looser than the generic wns tolerance, or the
        # whole point of the fix_hold-specific tolerance is defeated.
        self.assertGreater(ECO_TOLERANCES["wns_hold"], ECO_TOLERANCES["wns"])


ECO_REPAIR_TCL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "eco_repair.tcl"
)


@unittest.skipUnless(shutil.which("tclsh"), "tclsh not available")
class TestEcoRepairTclVerdict(unittest.TestCase):
    """Exercises eco_repair.tcl's pure Tcl logic (terms_compatible,
    eco_verdict) directly via tclsh, with no ODB/OpenROAD dependency —
    these procs only manipulate Tcl dicts/lists. Covers the swapMaster
    compatibility pre-check, the tns=="NA" rejection path, and the
    fix_hold-specific setup_viol_count tolerance.
    """

    def _run(self, tcl_body):
        script = f'source "{ECO_REPAIR_TCL}"\n{tcl_body}'
        with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as f:
            f.write(script)
            path = f.name
        try:
            result = subprocess.run(
                ["tclsh", path], capture_output=True, text=True, timeout=30
            )
        finally:
            os.remove(path)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        return result.stdout

    def _metrics(self, wns, tns, hold, setup_viol, hold_viol):
        return (
            f"[dict create wns {wns} tns {tns} worst_hold_slack {hold} "
            f"setup_viol_count {setup_viol} hold_viol_count {hold_viol}]"
        )

    # -- swapMaster compatibility pre-check (terms_compatible) --

    def test_terms_compatible_matching_regardless_of_order(self):
        out = self._run('puts [trepair::terms_compatible {A B ZN} {ZN B A}]')
        self.assertEqual(out.strip(), "1")

    def test_terms_compatible_rejects_mismatched_terminals(self):
        # e.g. a real NAND2_X4 (A B ZN) vs an explicit `cell` of INV_X4 (A ZN):
        # this is the exact incompatible-swap shape from the live-reproduced
        # bug report (resize_down NAND2-like inst to an INV-like cell).
        out = self._run('puts [trepair::terms_compatible {A B ZN} {A ZN}]')
        self.assertEqual(out.strip(), "0")

    # -- tns == "NA" rejection --

    def test_verdict_rejects_when_tns_is_na(self):
        before = self._metrics(-0.031, "NA", 0.021, 37, 0)
        after = self._metrics(-0.019, -0.864, 0.020, 30, 0)
        out = self._run(
            f"set r [trepair::eco_verdict resize_down {before} {after} "
            "0.001 0.05 0.001 0.01]\n"
            "puts [dict get $r accepted]\n"
            "puts [dict get $r reason]\n"
        )
        lines = out.strip().splitlines()
        self.assertEqual(lines[0], "false")
        self.assertIn("insufficient data", lines[1])
        self.assertIn("tns", lines[1])

    def test_verdict_accepts_resize_down_when_all_metrics_measured(self):
        before = self._metrics(-0.031, -1.204, 0.021, 37, 0)
        after = self._metrics(-0.031, -1.204, 0.021, 30, 0)
        out = self._run(
            f"set r [trepair::eco_verdict resize_down {before} {after} "
            "0.001 0.05 0.001 0.01]\n"
            "puts [dict get $r accepted]\n"
        )
        self.assertEqual(out.strip(), "true")

    # -- fix_hold-specific setup_viol_count tolerance --

    def test_fix_hold_tolerates_small_setup_viol_increase_from_collateral_wns(self):
        # Mirrors the live-reproduced case: a hold fix's collateral WNS cost
        # of -0.00018ns (well within tol_wns_hold=0.01) flips one endpoint's
        # setup_viol_count from 23 to 24. tol_wns_hold alone would accept
        # this; a zero-tolerance setup_viol_count check would wrongly reject
        # it. The fix-type-aware tolerance must let it through.
        before = self._metrics(-0.001, -0.5, -0.005, 23, 2)
        after = self._metrics(-0.00118, -0.5, 0.001, 24, 0)
        out = self._run(
            f"set r [trepair::eco_verdict fix_hold {before} {after} "
            "0.001 0.05 0.001 0.01]\n"
            "puts [dict get $r accepted]\n"
        )
        self.assertEqual(out.strip(), "true")

    def test_fix_hold_still_rejects_large_setup_viol_regression(self):
        before = self._metrics(-0.001, -0.5, -0.005, 23, 2)
        after = self._metrics(-0.00118, -0.5, 0.001, 30, 0)
        out = self._run(
            f"set r [trepair::eco_verdict fix_hold {before} {after} "
            "0.001 0.05 0.001 0.01]\n"
            "puts [dict get $r accepted]\n"
            "puts [dict get $r reason]\n"
        )
        lines = out.strip().splitlines()
        self.assertEqual(lines[0], "false")
        self.assertIn("setup_viol_count", lines[1])

    def test_resize_down_zero_tolerance_still_rejects_setup_viol_regression(self):
        before = self._metrics(-0.001, -0.5, 0.005, 23, 0)
        after = self._metrics(-0.001, -0.5, 0.005, 24, 0)
        out = self._run(
            f"set r [trepair::eco_verdict resize_down {before} {after} "
            "0.001 0.05 0.001 0.01]\n"
            "puts [dict get $r accepted]\n"
        )
        self.assertEqual(out.strip(), "false")


if __name__ == "__main__":
    unittest.main(verbosity=2)
