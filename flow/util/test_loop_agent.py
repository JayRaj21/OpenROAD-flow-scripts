#!/usr/bin/env python3
"""Unit tests for loop_agent.py — no Docker, no API, no filesystem side-effects."""

import os
import sys
import tempfile
import textwrap
import unittest

# Make loop_agent importable without triggering argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_agent import (
    CONFIG_HOOK_PATHS,
    HOOK_PATHS,
    PARAM_ALLOWLIST,
    STAGE_STALE_FILES,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
