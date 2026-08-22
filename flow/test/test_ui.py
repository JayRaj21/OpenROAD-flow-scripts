"""
Tests for flow/ui/layout_parser.py and the Flask endpoints in flow/ui/app.py.

Run from the repo root:
    python3 -m pytest flow/test/test_ui.py -v
or:
    python3 flow/test/test_ui.py
"""

import os
import sys
import tempfile
import textwrap
import unittest

# Add flow/ui/ so layout_parser and app can be imported without a package prefix
_UI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui"))
sys.path.insert(0, _UI_DIR)

import layout_parser

# ---------------------------------------------------------------------------
# layout_parser — get_design_nickname
# ---------------------------------------------------------------------------


class TestGetDesignNickname(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _make_config(self, platform, design, content):
        d = os.path.join(self.tmp.name, "designs", platform, design)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.mk"), "w") as f:
            f.write(content)

    def test_explicit_nickname_question_equals(self):
        self._make_config("nangate45", "black_parrot", "export DESIGN_NICKNAME ?= bp\n")
        self.assertEqual(
            layout_parser.get_design_nickname(
                "nangate45", "black_parrot", self.tmp.name
            ),
            "bp",
        )

    def test_explicit_nickname_plain_equals(self):
        self._make_config("nangate45", "gcd", "export DESIGN_NICKNAME = gcd\n")
        self.assertEqual(
            layout_parser.get_design_nickname("nangate45", "gcd", self.tmp.name), "gcd"
        )

    def test_fallback_when_no_config(self):
        self.assertEqual(
            layout_parser.get_design_nickname("nangate45", "adder", self.tmp.name),
            "adder",
        )

    def test_fallback_when_no_nickname_key(self):
        self._make_config("nangate45", "gcd", "export DESIGN_CONFIG = something\n")
        self.assertEqual(
            layout_parser.get_design_nickname("nangate45", "gcd", self.tmp.name), "gcd"
        )


# ---------------------------------------------------------------------------
# layout_parser — parse_timing_reports
# ---------------------------------------------------------------------------


class TestParseTimingReports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _make_design(self, platform, design, nickname=None):
        if nickname is None:
            nickname = design
        d = os.path.join(self.tmp.name, "designs", platform, design)
        os.makedirs(d, exist_ok=True)
        if nickname != design:
            with open(os.path.join(d, "config.mk"), "w") as f:
                f.write(f"export DESIGN_NICKNAME ?= {nickname}\n")
        return nickname

    def _make_reports_dir(self, platform, nickname):
        rdir = os.path.join(self.tmp.name, "reports", platform, nickname, "base")
        os.makedirs(rdir, exist_ok=True)
        return rdir

    def test_no_reports_dir_returns_empty(self):
        result = layout_parser.parse_timing_reports("nangate45", "gcd", self.tmp.name)
        self.assertEqual(result["stages"], {})
        self.assertIsNone(result["summary"])
        self.assertEqual(result["reports_available"], [])

    def test_parses_wns_tns_fmax(self):
        self._make_design("nangate45", "gcd")
        rdir = self._make_reports_dir("nangate45", "gcd")
        rpt = textwrap.dedent("""\
            tns   path_type  -1.23
            wns   path_type  -0.45
            worst slack setup -0.45
            period_min = 1.000 fmax = 1000.000
        """)
        with open(os.path.join(rdir, "6_finish.rpt"), "w") as f:
            f.write(rpt)

        result = layout_parser.parse_timing_reports("nangate45", "gcd", self.tmp.name)
        self.assertIn("Final", result["stages"])
        m = result["stages"]["Final"]
        self.assertAlmostEqual(m["tns"], -1.23)
        self.assertAlmostEqual(m["wns"], -0.45)
        self.assertAlmostEqual(m["fmax"], 1000.0)
        self.assertAlmostEqual(m["period_min"], 1.0)
        self.assertIsNotNone(result["summary"])

    def test_multiple_stages(self):
        self._make_design("nangate45", "gcd")
        rdir = self._make_reports_dir("nangate45", "gcd")
        for fname in ("3_detailed_place.rpt", "4_cts_final.rpt", "6_finish.rpt"):
            with open(os.path.join(rdir, fname), "w") as f:
                f.write("wns   path_type  0.00\ntns   path_type  0.00\n")

        result = layout_parser.parse_timing_reports("nangate45", "gcd", self.tmp.name)
        self.assertEqual(len(result["stages"]), 3)
        self.assertIn("Detailed Place", result["stages"])
        self.assertIn("CTS", result["stages"])
        self.assertIn("Final", result["stages"])

    def test_nickname_used_for_path(self):
        self._make_design("nangate45", "black_parrot", nickname="bp")
        rdir = self._make_reports_dir("nangate45", "bp")
        with open(os.path.join(rdir, "6_finish.rpt"), "w") as f:
            f.write("wns   path_type  -0.1\ntns   path_type  -0.5\n")

        result = layout_parser.parse_timing_reports(
            "nangate45", "black_parrot", self.tmp.name
        )
        self.assertIn("Final", result["stages"])


# ---------------------------------------------------------------------------
# layout_parser — parse_lef_macros
# ---------------------------------------------------------------------------


class TestParseLefMacros(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_lef(self, content, name="test.lef"):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_empty_list_returns_empty(self):
        self.assertEqual(layout_parser.parse_lef_macros([]), {})

    def test_nonexistent_file_skipped(self):
        result = layout_parser.parse_lef_macros(["/does/not/exist.lef"])
        self.assertEqual(result, {})

    def test_parses_block_macro(self):
        path = self._write_lef(textwrap.dedent("""\
            MACRO SRAM_256x32
              CLASS BLOCK ;
              SIZE 10.5 BY 20.3 ;
            END SRAM_256x32
        """))
        macros = layout_parser.parse_lef_macros([path])
        self.assertIn("SRAM_256x32", macros)
        self.assertTrue(macros["SRAM_256x32"]["is_macro"])
        self.assertAlmostEqual(macros["SRAM_256x32"]["w"], 10.5)
        self.assertAlmostEqual(macros["SRAM_256x32"]["h"], 20.3)

    def test_core_cell_not_marked_as_macro(self):
        path = self._write_lef(textwrap.dedent("""\
            MACRO INV_X1
              CLASS CORE ;
              SIZE 0.38 BY 1.4 ;
            END INV_X1
        """))
        macros = layout_parser.parse_lef_macros([path])
        self.assertIn("INV_X1", macros)
        self.assertFalse(macros["INV_X1"]["is_macro"])
        self.assertAlmostEqual(macros["INV_X1"]["w"], 0.38)

    def test_multiple_macros_in_one_file(self):
        path = self._write_lef(textwrap.dedent("""\
            MACRO CELL_A
              CLASS CORE ;
              SIZE 0.5 BY 1.0 ;
            END CELL_A
            MACRO CELL_B
              CLASS BLOCK ;
              SIZE 5.0 BY 10.0 ;
            END CELL_B
        """))
        macros = layout_parser.parse_lef_macros([path])
        self.assertEqual(len(macros), 2)
        self.assertFalse(macros["CELL_A"]["is_macro"])
        self.assertTrue(macros["CELL_B"]["is_macro"])


# ---------------------------------------------------------------------------
# layout_parser — parse_def
# ---------------------------------------------------------------------------


class TestParseDef(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        layout_parser._cache.clear()

    def tearDown(self):
        self.tmp.cleanup()
        layout_parser._cache.clear()

    def _write_def(self, content, name="test.def"):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_returns_none_for_missing_file(self):
        self.assertIsNone(layout_parser.parse_def("/nonexistent/path.def", {}))

    def test_diearea_and_units(self):
        path = self._write_def(textwrap.dedent("""\
            UNITS DISTANCE MICRONS 2000 ;
            DIEAREA ( 0 0 ) ( 200000 160000 ) ;
        """))
        result = layout_parser.parse_def(path, {})
        self.assertIsNotNone(result)
        self.assertEqual(result["dbUnitsPerMicron"], 2000)
        self.assertEqual(result["dieArea"], [0.0, 0.0, 200000.0, 160000.0])

    def test_component_placement_parsed(self):
        path = self._write_def(textwrap.dedent("""\
            UNITS DISTANCE MICRONS 1000 ;
            DIEAREA ( 0 0 ) ( 100000 80000 ) ;
            COMPONENTS 1 ;
            - u1 INV_X1 + PLACED ( 1000 2000 ) N ;
            END COMPONENTS
        """))
        macro_sizes = {"INV_X1": {"w": 0.38, "h": 1.4, "is_macro": False}}
        result = layout_parser.parse_def(path, macro_sizes)
        self.assertEqual(len(result["components"]), 1)
        comp = result["components"][0]
        self.assertEqual(comp["name"], "u1")
        self.assertEqual(comp["cell"], "INV_X1")
        self.assertAlmostEqual(comp["x"], 1000.0)
        self.assertAlmostEqual(comp["y"], 2000.0)
        self.assertFalse(comp["is_macro"])

    def test_fixed_component_parsed(self):
        path = self._write_def(textwrap.dedent("""\
            UNITS DISTANCE MICRONS 1000 ;
            DIEAREA ( 0 0 ) ( 100000 80000 ) ;
            COMPONENTS 1 ;
            - macro0 SRAM + FIXED ( 5000 5000 ) N ;
            END COMPONENTS
        """))
        macro_sizes = {"SRAM": {"w": 50.0, "h": 30.0, "is_macro": True}}
        result = layout_parser.parse_def(path, macro_sizes)
        self.assertEqual(len(result["components"]), 1)
        self.assertTrue(result["components"][0]["is_macro"])

    def test_pin_parsed(self):
        path = self._write_def(textwrap.dedent("""\
            UNITS DISTANCE MICRONS 1000 ;
            DIEAREA ( 0 0 ) ( 100000 80000 ) ;
            PINS 1 ;
            - clk + NET clk + PLACED ( 500 40000 ) N ;
            END PINS
        """))
        result = layout_parser.parse_def(path, {})
        self.assertEqual(len(result["pins"]), 1)
        self.assertEqual(result["pins"][0]["name"], "clk")
        self.assertAlmostEqual(result["pins"][0]["x"], 500.0)

    def test_routes_truncated_flag(self):
        nets_lines = ["NETS 1 ;", "- sig0"]
        for i in range(20):
            nets_lines.append(
                f"  + ROUTED metal1 100 ( {i*1000} 0 ) ( {i*1000+1000} 0 )"
            )
        nets_lines += [";", "END NETS"]
        path = self._write_def(
            "\n".join(
                [
                    "UNITS DISTANCE MICRONS 1000 ;",
                    "DIEAREA ( 0 0 ) ( 1000000 1000000 ) ;",
                ]
                + nets_lines
            )
        )
        result = layout_parser.parse_def(path, {}, max_routes=5)
        self.assertTrue(result["routesTruncated"])
        self.assertLessEqual(len(result["routes"]), 5)

    def test_result_is_cached(self):
        path = self._write_def(
            "UNITS DISTANCE MICRONS 1000 ;\nDIEAREA ( 0 0 ) ( 1000 1000 ) ;\n"
        )
        r1 = layout_parser.parse_def(path, {})
        r2 = layout_parser.parse_def(path, {})
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# layout_parser — get_available_stages
# ---------------------------------------------------------------------------


class TestGetAvailableStages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _results_dir(self, platform="nangate45", nickname="gcd"):
        d = os.path.join(self.tmp.name, "results", platform, nickname, "base")
        os.makedirs(d, exist_ok=True)
        return d

    def test_returns_empty_when_no_results_dir(self):
        stages = layout_parser.get_available_stages("nangate45", "gcd", self.tmp.name)
        self.assertEqual(stages, [])

    def test_finds_odb_stages(self):
        rdir = self._results_dir()
        open(os.path.join(rdir, "2_floorplan.odb"), "w").close()
        open(os.path.join(rdir, "3_5_place_dp.odb"), "w").close()
        stages = layout_parser.get_available_stages("nangate45", "gcd", self.tmp.name)
        ids = [s["id"] for s in stages]
        self.assertIn("2_floorplan", ids)
        self.assertIn("3_5_place_dp", ids)

    def test_stage_ordering(self):
        rdir = self._results_dir()
        # Create out-of-order
        open(os.path.join(rdir, "3_5_place_dp.odb"), "w").close()
        open(os.path.join(rdir, "2_floorplan.odb"), "w").close()
        stages = layout_parser.get_available_stages("nangate45", "gcd", self.tmp.name)
        ids = [s["id"] for s in stages]
        self.assertLess(ids.index("2_floorplan"), ids.index("3_5_place_dp"))

    def test_odb_stage_has_odb_path(self):
        rdir = self._results_dir()
        open(os.path.join(rdir, "4_cts.odb"), "w").close()
        stages = layout_parser.get_available_stages("nangate45", "gcd", self.tmp.name)
        s = next((x for x in stages if x["id"] == "4_cts"), None)
        self.assertIsNotNone(s)
        self.assertIsNotNone(s["odb_path"])
        self.assertIn("4_cts.odb", s["odb_path"])

    def test_existing_def_preferred_over_odb(self):
        rdir = self._results_dir()
        odb = os.path.join(rdir, "2_floorplan.odb")
        def_ = os.path.join(rdir, "2_floorplan.def")
        open(odb, "w").close()
        open(def_, "w").close()
        stages = layout_parser.get_available_stages("nangate45", "gcd", self.tmp.name)
        s = next(x for x in stages if x["id"] == "2_floorplan")
        # When DEF exists, odb_path should be None (no conversion needed)
        self.assertIsNone(s["odb_path"])

    def test_unknown_odb_name_excluded(self):
        rdir = self._results_dir()
        open(os.path.join(rdir, "99_unknown_stage.odb"), "w").close()
        stages = layout_parser.get_available_stages("nangate45", "gcd", self.tmp.name)
        ids = [s["id"] for s in stages]
        self.assertNotIn("99_unknown_stage", ids)


# ---------------------------------------------------------------------------
# Flask endpoint tests
# ---------------------------------------------------------------------------


class TestFlaskEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as flask_app

        flask_app.app.config["TESTING"] = True
        cls.client = flask_app.app.test_client()

    def setUp(self):
        layout_parser._cache.clear()

    def test_index_returns_200(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)

    def test_list_designs_returns_list(self):
        resp = self.client.get("/api/designs")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsInstance(data, list)
        if data:
            self.assertIn("platform", data[0])
            self.assertIn("design", data[0])

    def test_congestion_image_missing_returns_404(self):
        resp = self.client.get("/api/congestion/image/nangate45/__no_such_design__")
        self.assertEqual(resp.status_code, 404)

    def test_floorplan_tcl_missing_returns_404(self):
        resp = self.client.get("/api/floorplan/tcl/nangate45/__no_such_design__")
        self.assertEqual(resp.status_code, 404)

    def test_layout_stages_no_results_returns_empty_list(self):
        resp = self.client.get("/api/layout/stages/nangate45/__no_such_design__")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), [])

    def test_timing_no_reports_returns_valid_structure(self):
        resp = self.client.get("/api/timing/nangate45/__no_such_design__")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("stages", data)
        self.assertIn("summary", data)
        self.assertIn("reports_available", data)
        self.assertIsInstance(data["stages"], dict)
        self.assertIsInstance(data["reports_available"], list)

    def test_layout_unknown_stage_returns_404(self):
        resp = self.client.get("/api/layout/nangate45/__no_such_design__/99_fake")
        self.assertEqual(resp.status_code, 404)

    def test_list_designs_gcd_present(self):
        resp = self.client.get("/api/designs")
        data = resp.get_json()
        found = any(d["platform"] == "nangate45" and d["design"] == "gcd" for d in data)
        self.assertTrue(found, "nangate45/gcd should always be in the design list")


if __name__ == "__main__":
    unittest.main()
