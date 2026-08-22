"""
DEF/LEF parser for layout visualization.

Parses:
  - DIEAREA
  - COMPONENTS (cell placements)
  - SPECIALNETS (power/ground routing)
  - NETS (signal routing, up to max_routes segments)
  - PINS
"""

import os
import re
from glob import glob

# Standard EDA layer colors (lowercase layer names)
LAYER_COLORS = {
    # NanGate45
    "metal1": "#4488ff",
    "metal2": "#ff4444",
    "metal3": "#44bb44",
    "metal4": "#ff8800",
    "metal5": "#aa44ff",
    "metal6": "#00cccc",
    "metal7": "#ff44cc",
    "metal8": "#888800",
    "metal9": "#008888",
    "metal10": "#884400",
    # sky130
    "li1": "#ffcc44",
    "met1": "#4488ff",
    "met2": "#ff4444",
    "met3": "#44bb44",
    "met4": "#ff8800",
    "met5": "#aa44ff",
    # asap7
    "m1": "#4488ff",
    "m2": "#ff4444",
    "m3": "#44bb44",
    "m4": "#ff8800",
    "m5": "#aa44ff",
    "m6": "#00cccc",
    "m7": "#ff44cc",
    "m8": "#888800",
    "m9": "#008888",
}

STAGE_LABELS = {
    "1_synth": "Synthesis",
    "2_floorplan": "Floorplan",
    "3_place": "Placement",
    "3_5_place_dp": "Detail Placement",
    "4_cts": "CTS",
    "5_1_grt": "Global Route",
    "5_route": "Route",
    "6_final": "Final",
}

# Simple in-memory cache: (platform, design, stage) -> parsed result
_cache = {}


def parse_timing_reports(platform, design, flow_dir):
    """
    Parse all available stage timing reports and return a dict of:
      { stage_label: { wns, tns, worst_slack, fmax, period_min, setup_skew,
                       power_mw, power_breakdown, wirelength_um, utilization,
                       critical_path } }
    plus a 'summary' key with the final (6_finish) values.
    Missing values are None.
    """
    import json as _json

    nickname = get_design_nickname(platform, design, flow_dir)
    reports_dir = os.path.join(flow_dir, "reports", platform, nickname, "base")
    logs_dir = os.path.join(flow_dir, "logs", platform, nickname, "base")

    # (rpt_filename, label, json_prefix_glob)
    REPORT_STAGES = [
        ("3_detailed_place.rpt", "Detailed Place", "3_5_place_dp"),
        ("4_cts_final.rpt", "CTS", "4_1_cts"),
        ("5_global_route.rpt", "Global Route", "5_1_grt"),
        ("6_finish.rpt", "Final", "6_report"),
    ]

    def _parse_rpt(path):
        if not os.path.exists(path):
            return None
        metrics = {
            "wns": None,
            "tns": None,
            "worst_slack": None,
            "fmax": None,
            "period_min": None,
            "setup_skew": None,
        }
        critical_path = []
        in_path = False
        with open(path, errors="replace") as f:
            for line in f:
                s = line.strip()
                m = re.match(r"^tns\s+\S+\s+([-\d.]+)", s)
                if m:
                    metrics["tns"] = float(m.group(1))
                m = re.match(r"^wns\s+\S+\s+([-\d.]+)", s)
                if m:
                    metrics["wns"] = float(m.group(1))
                m = re.match(r"^worst slack\s+\S+\s+([-\d.]+)", s)
                if m:
                    metrics["worst_slack"] = float(m.group(1))
                m = re.search(r"period_min\s*=\s*([\d.]+)\s+fmax\s*=\s*([\d.]+)", s)
                if m:
                    metrics["period_min"] = float(m.group(1))
                    metrics["fmax"] = float(m.group(2))
                m = re.match(r"([-\d.]+)\s+setup skew", s)
                if m:
                    metrics["setup_skew"] = float(m.group(1))
                # Critical path parsing — only capture first path found
                if not critical_path:
                    if re.match(r"Path Type:", s):
                        in_path = True
                    if in_path:
                        # Data lines: <fanout> <cap> <slew> <delay> <arrival> <desc>
                        pm = re.match(
                            r"^\s*\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+(.+)$",
                            line,
                        )
                        if pm:
                            desc = pm.group(3).strip()
                            entry_type = "net" if "(net)" in desc else "cell"
                            cell_name_m = re.match(r"(\S+)", desc)
                            critical_path.append(
                                {
                                    "name": (
                                        cell_name_m.group(1) if cell_name_m else desc
                                    ),
                                    "delay_ps": round(float(pm.group(1)) * 1000, 1),
                                    "arrival_ps": round(float(pm.group(2)) * 1000, 1),
                                    "type": entry_type,
                                }
                            )
                        # End of path block
                        if s.startswith("slack"):
                            in_path = False
        metrics["critical_path"] = critical_path
        return metrics

    def _parse_json_metrics(json_stem):
        path = os.path.join(logs_dir, json_stem + ".json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                d = _json.load(f)
        except Exception:
            return {}
        # Collect power fields — key prefixes vary by stage
        power_internal = next((v for k, v in d.items() if "power__internal" in k), None)
        power_switch = next((v for k, v in d.items() if "power__switching" in k), None)
        power_leak = next(
            (v for k, v in d.items() if "power__leakage__total" in k), None
        )
        power_total = next(
            (v for k, v in d.items() if k.endswith("power__total")), None
        )
        wirelength = next(
            (v for k, v in d.items() if "wirelength" in k and "estimated" not in k),
            None,
        )
        util = next(
            (v for k, v in d.items() if "utilization" in k and "design__instance" in k),
            None,
        )
        result = {}
        if power_total is not None:
            result["power_mw"] = round(power_total * 1000, 4)
            result["power_breakdown"] = {
                "internal_mw": round((power_internal or 0) * 1000, 4),
                "switching_mw": round((power_switch or 0) * 1000, 4),
                "leakage_mw": round((power_leak or 0) * 1000, 4),
            }
        if wirelength is not None:
            result["wirelength_um"] = wirelength
        if util is not None:
            result["utilization"] = round(util, 4)
        return result

    stages = {}
    for filename, label, json_stem in REPORT_STAGES:
        path = os.path.join(reports_dir, filename)
        result = _parse_rpt(path)
        if result is not None:
            result.update(_parse_json_metrics(json_stem))
            stages[label] = result

    return {
        "stages": stages,
        "summary": stages.get("Final")
        or (list(stages.values())[-1] if stages else None),
        "reports_available": list(stages.keys()),
    }


def get_design_nickname(platform, design, flow_dir):
    config = os.path.join(flow_dir, "designs", platform, design, "config.mk")
    if os.path.exists(config):
        with open(config) as f:
            for line in f:
                m = re.match(r"export\s+DESIGN_NICKNAME\s*[?:]?=\s*(\S+)", line.strip())
                if m:
                    return m.group(1)
    return design


def find_lef_files(platform, flow_dir):
    platform_dir = os.path.join(flow_dir, "platforms", platform)
    lefs = glob(os.path.join(platform_dir, "lef", "*.lef")) + glob(
        os.path.join(platform_dir, "*.lef")
    )
    return list(set(lefs))


def parse_lef_macros(lef_paths):
    """Return {cellName: {w, h, is_macro}} with dimensions in microns."""
    macros = {}
    for path in lef_paths:
        if not os.path.exists(path):
            continue
        name = None
        with open(path, errors="replace") as f:
            for raw in f:
                line = raw.strip()
                m = re.match(r"^MACRO\s+(\S+)", line)
                if m:
                    name = m.group(1)
                    macros[name] = {"w": 0.0, "h": 0.0, "is_macro": False}
                elif name:
                    if line.startswith("CLASS"):
                        cls = line.split()[1].upper() if len(line.split()) > 1 else ""
                        macros[name]["is_macro"] = cls == "BLOCK"
                    elif line.startswith("SIZE"):
                        m2 = re.match(r"SIZE\s+([\d.]+)\s+BY\s+([\d.]+)", line)
                        if m2:
                            macros[name]["w"] = float(m2.group(1))
                            macros[name]["h"] = float(m2.group(2))
                    elif line == f"END {name}":
                        name = None
    return macros


def get_available_stages(platform, design, flow_dir):
    """
    Return stages that have either an existing DEF file or an ODB that can be
    converted to DEF on demand.  Each entry includes 'odb_path' when conversion
    is needed so the caller knows to trigger the export first.
    """
    nickname = get_design_nickname(platform, design, flow_dir)
    results_dir = os.path.join(flow_dir, "results", platform, nickname, "base")
    if not os.path.exists(results_dir):
        return []

    # Collect ODB checkpoints that are named like a stage
    seen_stems = set()
    stages = []

    # Prefer existing DEF files first
    for def_file in sorted(glob(os.path.join(results_dir, "*.def"))):
        stem = os.path.splitext(os.path.basename(def_file))[0]
        seen_stems.add(stem)
        stages.append(
            {
                "id": stem,
                "label": STAGE_LABELS.get(stem, stem),
                "def_path": def_file,
                "odb_path": None,
            }
        )

    # Add ODB stages without a corresponding DEF
    for odb_file in sorted(glob(os.path.join(results_dir, "*.odb"))):
        stem = os.path.splitext(os.path.basename(odb_file))[0]
        if stem in seen_stems or stem not in STAGE_LABELS:
            continue
        seen_stems.add(stem)
        def_path = odb_file.replace(".odb", ".def")
        stages.append(
            {
                "id": stem,
                "label": STAGE_LABELS.get(stem, stem),
                "def_path": def_path,  # target path (may not exist yet)
                "odb_path": odb_file,  # source to convert from
            }
        )

    # Sort by stage label order
    order = list(STAGE_LABELS.keys())
    stages.sort(key=lambda s: order.index(s["id"]) if s["id"] in order else 99)
    return stages


def _iter_statements(f):
    """Yield complete DEF statements (terminated by ';' or 'END <section>')."""
    buf = []
    for raw in f:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        buf.append(line)
        if ";" in line or (line.startswith("END ") and len(line.split()) == 2):
            yield " ".join(buf)
            buf = []
    if buf:
        yield " ".join(buf)


def _parse_route_segments(text):
    """Extract (layer, width, [(x1,y1,x2,y2), ...]) groups from a routing statement."""
    results = []
    for chunk in re.split(r"\+\s+ROUTED\b|\bNEW\b", text):
        m = re.match(r"\s*(\S+)\s+(\d+)", chunk)
        if not m:
            continue
        layer = m.group(1).lower()
        # skip DEF keywords that can appear here
        if layer in {
            "use",
            "shape",
            "style",
            "source",
            "net",
            "taper",
            "noshape",
            "virtual",
            "fix",
            "cover",
            "+",
            "end",
        }:
            continue
        width = int(m.group(2))
        segs = re.findall(
            r"\(\s*([\d*.-]+)\s+([\d*.-]+)\s*\)\s+\(\s*([\d*.-]+)\s+([\d*.-]+)\s*\)",
            chunk,
        )
        coords = []
        for s in segs:
            try:
                coords.append((float(s[0]), float(s[1]), float(s[2]), float(s[3])))
            except ValueError:
                pass
        if coords:
            results.append((layer, width, coords))
    return results


def parse_def(def_path, macro_sizes, max_routes=60000):
    key = def_path
    if key in _cache:
        return _cache[key]

    result = {
        "dbUnitsPerMicron": 1000,
        "dieArea": [0, 0, 100000, 100000],
        "components": [],
        "specialRoutes": [],
        "routes": [],
        "pins": [],
        "layers": [],
        "layerColors": {},
        "routesTruncated": False,
    }

    if not os.path.exists(def_path):
        return None

    section = None
    layers_seen = set()
    route_count = 0

    with open(def_path, errors="replace") as f:
        for stmt in _iter_statements(f):
            # ── Top-level keywords ──────────────────────────────────────────
            if re.match(r"^UNITS\s+DISTANCE\s+MICRONS", stmt, re.I):
                m = re.search(r"MICRONS\s+(\d+)", stmt, re.I)
                if m:
                    result["dbUnitsPerMicron"] = int(m.group(1))

            elif stmt.upper().startswith("DIEAREA"):
                pts = re.findall(r"\(\s*([\d.-]+)\s+([\d.-]+)\s*\)", stmt)
                if len(pts) >= 2:
                    xs = [float(p[0]) for p in pts]
                    ys = [float(p[1]) for p in pts]
                    result["dieArea"] = [min(xs), min(ys), max(xs), max(ys)]

            # ── Section transitions ─────────────────────────────────────────
            elif re.match(r"^COMPONENTS\s+\d+", stmt, re.I):
                section = "comp"
            elif re.match(r"^END\s+COMPONENTS", stmt, re.I):
                section = None
            elif re.match(r"^SPECIALNETS\s+\d+", stmt, re.I):
                section = "special"
            elif re.match(r"^END\s+SPECIALNETS", stmt, re.I):
                section = None
            elif re.match(r"^NETS\s+\d+", stmt, re.I):
                section = "nets"
            elif re.match(r"^END\s+NETS", stmt, re.I):
                section = None
            elif re.match(r"^PINS\s+\d+", stmt, re.I):
                section = "pins"
            elif re.match(r"^END\s+PINS", stmt, re.I):
                section = None

            # ── Components ──────────────────────────────────────────────────
            elif section == "comp" and stmt.startswith("- "):
                m = re.search(
                    r"^-\s+(\S+)\s+(\S+).*?\+\s+(?:PLACED|FIXED|COVER)\s+"
                    r"\(\s*([\d.-]+)\s+([\d.-]+)\s*\)\s+(\S+)",
                    stmt,
                )
                if m:
                    cell = m.group(2)
                    dbu = result["dbUnitsPerMicron"]
                    sz = macro_sizes.get(cell, {"w": 0.0, "h": 0.0, "is_macro": False})
                    result["components"].append(
                        {
                            "name": m.group(1),
                            "cell": cell,
                            "x": float(m.group(3)),
                            "y": float(m.group(4)),
                            "w": sz["w"] * dbu,
                            "h": sz["h"] * dbu,
                            "orient": m.group(5),
                            "is_macro": sz.get("is_macro", False),
                        }
                    )

            # ── Special nets (power/ground routing) ─────────────────────────
            elif section == "special" and stmt.startswith("- "):
                for layer, width, coords in _parse_route_segments(stmt):
                    layers_seen.add(layer)
                    for x1, y1, x2, y2 in coords:
                        result["specialRoutes"].append(
                            {
                                "layer": layer,
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "w": width,
                            }
                        )

            # ── Signal routing ──────────────────────────────────────────────
            elif (
                section == "nets"
                and stmt.startswith("- ")
                and not result["routesTruncated"]
            ):
                for layer, width, coords in _parse_route_segments(stmt):
                    layers_seen.add(layer)
                    for x1, y1, x2, y2 in coords:
                        result["routes"].append(
                            {
                                "layer": layer,
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "w": width,
                            }
                        )
                        route_count += 1
                        if route_count >= max_routes:
                            result["routesTruncated"] = True
                            break
                    if result["routesTruncated"]:
                        break

            # ── Pins ────────────────────────────────────────────────────────
            elif section == "pins" and stmt.startswith("- "):
                mp = re.match(r"^-\s+(\S+)", stmt)
                mpl = re.search(
                    r"\+\s+PLACED\s+\(\s*([\d.-]+)\s+([\d.-]+)\s*\)\s+(\S+)", stmt
                )
                if mp and mpl:
                    result["pins"].append(
                        {
                            "name": mp.group(1),
                            "x": float(mpl.group(1)),
                            "y": float(mpl.group(2)),
                        }
                    )

    result["layers"] = sorted(layers_seen)
    result["layerColors"] = {
        l: LAYER_COLORS.get(l, "#888888") for l in result["layers"]
    }
    _cache[key] = result
    return result
