"""
Extract IR-drop labels from a routed ODB using OpenROAD's native PDNSim
(`analyze_power_grid`) — no external simulator needed, unlike the thermal
track's HotSpot dependency.

Pipeline:
  1. Write a small Tcl driver that loads liberty (needed by read_spef),
     the routed ODB, and (optionally) the SPEF, then runs
     `analyze_power_grid -voltage_file ...` for the requested net.
  2. Run it via `openroad -no_init -exit <driver.tcl>` and parse the
     per-instance voltage CSV it writes.
  3. Separately open the ODB via the `openroad.Design`/`Tech` Python API
     (same pattern as extract_thermal_labels.py) to rasterize PDN wire
     geometry (stripe_density, via_density) and a cell-power-weighted
     current-density proxy on the same grid.
  4. Bin the per-instance voltages onto the grid, nearest-neighbor fill
     any empty cells, and save everything to one .npz.

`analyze_power_grid -voltage_file` CSV format (verified against a real
nangate45/gcd `make finish` run, 997 data rows for a 997-instance design):
    Instance,Terminal,Layer,X location,Y location,Voltage
    FILLER_23_180,VDD,metal1,35.4350,33.6000,1.099878
    ...
  - One row per instance terminal on the analyzed net (dense, not a sparse
    PDN-node sample) — X/Y are in **microns**, Voltage is raw volts on the
    analyzed net. -error_file is only written when PDNSim hits an
    error/warning, so it is treated as optional/diagnostic here, not a
    required output.

Run inside Docker (works with the stock openroad/orfs:latest — no HotSpot
or -ml image needed, since PDNSim ships with OpenROAD itself):
  openroad -python extract_irdrop_labels.py \\
      --odb      <6_final.odb> \\
      --spef     <6_final.spef> \\
      --liberty  <platform lib>/NangateOpenCellLibrary_typical.lib \\
      --out      <path_irdrop_labels.npz> \\
      [--grid 64] [--net VDD] [--voltage 1.1]
"""

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from openroad import Design, Tech

# Cell-type power weight multipliers for the current_density_proxy channel.
# Duplicated from extract_thermal_labels.py's _cell_power_weight — same
# convention (clock/sequential cells dissipate more current per unit area
# than combinational logic), kept as a separate copy here since the two
# extractors are independent, additive tracks and extract_thermal_labels.py
# must not be modified.
_CLOCK_WEIGHT = 5.0
_SEQUENTIAL_WEIGHT = 3.0
_MACRO_WEIGHT = 2.0
_COMBINATIONAL_WEIGHT = 1.0

_CLOCK_NAME_TOKENS = (
    "clkbuf",
    "ckbuf",
    "clkinv",
    "ckinv",
    "icg",
    "clkgate",
    "ckgate",
    "clkdly",
    "clkand",
    "clkor",
    "clkmux",
)
_SEQUENTIAL_NAME_TOKENS = (
    "dff",
    "sdff",
    "latch",
    "dlxtp",
    "dfxtp",
    "dlrtp",
    "dfrtp",
    "dfstp",
    "sdlxtp",
    "sdfxtp",
)


def _cell_power_weight(master_name: str, is_block: bool) -> float:
    """Power-density multiplier for a standard cell — see extract_thermal_labels.py."""
    if is_block:
        return _MACRO_WEIGHT

    name = master_name.lower()
    if any(tok in name for tok in _CLOCK_NAME_TOKENS):
        return _CLOCK_WEIGHT
    if any(tok in name for tok in _SEQUENTIAL_NAME_TOKENS):
        return _SEQUENTIAL_WEIGHT
    return _COMBINATIONAL_WEIGHT


def _parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--odb", required=True, help="Routed ODB (6_final.odb)")
    ap.add_argument(
        "--spef", default=None, help="SPEF for the same design (6_final.spef)"
    )
    ap.add_argument(
        "--liberty",
        nargs="*",
        default=None,
        help="Liberty file(s) for the design's cell library (platform LIB_FILES). "
        "Required if --spef is given — read_spef needs liberty loaded first "
        "or OpenSTA raises STA-2141.",
    )
    ap.add_argument("--out", required=True, help="Output .npz path for IR-drop labels")
    ap.add_argument(
        "--grid", type=int, default=64, help="Output grid resolution (default 64)"
    )
    ap.add_argument("--net", default="VDD", help="Power net to analyze (default VDD)")
    ap.add_argument(
        "--voltage",
        type=float,
        default=1.1,
        help="Nominal supply voltage in volts for --net (default 1.1)",
    )
    ap.add_argument(
        "--openroad-bin",
        default="openroad",
        help="Path to the openroad binary (default: 'openroad' on PATH)",
    )
    return ap.parse_args()


# ── Phase 1: run PDNSim via a Tcl driver ───────────────────────────────────


def _write_tcl_driver(
    odb_path: str,
    spef_path: str,
    liberty_paths: list,
    net: str,
    voltage: float,
    voltage_file: Path,
    error_file: Path,
    tcl_path: Path,
) -> None:
    lines = []
    for lib in liberty_paths or []:
        lines.append(f"read_liberty {{{lib}}}")
    lines.append(f"read_db {{{odb_path}}}")
    if spef_path:
        lines.append(f"read_spef {{{spef_path}}}")
    lines.append(f"set_pdnsim_net_voltage -net {net} -voltage {voltage}")
    lines.append(
        f"analyze_power_grid -net {net} "
        f"-voltage_file {{{voltage_file}}} -error_file {{{error_file}}}"
    )
    lines.append("exit")
    tcl_path.write_text("\n".join(lines) + "\n")


def run_pdnsim(
    openroad_bin: str,
    odb_path: str,
    spef_path: str,
    liberty_paths: list,
    net: str,
    voltage: float,
    work_dir: Path,
) -> Path:
    """Run analyze_power_grid via a Tcl driver, return path to the voltage CSV."""
    tcl_path = work_dir / "analyze_power_grid.tcl"
    voltage_file = work_dir / "voltage.rpt"
    error_file = work_dir / "error.rpt"

    _write_tcl_driver(
        odb_path,
        spef_path,
        liberty_paths,
        net,
        voltage,
        voltage_file,
        error_file,
        tcl_path,
    )

    cmd = [openroad_bin, "-no_init", "-exit", str(tcl_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"analyze_power_grid failed (exit {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    if not voltage_file.exists():
        raise RuntimeError(
            "analyze_power_grid did not produce a voltage file. "
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return voltage_file


# ── Phase 2: parse the voltage CSV ─────────────────────────────────────────


def parse_voltage_file(voltage_file: Path) -> list:
    """
    Parse `analyze_power_grid -voltage_file` output.

    Header: Instance,Terminal,Layer,X location,Y location,Voltage
    Verified against a real nangate45/gcd `make finish` run (997 rows for a
    997-instance design — one row per instance terminal on the analyzed
    net, dense rather than a sparse PDN-node sample). X/Y are in microns.

    Returns a list of (x_um, y_um, voltage) tuples.
    """
    rows = []
    with open(voltage_file, newline="") as f:
        reader = csv.reader(f)
        for parts in reader:
            if not parts:
                continue
            if parts[0].strip().lower() == "instance":
                continue  # header row
            if len(parts) < 6:
                continue
            try:
                x_um = float(parts[3])
                y_um = float(parts[4])
                voltage = float(parts[5])
            except ValueError:
                continue
            rows.append((x_um, y_um, voltage))
    return rows


# ── Phase 3: rasterize voltage samples + PDN geometry ──────────────────────


def _bin_voltage(
    rows: list, grid: int, x0_um: float, y0_um: float, die_w_um: float, die_h_um: float
) -> np.ndarray:
    """
    Bin (x_um, y_um, voltage) samples onto (grid, grid) by averaging samples
    that fall in the same cell, then nearest-neighbor fill any empty cells
    (same spatial-binning technique as build_power_grid in
    extract_thermal_labels.py, followed by a fill pass since instance
    terminals are not guaranteed to cover every cell).
    """
    sum_grid = np.zeros((grid, grid), dtype=np.float64)
    count_grid = np.zeros((grid, grid), dtype=np.float64)

    for x_um, y_um, voltage in rows:
        gx = int((x_um - x0_um) / die_w_um * grid)
        gy = int((y_um - y0_um) / die_h_um * grid)
        gx = min(max(gx, 0), grid - 1)
        gy = min(max(gy, 0), grid - 1)
        sum_grid[gy, gx] += voltage
        count_grid[gy, gx] += 1

    filled = count_grid > 0
    voltage_map = np.where(filled, sum_grid / np.maximum(count_grid, 1), 0.0)

    if not filled.all():
        voltage_map = _nearest_fill(voltage_map, filled)

    return voltage_map.astype(np.float32)


def _nearest_fill(values: np.ndarray, filled: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbor fill of empty cells with the value of the closest
    filled cell, via iterative single-cell dilation (Chebyshev distance).
    Pure numpy — no scipy, since the stock openroad/orfs:latest image (used
    by this extractor, unlike thermal's HotSpot image) doesn't have it.
    """
    out = values.copy()
    mask = filled.copy()
    grid = out.shape[0]
    # At most `grid` passes are ever needed to flood-fill a (grid, grid)
    # array from any non-empty starting mask.
    for _ in range(grid):
        if mask.all():
            break
        empty_y, empty_x = np.nonzero(~mask)
        newly_filled_cells = []
        for y, x in zip(empty_y, empty_x):
            y0, y1 = max(0, y - 1), min(grid, y + 2)
            x0, x1 = max(0, x - 1), min(grid, x + 2)
            neighborhood_mask = mask[y0:y1, x0:x1]
            if neighborhood_mask.any():
                neighborhood_vals = out[y0:y1, x0:x1]
                out[y, x] = neighborhood_vals[neighborhood_mask][0]
                newly_filled_cells.append((y, x))
        if not newly_filled_cells:
            break  # no filled cells reachable — leave remaining cells as-is
        for y, x in newly_filled_cells:
            mask[y, x] = True
    return out


def build_pdn_geometry_grids(block, grid: int, net_name: str) -> tuple:
    """
    Return (stripe_density, via_density, current_density_proxy), each
    (grid, grid) float32, rasterized from the placed/routed ODB.

    stripe_density: PDN wire segment area (um^2) per cell, from special
      wires (SWires) on power/ground nets — same grid-binning technique as
      build_power_grid in extract_thermal_labels.py (accumulate by segment
      midpoint), then normalized by cell area.
    via_density: count of PDN vias per cell, from SWire shapes where
      isVia() is true.
    current_density_proxy: cell-type-weighted standard-cell area per cell
      (same weighting as extract_thermal_labels.py's power_grid, renamed —
      current draw scales with the same clock/sequential/macro/
      combinational breakdown as leakage power), normalized by cell area.
    """
    die = block.getDieArea()
    x0_dbu, y0_dbu = die.xMin(), die.yMin()
    x1_dbu, y1_dbu = die.xMax(), die.yMax()
    die_w_dbu = x1_dbu - x0_dbu
    die_h_dbu = y1_dbu - y0_dbu
    dbu_per_um = block.getDbUnitsPerMicron()

    stripe_area = np.zeros((grid, grid), dtype=np.float64)
    via_count = np.zeros((grid, grid), dtype=np.float64)

    for net in block.getNets():
        sig_type = net.getSigType()
        is_target = net.getName() == net_name
        is_pwr_gnd = sig_type in ("POWER", "GROUND")
        if not (is_target or is_pwr_gnd):
            continue
        for swire in net.getSWires():
            for box in swire.getWires():
                cx = (box.xMin() + box.xMax()) / 2
                cy = (box.yMin() + box.yMax()) / 2
                gx = int((cx - x0_dbu) / die_w_dbu * grid)
                gy = int((cy - y0_dbu) / die_h_dbu * grid)
                gx = min(max(gx, 0), grid - 1)
                gy = min(max(gy, 0), grid - 1)

                if box.isVia():
                    via_count[gy, gx] += 1
                else:
                    w_um = (box.xMax() - box.xMin()) / dbu_per_um
                    h_um = (box.yMax() - box.yMin()) / dbu_per_um
                    stripe_area[gy, gx] += w_um * h_um

    current_area = np.zeros((grid, grid), dtype=np.float64)
    for inst in block.getInsts():
        bbox = inst.getBBox()
        cx = (bbox.xMin() + bbox.xMax()) / 2
        cy = (bbox.yMin() + bbox.yMax()) / 2
        gx = int((cx - x0_dbu) / die_w_dbu * grid)
        gy = int((cy - y0_dbu) / die_h_dbu * grid)
        gx = min(max(gx, 0), grid - 1)
        gy = min(max(gy, 0), grid - 1)

        w_um = (bbox.xMax() - bbox.xMin()) / dbu_per_um
        h_um = (bbox.yMax() - bbox.yMin()) / dbu_per_um
        area_um2 = w_um * h_um

        master = inst.getMaster()
        weight = _cell_power_weight(master.getName(), master.isBlock())
        current_area[gy, gx] += area_um2 * weight

    cell_area_um2 = (die_w_dbu / dbu_per_um / grid) * (die_h_dbu / dbu_per_um / grid)
    stripe_density = (stripe_area / (cell_area_um2 + 1e-9)).astype(np.float32)
    via_density = via_count.astype(np.float32)
    current_density_proxy = (current_area / (cell_area_um2 + 1e-9)).astype(np.float32)

    return stripe_density, via_density, current_density_proxy


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    args = _parse_args()
    grid = args.grid

    if args.spef and not args.liberty:
        raise SystemExit(
            "--liberty is required when --spef is given "
            "(read_spef needs liberty loaded first, or OpenSTA raises STA-2141)."
        )

    print(
        f"[irdrop] Running analyze_power_grid on net {args.net} "
        f"({args.voltage} V nominal)"
    )
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        voltage_file = run_pdnsim(
            args.openroad_bin,
            args.odb,
            args.spef,
            args.liberty,
            args.net,
            args.voltage,
            work_dir,
        )
        rows = parse_voltage_file(voltage_file)

    print(f"[irdrop] Parsed {len(rows)} voltage samples from PDNSim output")
    if not rows:
        raise RuntimeError(
            "No voltage samples parsed from analyze_power_grid output — "
            "check that the ODB/SPEF/net are consistent with the CSV format "
            "documented in this file's docstring."
        )

    tech = Tech()
    design = Design(tech)
    design.readDb(args.odb)
    block = design.getBlock()

    die = block.getDieArea()
    dbu_per_um = block.getDbUnitsPerMicron()
    x0_um = die.xMin() / dbu_per_um
    y0_um = die.yMin() / dbu_per_um
    die_w_um = (die.xMax() - die.xMin()) / dbu_per_um
    die_h_um = (die.yMax() - die.yMin()) / dbu_per_um
    print(f"[irdrop] Die: {die_w_um:.2f}x{die_h_um:.2f} um  Grid: {grid}x{grid}")

    voltage_map = _bin_voltage(rows, grid, x0_um, y0_um, die_w_um, die_h_um)
    irdrop_map = (args.voltage - voltage_map).astype(np.float32)

    stripe_density, via_density, current_density_proxy = build_pdn_geometry_grids(
        block, grid, args.net
    )

    print(
        f"[irdrop] Voltage: min={voltage_map.min():.4f}V max={voltage_map.max():.4f}V  "
        f"Drop: min={irdrop_map.min()*1e3:.2f}mV max={irdrop_map.max()*1e3:.2f}mV"
    )
    print(
        f"[irdrop] stripe_density: mean={stripe_density.mean():.4f}  "
        f"via_density: total={via_density.sum():.0f}  "
        f"current_density_proxy: mean={current_density_proxy.mean():.4f}"
    )

    np.savez(
        args.out,
        irdrop_map=irdrop_map,
        voltage_map=voltage_map,
        stripe_density=stripe_density,
        via_density=via_density,
        current_density_proxy=current_density_proxy,
    )
    print(f"[irdrop] Saved -> {args.out}")


if __name__ == "__main__":
    main()
