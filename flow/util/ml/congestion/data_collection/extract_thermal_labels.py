"""
Extract thermal labels from a placed ODB using HotSpot v7.

Pipeline:
  1. Read ODB  →  get per-instance (position, area)
  2. Use cell area as a leakage-power proxy, binned onto a grid
  3. Write HotSpot .flp (floorplan) and .ptrace (power trace) files
  4. Run HotSpot in steady-state block-model mode
  5. Parse .steady output  →  thermal heatmap on the same 64x64 grid
     used by congestion labels

Why area as a power proxy?
  Steady-state temperature is dominated by power density distribution,
  and leakage power in standard cells scales roughly linearly with area
  for a given process node. This gives a realistic relative thermal map
  without needing switching activity or full STA.

Power scaling — constant power density (W/mm²):
  Total power is set to POWER_DENSITY_W_PER_MM2 × die_area so that
  power density is constant across all designs and process nodes.
  With fixed total power (the old approach), a 50 µm asap7 die got
  the same 500 mW as a 3 mm ariane133 die, producing absurd 2000°C
  temperatures on tiny dies. With constant density, steady-state
  temperatures become comparable across designs (~100°C range), making
  the thermal labels physically consistent for cross-design training.

Adaptive grid:
  HotSpot's block RC model becomes singular when cells are smaller than
  ~5 µm. For small dies (asap7 at 50 µm × 50 µm), the HotSpot run uses
  a coarser grid (capped so each cell is ≥ MIN_CELL_UM, and never below
  MIN_HOTSPOT_GRID), then the result is bilinearly upsampled to the
  target grid (default 64) for training consistency. The saved
  power_grid is always at the target resolution. A die too small to
  give a sane grid is not silently collapsed further — it is left to
  the degeneracy check in main() to fail loudly instead.

Package model:
  HotSpot's built-in package defaults (silicon die, heat spreader, heat
  sink) are sized for cm-scale chips (s_spreader=3cm, s_sink=6cm). Every
  ORFS test design is orders of magnitude smaller (9µm-1mm dies), so
  with the defaults the spreader/sink conduct heat away laterally far
  faster than the die can develop a spatial gradient — die size becomes
  a confound that swamps the actual power distribution: the same
  relative power map produces 0.05°C peak-to-peak at 51µm but 14.9°C at
  1021µm. hotspot_package_args() rescales chip thickness, spreader, and
  sink geometry so they scale with each design's own die extent, which
  substantially reduces (but, per tests/thermal_scale_check.py and
  DESIGN_RUNS.md, does not eliminate) this confound, and sets r_convec
  from total power to control one specific *contribution* to mean die
  temperature rise (see TARGET_MEAN_RISE_K below) — it does not pin the
  actual total mean rise, which in the shipped 12-design dataset is
  90-95K (mean temps 135-140°C), not TARGET_MEAN_RISE_K's 40K. The
  remaining ~50-60K comes from the rest of the package stack
  (spreader/sink/interface resistances), which r_convec does not
  control and which itself varies with die size — see DESIGN_RUNS.md
  for the measured numbers.

Run inside Docker (requires openroad/orfs-ml:latest):
  openroad -python extract_thermal_labels.py \\
      --odb  <3_place.odb or 5_1_grt.odb> \\
      --out  <path_thermal_labels.npz>     \\
      [--grid 64] [--power-density 10.0]
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# HotSpot's block model becomes ill-conditioned below this cell dimension.
# Derived empirically: 5 µm gives stable LU decomposition across all tested
# process nodes (asap7 at 7nm, nangate45, sky130hd).
MIN_CELL_UM = 5.0

# Silicon die thickness as a fraction of the die's equivalent square edge
# (sqrt(die_w * die_h)). Calibrated (see tests/thermal_scale_check.py) to
# eliminate the near-total isothermal collapse the unscaled HotSpot default
# package produced at small die sizes. It does NOT make the normalized
# thermal pattern scale-invariant: per DESIGN_RUNS.md's calibration sweep,
# relative contrast still grows faster-than-linearly with die extent (a
# ~52x range from 9µm to 102µm), and the smallest shipped die (asap7/gcd,
# 8.98µm) produces only ~0.24-0.41°C peak-to-peak, not 1-20°C.
CHIP_THICKNESS_FRAC = 0.02
T_CHIP_MIN_M = 1e-7  # floor on t_chip (0.1 µm), a numerical-safety-only
# floor (prevents zero/negative thickness for pathological micro-dies), not
# a physical scaling choice. It no longer clamps any of the 12 shipped
# designs (previously 1e-6 = 1 µm clamped every die below CHIP_THICKNESS_FRAC
# * l_eq < 1 µm, i.e. l_eq < 50 µm, which silently flattened asap7/gcd's
# 8.98 µm die and nangate45/gcd's 36.73 µm die — see DESIGN_RUNS.md).
T_CHIP_MAX_M = 1.5e-4  # ceiling = HotSpot's own default

# Heat-spreader and heat-sink lateral extent, as multiples of the die's
# longer edge, so package geometry scales with die size instead of using
# HotSpot's fixed cm-scale defaults (s_spreader=3cm, s_sink=6cm).
SPREADER_RATIO = 2.0
SINK_RATIO = 4.0
SPREADER_T_FRAC = 0.10  # t_spreader as a fraction of s_spreader
SINK_T_FRAC = 0.115  # t_sink/s_sink, matches HotSpot's own default ratio

# r_convec is chosen per-design so it contributes this much to the die's
# mean temperature rise above ambient. It is NOT the total mean rise: the
# rest of the package stack (spreader/sink/interface resistances) adds
# another ~50-60K on top, and that additional contribution itself varies
# with die size (it is not held constant by this constant). In the shipped
# 12-design dataset, real mean temperatures are 135-140°C (rise 90-95K),
# not ambient+40K. Do not treat this as pinning absolute temperature —
# what the labels encode is on-die spatial non-uniformity, and that is
# also not fully scale-invariant (see DESIGN_RUNS.md).
TARGET_MEAN_RISE_K = 40.0

# Hard lower bound on the adaptive HotSpot grid (see _adaptive_hotspot_grid).
# Note: at MIN_HOTSPOT_GRID=8, a die below ~40µm produces cells smaller than
# MIN_CELL_UM (5µm) — exactly the regime this module's own docstring says
# makes HotSpot's block RC model ill-conditioned. It has converged for every
# die tested so far (down to asap7/gcd's 8.98µm), but nothing bounds this;
# a different sub-40µm design could still hit a HotSpot solver failure
# (`lupdcmp: singular matrix`). That is a loud failure (non-zero exit,
# caught by run_hotspot()'s RuntimeError), not a silent one, so it is a
# known risk to watch for rather than something this fix blocks on.
MIN_HOTSPOT_GRID = 8

# Below this peak-to-peak (°C), a thermal map counts as degenerate/constant
# and extraction fails loudly instead of silently saving flat labels.
DEGENERATE_PTP_C = 1e-3

# Uniform baseline added to every power cell to prevent zero-power rows in the
# thermal conductance matrix (which cause singular-matrix errors in lupdcmp).
# Expressed as a fraction of total_power_w / number_of_cells.
POWER_FLOOR_FRAC = 0.01

# Power density used to compute total chip power from die area.
# Calibrated so nangate45 designs (~0.06 mm²) get ~100°C with HotSpot's
# default package model. Constant density means temperature is comparable
# across all designs and process nodes.
POWER_DENSITY_W_PER_MM2 = 10.0

# Cell-type power weight multipliers, applied to area before it's used as the
# leakage/power proxy. Clock-network and sequential cells dissipate several
# times more power per unit area than combinational logic (higher switching
# activity, larger drive strength); using raw area alone produces a nearly
# uniform power (and therefore thermal) grid.
#
# Name matching is a lowercased *substring* test with no word-boundary
# requirement, so it matches concatenated PDK naming schemes uniformly, e.g.:
#   nangate45: CLKBUF_X1, DFF_X1
#   asap7:     CKBUFx2_ASAP7_75t_R, DFFHQNx1_ASAP7_75t_R, ICGx1_ASAP7_75t_R
#   sky130hd:  sky130_fd_sc_hd__clkbuf_1, sky130_fd_sc_hd__dfxtp_1
# A word-boundary/exact-token match (e.g. r"\bDFF\b") would miss asap7's
# concatenated names like "DFFHQNx1" — substring matching avoids that class
# of bug entirely.
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
# Note: sky130's "sky130_fd_sc_hd__*" prefix contains "fd_" in every cell name
# (standard-cell library tag, unrelated to flip-flops) — a bare "fd_"/"_fd_"
# token would misclassify the entire sky130 library as sequential, so it's
# deliberately excluded here.


def _cell_power_weight(master_name: str, is_block: bool) -> float:
    """
    Power-density multiplier for a standard cell, based on its master name
    and whether it's a hard macro (master.isBlock()). Macros get their own
    weight regardless of name; standard cells are classified by name substring.
    """
    if is_block:
        return _MACRO_WEIGHT

    name = master_name.lower()
    if any(tok in name for tok in _CLOCK_NAME_TOKENS):
        return _CLOCK_WEIGHT
    if any(tok in name for tok in _SEQUENTIAL_NAME_TOKENS):
        return _SEQUENTIAL_WEIGHT
    return _COMBINATIONAL_WEIGHT


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--odb", required=True, help="Placed ODB (3_place.odb or 5_1_grt.odb)"
    )
    ap.add_argument("--out", required=True, help="Output .npz path for thermal labels")
    ap.add_argument(
        "--grid",
        type=int,
        default=64,
        help="Output grid resolution (default 64 → 64x64). "
        "HotSpot may use a coarser grid internally for small dies.",
    )
    ap.add_argument(
        "--power-density",
        type=float,
        default=POWER_DENSITY_W_PER_MM2,
        help=f"Power density in W/mm² (default {POWER_DENSITY_W_PER_MM2}). "
        "Total power = density × die_area, keeping temperatures "
        "physically consistent across process nodes.",
    )
    return ap.parse_args()


# ── Phase 1: power grid from ODB ───────────────────────────────────────────


def _dbu_to_m(val: float, dbu_per_um: float) -> float:
    """Convert OpenDB database units → metres."""
    return val / dbu_per_um * 1e-6


def _adaptive_hotspot_grid(die_w_m: float, die_h_m: float, target_grid: int) -> int:
    """
    Return the HotSpot grid size to use, capped so each cell is >= MIN_CELL_UM,
    and never below MIN_HOTSPOT_GRID. If the die is large enough for the full
    target_grid, returns target_grid.

    A die too small to give MIN_CELL_UM cells at MIN_HOTSPOT_GRID is
    deliberately over-gridded here rather than collapsed to an even coarser
    grid; the resulting thermal map is caught by _check_nondegenerate()
    instead of being silently accepted.
    """
    min_dim_um = min(die_w_m, die_h_m) * 1e6
    max_grid = int(min_dim_um / MIN_CELL_UM)
    return min(target_grid, max(MIN_HOTSPOT_GRID, max_grid))


def hotspot_package_args(
    die_w_m: float, die_h_m: float, total_power_w: float
) -> list:
    """
    Build HotSpot CLI package-model overrides scaled to this design's die
    extent, intended to make the solve scale-covariant: the same relative
    power map should produce a similar normalized thermal pattern at any
    die size. In practice this is only partially achieved (see
    DESIGN_RUNS.md's calibration sweep: relative contrast still grows
    faster-than-linearly with die extent, a ~52x range from 9-102µm). All
    geometry (chip thickness, spreader, sink) scales with the die; r_convec
    is the one non-geometric knob, set from total power so it contributes
    exactly TARGET_MEAN_RISE_K to mean die temperature rise — it does not
    pin *total* mean rise, which also picks up ~50-60K from the rest of the
    package stack (spreader/sink/interface), itself varying with die size.
    Real shipped mean temperatures are 135-140°C, not ambient+40K. The
    residual ΔT-vs-die-size growth is real physics from a finite-thickness
    package model and is NOT rescued by ThermalDataset's per-sample min-max
    normalization: min-max removes absolute magnitude but preserves
    relative shape/sharpness, and sharpness itself is what varies with die
    size here (normalized-map correlation between the smallest and largest
    calibration extents drops to 0.89, not ~1.0). Do not assume pooling
    across die sizes is safe without accounting for this.
    """
    l_eq = np.sqrt(die_w_m * die_h_m)
    l_max = max(die_w_m, die_h_m)

    t_chip = min(max(CHIP_THICKNESS_FRAC * l_eq, T_CHIP_MIN_M), T_CHIP_MAX_M)

    s_spreader = SPREADER_RATIO * l_max
    t_spreader = SPREADER_T_FRAC * s_spreader

    s_sink = SINK_RATIO * l_max
    t_sink = SINK_T_FRAC * s_sink

    r_convec = TARGET_MEAN_RISE_K / max(total_power_w, 1e-12)

    return [
        "-t_chip", f"{t_chip:.6e}",
        "-s_spreader", f"{s_spreader:.6e}",
        "-t_spreader", f"{t_spreader:.6e}",
        "-s_sink", f"{s_sink:.6e}",
        "-t_sink", f"{t_sink:.6e}",
        "-r_convec", f"{r_convec:.6e}",
    ]


def build_power_grid(block, grid: int, total_power_w: float) -> tuple:
    """
    Return (power_grid_W, die_bounds_m) where power_grid_W is (grid, grid)
    in Watts and die_bounds_m is (x0, y0, width, height) in metres.

    Cell area (µm²), scaled by a cell-type power weight (_cell_power_weight:
    clock cells 5x, sequential 3x, macros 2x, combinational 1x), is
    accumulated per bin. The whole grid is then rescaled so it sums to
    total_power_w — weighting only reshapes the spatial distribution, it does
    not change total chip power. A small uniform floor (POWER_FLOOR_FRAC of
    total) is added to every cell to prevent zero-power rows from making
    HotSpot's conductance matrix singular.
    """
    die = block.getDieArea()
    x0_dbu, y0_dbu = die.xMin(), die.yMin()
    x1_dbu, y1_dbu = die.xMax(), die.yMax()
    die_w_dbu = x1_dbu - x0_dbu
    die_h_dbu = y1_dbu - y0_dbu

    dbu_per_um = block.getDbUnitsPerMicron()

    area_grid = np.zeros((grid, grid), dtype=np.float64)
    type_area_um2 = {
        "clock": 0.0,
        "sequential": 0.0,
        "macro": 0.0,
        "combinational": 0.0,
    }
    type_weighted = {
        "clock": 0.0,
        "sequential": 0.0,
        "macro": 0.0,
        "combinational": 0.0,
    }
    _weight_to_label = {
        _CLOCK_WEIGHT: "clock",
        _SEQUENTIAL_WEIGHT: "sequential",
        _MACRO_WEIGHT: "macro",
        _COMBINATIONAL_WEIGHT: "combinational",
    }

    for inst in block.getInsts():
        bbox = inst.getBBox()
        cx = (bbox.xMin() + bbox.xMax()) / 2
        cy = (bbox.yMin() + bbox.yMax()) / 2

        gx = int((cx - x0_dbu) / die_w_dbu * grid)
        gy = int((cy - y0_dbu) / die_h_dbu * grid)
        gx = min(max(gx, 0), grid - 1)
        gy = min(max(gy, 0), grid - 1)

        # Area in µm²
        w_um = (bbox.xMax() - bbox.xMin()) / dbu_per_um
        h_um = (bbox.yMax() - bbox.yMin()) / dbu_per_um
        area_um2 = w_um * h_um

        master = inst.getMaster()
        weight = _cell_power_weight(master.getName(), master.isBlock())
        area_grid[gy, gx] += area_um2 * weight

        label = _weight_to_label.get(weight, "combinational")
        type_area_um2[label] += area_um2
        type_weighted[label] += area_um2 * weight

    total_weighted_area = sum(type_weighted.values())
    if total_weighted_area > 0:
        print("Cell-type power breakdown:")
        for label in ("clock", "sequential", "macro", "combinational"):
            if type_area_um2[label] == 0:
                continue
            pct = 100.0 * type_weighted[label] / total_weighted_area
            print(
                f"  {label:13s} area={type_area_um2[label]:10.2f} um^2  "
                f"weighted_power={pct:5.1f}%"
            )

    # Rescale so total power equals the user-supplied value
    total_area = area_grid.sum()
    if total_area > 0:
        power_grid = area_grid / total_area * total_power_w
    else:
        power_grid = (
            np.ones((grid, grid), dtype=np.float64) * total_power_w / (grid * grid)
        )

    # Add power floor: prevents zero-power cells from creating singular rows.
    # Floor = POWER_FLOOR_FRAC * total / N_cells so it's small relative to peaks.
    floor_per_cell = POWER_FLOOR_FRAC * total_power_w / (grid * grid)
    power_grid = np.maximum(power_grid, floor_per_cell)
    # Renormalize to keep total power constant
    power_grid = power_grid / power_grid.sum() * total_power_w

    # Die bounds in metres for HotSpot
    die_x0_m = _dbu_to_m(x0_dbu, dbu_per_um)
    die_y0_m = _dbu_to_m(y0_dbu, dbu_per_um)
    die_w_m = _dbu_to_m(die_w_dbu, dbu_per_um)
    die_h_m = _dbu_to_m(die_h_dbu, dbu_per_um)

    return power_grid, (die_x0_m, die_y0_m, die_w_m, die_h_m)


# ── Phase 2: write HotSpot input files ─────────────────────────────────────


def write_hotspot_inputs(
    power_grid: np.ndarray, die_bounds_m: tuple, grid: int, work_dir: Path
) -> tuple:
    """
    Write a .flp (floorplan) and .ptrace (power trace) for HotSpot.

    HotSpot .flp format (one block per line):
        name  width_m  height_m  x_left_m  y_bottom_m

    HotSpot .ptrace format:
        <tab-separated block names>   ← header
        <tab-separated power values>  ← steady-state row (Watts)

    We name blocks u{row}_{col} so we can recover the grid position
    when parsing the temperature output.
    """
    die_x0, die_y0, die_w, die_h = die_bounds_m
    cell_w = die_w / grid
    cell_h = die_h / grid

    names = [f"u{gy}_{gx}" for gy in range(grid) for gx in range(grid)]

    flp_path = work_dir / "design.flp"
    with open(flp_path, "w") as f:
        for gy in range(grid):
            for gx in range(grid):
                name = f"u{gy}_{gx}"
                x_pos = die_x0 + gx * cell_w
                y_pos = die_y0 + gy * cell_h
                f.write(
                    f"{name}\t{cell_w:.6e}\t{cell_h:.6e}"
                    f"\t{x_pos:.6e}\t{y_pos:.6e}\n"
                )

    ptrace_path = work_dir / "design.ptrace"
    with open(ptrace_path, "w") as f:
        f.write("\t".join(names) + "\n")
        powers = [
            f"{power_grid[gy, gx]:.6e}" for gy in range(grid) for gx in range(grid)
        ]
        f.write("\t".join(powers) + "\n")

    return flp_path, ptrace_path


# ── Phase 3: run HotSpot ───────────────────────────────────────────────────


def run_hotspot(
    flp_path: Path, ptrace_path: Path, work_dir: Path, package_args: list
) -> Path:
    """
    Run HotSpot steady-state block model. Returns path to .steady output.

    package_args (from hotspot_package_args()) override HotSpot's built-in
    package defaults, which are sized for cm-scale chips (s_spreader=3cm,
    s_sink=6cm) — orders of magnitude larger than any ORFS test die
    (9µm-1mm). Without them, die size is a confound: the same relative
    power map produces 0.05°C peak-to-peak at 51µm, 1.2°C at 255µm, 4.7°C
    at 510µm, and 14.9°C at 1021µm, because the oversized spreader/sink
    conduct heat away laterally faster than the die can develop a spatial
    gradient. package_args is required (no default) so this can't be
    silently skipped. Scaling the package geometry (see
    hotspot_package_args()) substantially shrinks this confound but does
    not eliminate it — see DESIGN_RUNS.md for the residual magnitude.
    """
    steady_path = work_dir / "design.steady"
    cmd = [
        "hotspot",
        "-f",
        str(flp_path),
        "-p",
        str(ptrace_path),
        "-steady_file",
        str(steady_path),
        "-model_type",
        "block",
    ] + package_args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work_dir))
    if result.returncode != 0:
        raise RuntimeError(
            f"HotSpot exited with code {result.returncode}:\n{result.stderr}"
        )
    return steady_path


# ── Phase 4: parse HotSpot output ──────────────────────────────────────────


def parse_steady(steady_path: Path, grid: int) -> np.ndarray:
    """
    Parse HotSpot .steady file into a (grid, grid) float32 array in °C.

    .steady format (one block per line):
        block_name  temperature_K
    """
    temp_map = np.zeros((grid, grid), dtype=np.float32)
    with open(steady_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            name, temp_k = parts[0], float(parts[1])
            if not name.startswith("u"):
                continue  # skip package/spreader blocks HotSpot may add
            coords = name[1:]  # strip leading "u"
            gy, gx = map(int, coords.split("_", 1))
            temp_map[gy, gx] = temp_k - 273.15  # K → °C
    return temp_map


def _check_nondegenerate(
    temp_map: np.ndarray, power_grid: np.ndarray, hs_grid: int, die_w_m: float,
    die_h_m: float,
) -> None:
    """
    Raise RuntimeError if the extracted labels are degenerate: NaN/Inf,
    a constant thermal map, or a constant power grid. Must be called before
    np.savez so a degenerate design leaves no .npz file — that absence is
    how ThermalDataset excludes it.
    """
    if not np.all(np.isfinite(temp_map)) or not np.all(np.isfinite(power_grid)):
        raise RuntimeError(
            f"Non-finite values in thermal_map/power_grid "
            f"(grid={hs_grid}, die={die_w_m*1e6:.1f}x{die_h_m*1e6:.1f} um)"
        )
    ptp = float(np.ptp(temp_map))
    if ptp < DEGENERATE_PTP_C:
        raise RuntimeError(
            f"Degenerate thermal_map: peak-to-peak={ptp:.6f}°C < "
            f"{DEGENERATE_PTP_C}°C (grid={hs_grid}, "
            f"die={die_w_m*1e6:.1f}x{die_h_m*1e6:.1f} um)"
        )
    if np.ptp(power_grid) <= 0:
        raise RuntimeError(
            f"Degenerate power_grid: constant (grid={hs_grid}, "
            f"die={die_w_m*1e6:.1f}x{die_h_m*1e6:.1f} um)"
        )


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    from openroad import Design, Tech

    args = _parse_args()
    target_grid = args.grid

    tech = Tech()
    design = Design(tech)
    design.readDb(args.odb)
    block = design.getBlock()

    print(f"[thermal] Reading placement from {args.odb}")

    # Determine die size first — used for both power scaling and adaptive grid.
    die = block.getDieArea()
    dbu_per_um = block.getDbUnitsPerMicron()
    die_w_m = _dbu_to_m(die.xMax() - die.xMin(), dbu_per_um)
    die_h_m = _dbu_to_m(die.yMax() - die.yMin(), dbu_per_um)

    # Power scales with die area so power *density* is constant across designs.
    die_area_mm2 = die_w_m * die_h_m * 1e6
    total_power_w = args.power_density * die_area_mm2

    hs_grid = _adaptive_hotspot_grid(die_w_m, die_h_m, target_grid)

    if hs_grid < target_grid:
        print(
            f"[thermal] Die {die_w_m*1e3:.2f}×{die_h_m*1e3:.2f} mm is small — "
            f"using {hs_grid}×{hs_grid} HotSpot grid (min cell ≥{MIN_CELL_UM} µm), "
            f"upsampling to {target_grid}×{target_grid} for output"
        )
    else:
        print(
            f"[thermal] Die: {die_w_m*1e3:.2f}×{die_h_m*1e3:.2f} mm  "
            f"({die_area_mm2:.4f} mm²)  Grid: {hs_grid}×{hs_grid}"
        )

    power_grid, die_bounds_m = build_power_grid(block, hs_grid, total_power_w)
    print(f"[thermal] Total power: {power_grid.sum()*1e3:.1f} mW")

    package_args = hotspot_package_args(die_w_m, die_h_m, total_power_w)
    print(
        "[thermal] Package model: "
        + "  ".join(
            f"{k}={v}"
            for k, v in zip(package_args[0::2], package_args[1::2])
        )
    )

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        flp_path, ptrace_path = write_hotspot_inputs(
            power_grid, die_bounds_m, hs_grid, work_dir
        )
        print(
            f"[thermal] Running HotSpot ({hs_grid}×{hs_grid} = {hs_grid**2} blocks)..."
        )
        steady_path = run_hotspot(flp_path, ptrace_path, work_dir, package_args)
        temp_map_hs = parse_steady(steady_path, hs_grid)

    # Upsample HotSpot output to target_grid if a coarser grid was used.
    if hs_grid < target_grid:
        from scipy.ndimage import zoom

        scale = target_grid / hs_grid
        temp_map = zoom(temp_map_hs, scale, order=1).astype(np.float32)
        # Also upsample power_grid so saved arrays are always (target_grid, target_grid)
        power_grid_out = zoom(power_grid.astype(np.float32), scale, order=1)
    else:
        temp_map = temp_map_hs
        power_grid_out = power_grid.astype(np.float32)

    print(
        f"[thermal] Temperature: min={temp_map.min():.1f}°C  "
        f"max={temp_map.max():.1f}°C  "
        f"peak-to-peak={temp_map.max()-temp_map.min():.1f}°C"
    )

    _check_nondegenerate(temp_map, power_grid_out, hs_grid, die_w_m, die_h_m)

    np.savez(
        args.out,
        thermal_map=temp_map,
        power_grid=power_grid_out,
    )
    print(f"[thermal] Saved → {args.out}")


if __name__ == "__main__":
    main()
