"""
Extract macro placement coordinates from a post-floorplan ODB file.

Run inside the ORFS Docker container via:
    openroad -python extract_floorplan.py \
        --odb /work/results/nangate45/ariane133/base/2_floorplan.odb \
        --out /work/ml/data/nangate45_ariane133_floorplan.npz

Saves a .npz file containing:
    macro_names     -- (M,) str:       macro instance names
    macro_positions -- (M, 4) float32: [x_norm, y_norm, w_norm, h_norm]
                       all values normalised to [0, 1] relative to die area
    die_area        -- (4,) float32:   [x_min, y_min, x_max, y_max] in dbu
"""

import argparse
import numpy as np

try:
    import openroad
    from openroad import Tech, Design

    HAS_OPENROAD = True
except ImportError:
    HAS_OPENROAD = False

MACRO_TYPES = {"block", "pad"}
MACRO_CELL_KEYWORDS = {"RAM", "ROM", "MEM", "SRAM", "FIFO", "REG_FILE", "fakeram"}


def _is_macro(inst) -> bool:
    master = inst.getMaster()
    return master.getType().lower() in MACRO_TYPES or any(
        kw in master.getName() for kw in MACRO_CELL_KEYWORDS
    )


def extract_floorplan(odb_path: str):
    if not HAS_OPENROAD:
        raise RuntimeError("Run via: openroad -python extract_floorplan.py ...")

    tech = Tech()
    design = Design(tech)
    design.readDb(odb_path)
    block = design.getBlock()

    die = block.getDieArea()
    die_x0, die_y0 = die.xMin(), die.yMin()
    die_w = die.xMax() - die_x0
    die_h = die.yMax() - die_y0

    macro_names, macro_positions = [], []

    for inst in block.getInsts():
        if not _is_macro(inst):
            continue
        bbox = inst.getBBox()
        x = (bbox.xMin() - die_x0) / die_w
        y = (bbox.yMin() - die_y0) / die_h
        w = (bbox.xMax() - bbox.xMin()) / die_w
        h = (bbox.yMax() - bbox.yMin()) / die_h
        macro_names.append(inst.getName())
        macro_positions.append([x, y, w, h])

    macro_positions = np.array(macro_positions, dtype=np.float32)
    if len(macro_positions) == 0:
        macro_positions = np.zeros((0, 4), dtype=np.float32)

    die_area = np.array(
        [die_x0, die_y0, die_x0 + die_w, die_y0 + die_h], dtype=np.float32
    )

    return np.array(macro_names), macro_positions, die_area


def main():
    parser = argparse.ArgumentParser(
        description="Extract macro placement from floorplan ODB"
    )
    parser.add_argument("--odb", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    names, positions, die_area = extract_floorplan(args.odb)
    np.savez(args.out, macro_names=names, macro_positions=positions, die_area=die_area)
    print(f"Saved floorplan -> {args.out}")
    print(f"  Macros: {len(names)}")
    for name, pos in zip(names, positions):
        print(
            f"    {name}: x={pos[0]:.3f} y={pos[1]:.3f} w={pos[2]:.3f} h={pos[3]:.3f}"
        )


if __name__ == "__main__":
    main()
