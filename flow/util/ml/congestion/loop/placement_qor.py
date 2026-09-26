"""
Placement quality-of-result summary from a placed ODB.

HPWL is computed here, rather than parsed from an ORFS metrics JSON, so the
key name is not a guess. It is the sum over signal nets of the half-perimeter
of the bounding box of the net's ITerm access points (ITerm.getAvgXY) and
BTerm pin boxes, in µm. Power and ground nets are excluded (they span the
die and would swamp the signal wirelength); nets with fewer than two
terminals contribute zero. num_insts and total_cell_area_um2 cover every
instance in the block (standard cells, taps, macros).

Run inside Docker:
  openroad -python placement_qor.py --odb <3_place.odb> --out <qor.json>
"""

import argparse
import json


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odb", required=True, help="Placed ODB (3_place.odb)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    return ap.parse_args()


def net_hpwl_dbu(net) -> int:
    xs, ys = [], []
    for iterm in net.getITerms():
        ok, x, y = iterm.getAvgXY()
        if not ok:
            raise RuntimeError(
                f"ITerm {iterm.getInst().getName()}/{iterm.getMTerm().getName()} "
                f"of net {net.getName()} has no pin geometry"
            )
        xs.append(x)
        ys.append(y)
    for bterm in net.getBTerms():
        box = bterm.getBBox()
        xs.append((box.xMin() + box.xMax()) // 2)
        ys.append((box.yMin() + box.yMax()) // 2)
    if len(xs) < 2:
        return 0
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def main():
    from openroad import Design, Tech

    args = _parse_args()

    tech = Tech()
    design = Design(tech)
    design.readDb(args.odb)
    block = design.getBlock()
    dbu = block.getDbUnitsPerMicron()

    hpwl_dbu = 0
    for net in block.getNets():
        if net.getSigType() in ("POWER", "GROUND"):
            continue
        hpwl_dbu += net_hpwl_dbu(net)

    cell_area_dbu2 = 0
    num_insts = 0
    for inst in block.getInsts():
        bbox = inst.getBBox()
        cell_area_dbu2 += (bbox.xMax() - bbox.xMin()) * (bbox.yMax() - bbox.yMin())
        num_insts += 1

    die = block.getDieArea()
    result = {
        "hpwl_um": hpwl_dbu / dbu,
        "num_insts": num_insts,
        "total_cell_area_um2": cell_area_dbu2 / (dbu * dbu),
        "die_w_um": (die.xMax() - die.xMin()) / dbu,
        "die_h_um": (die.yMax() - die.yMin()) / dbu,
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[qor] {result}")
    print(f"[qor] Saved → {args.out}")


if __name__ == "__main__":
    main()
