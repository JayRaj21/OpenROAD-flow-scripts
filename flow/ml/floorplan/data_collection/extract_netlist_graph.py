"""
Extract a netlist graph from a post-synthesis ODB file.

Run inside the ORFS Docker container via:
    openroad -python extract_netlist_graph.py \
        --odb /work/results/nangate45/ariane133/base/1_synth.odb \
        --out /work/ml/data/nangate45_ariane133_graph.npz

Saves a .npz file containing:
    node_features  -- (N, F) float32: [area, is_macro, is_seq, is_buf, fanin, fanout]
    edge_index     -- (2, E) int64:   [src, dst] pairs (directed, net-expanded)
    edge_weight    -- (E,) float32:   1.0 / fanout for each net edge
    node_names     -- (N,) str:       instance names
    num_macros     -- int:            number of macro instances
"""

import argparse
import numpy as np
import sys
import os

try:
    import openroad
    from openroad import Tech, Design
    HAS_OPENROAD = True
except ImportError:
    HAS_OPENROAD = False


MACRO_CELL_KEYWORDS = {"RAM", "ROM", "MEM", "SRAM", "FIFO", "REG_FILE", "fakeram"}
SEQ_SUFFIXES = {"DFF", "DFF_X", "DFF_P", "DFFR", "SDFF", "SDFFR"}
BUF_PREFIXES = {"BUF", "CLKBUF", "DELBUF"}


def _is_macro(inst) -> bool:
    master = inst.getMaster()
    return master.getType().lower() in ("block", "pad") or any(
        kw in master.getName() for kw in MACRO_CELL_KEYWORDS
    )


def _is_sequential(inst) -> bool:
    name = inst.getMaster().getName().upper()
    return any(name.startswith(s) for s in SEQ_SUFFIXES) or "DFF" in name


def _is_buffer(inst) -> bool:
    name = inst.getMaster().getName().upper()
    return any(name.startswith(p) for p in BUF_PREFIXES)


def extract_graph(odb_path: str):
    if not HAS_OPENROAD:
        raise RuntimeError("Run via: openroad -python extract_netlist_graph.py ...")

    tech = Tech()
    design = Design(tech)
    design.readDb(odb_path)
    block = design.getBlock()

    insts = list(block.getInsts())
    name_to_idx = {inst.getName(): i for i, inst in enumerate(insts)}

    # Node features: [area_norm, is_macro, is_seq, is_buf, fanin, fanout]
    areas = []
    for inst in insts:
        bbox = inst.getBBox()
        areas.append((bbox.xMax() - bbox.xMin()) * (bbox.yMax() - bbox.yMin()))

    max_area = max(areas) if areas else 1.0
    node_feat = np.zeros((len(insts), 6), dtype=np.float32)
    for i, inst in enumerate(insts):
        node_feat[i, 0] = areas[i] / max_area
        node_feat[i, 1] = float(_is_macro(inst))
        node_feat[i, 2] = float(_is_sequential(inst))
        node_feat[i, 3] = float(_is_buffer(inst))

    # Edges: expand nets into pairwise driver -> sink connections
    src_list, dst_list, weight_list = [], [], []

    for net in block.getNets():
        if net.isSpecial():
            continue
        iterms = list(net.getITerms())
        drivers, sinks = [], []
        for iterm in iterms:
            inst = iterm.getInst()
            if inst is None:
                continue
            idx = name_to_idx.get(inst.getName())
            if idx is None:
                continue
            if iterm.getIoType().lower() == "output":
                drivers.append(idx)
            else:
                sinks.append(idx)

        fanout = len(sinks)
        w = 1.0 / max(fanout, 1)
        for d in drivers:
            for s in sinks:
                src_list.append(d)
                dst_list.append(s)
                weight_list.append(w)
        # Update fanout/fanin counts
        for d in drivers:
            node_feat[d, 5] += fanout
        for s in sinks:
            node_feat[s, 4] += len(drivers)

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_weight = np.array(weight_list, dtype=np.float32)
    node_names = np.array([inst.getName() for inst in insts])
    num_macros = int(node_feat[:, 1].sum())

    return node_feat, edge_index, edge_weight, node_names, num_macros


def main():
    parser = argparse.ArgumentParser(description="Extract netlist graph from synthesis ODB")
    parser.add_argument("--odb", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    node_feat, edge_index, edge_weight, node_names, num_macros = extract_graph(args.odb)
    np.savez(
        args.out,
        node_features=node_feat,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_names=node_names,
        num_macros=np.array([num_macros]),
    )
    print(f"Saved graph -> {args.out}")
    print(f"  Nodes: {len(node_feat)}  Edges: {edge_index.shape[1]}  Macros: {num_macros}")


if __name__ == "__main__":
    main()
