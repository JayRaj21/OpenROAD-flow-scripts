# OpenROAD-flow-scripts — ML Extensions & Web UI

This document covers the machine learning features and web-based GUI added on top of the standard [OpenROAD-flow-scripts (ORFS)](README.md) framework. Everything described here lives under `flow/ml/` and `flow/ui/` and runs on the host machine alongside the existing Docker-based ORFS flow.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Directory Structure](#directory-structure)
4. [Web UI](#web-ui)
5. [Congestion Prediction](#congestion-prediction)
6. [Floorplan Suggestion](#floorplan-suggestion)
7. [Layout Viewer](#layout-viewer)
8. [Training the Models](#training-the-models)
9. [Supported Designs](#supported-designs)

---

## Overview

Three features are layered on top of ORFS:

| Feature | What it does |
|---|---|
| **Web UI** | Browser-based interface replacing terminal commands; streams live logs, shows ML results, renders chip layouts |
| **Congestion Prediction** | U-Net CNN that predicts per-layer routing congestion from a post-placement cell density grid, before the expensive global router runs |
| **Floorplan Suggestion** | GraphSAGE GNN that reads a post-synthesis netlist graph and suggests macro placement coordinates for designs containing hard macros |

Inference for both ML features runs in-process on the host GPU (CUDA) using the cached, compiled models. Data extraction (reading ODB files) still runs inside the ORFS Docker container via `openroad -python`.

---

## Prerequisites

### Required
- Docker with the `openroad/orfs:latest` image pulled
- Python 3.10 or newer on the host
- PyTorch (with CUDA if a GPU is available)
- PyTorch Geometric (for the GNN floorplan model)
- Flask and other Python dependencies

### Install Python dependencies

```bash
pip3 install --break-system-packages -r flow/ml/requirements.txt
pip3 install --break-system-packages flask matplotlib
```

### GPU (optional but recommended)

If an NVIDIA GPU is present, CUDA is used automatically for inference. Install the matching CUDA-enabled PyTorch build first:

```bash
pip3 install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Directory Structure

```
flow/
  ml/
    requirements.txt
    data/                          # Generated training data and predictions (gitignored)
    congestion/
      data_collection/
        batch_run.sh               # Collect training data for 12 designs
        extract_placement.py       # ODB → placement density grid (.npy)
        extract_congestion.py      # GRT log → congestion heatmap (.npy)
      model/
        unet.py                    # U-Net architecture
        train.py                   # Training loop
        predict.py                 # Standalone inference script
        checkpoints/best.pt        # Trained weights (gitignored)
      integration/
        orfs_hook.tcl              # Optional Tcl hook after detailed placement
        predict_congestion.py      # Called by the hook
    floorplan/
      data_collection/
        batch_run.sh               # Collect training data for macro designs
        extract_netlist_graph.py   # ODB → netlist graph (.npz)
        extract_floorplan.py       # ODB → macro coordinates (.npz)
      model/
        gnn.py                     # GraphSAGE GNN architecture
        train.py                   # Training loop
        checkpoints/best.pt        # Trained weights (gitignored)
      integration/
        suggest_floorplan.py       # Standalone inference script
  ui/
    app.py                         # Flask server
    layout_parser.py               # DEF/LEF parser for the layout viewer
    odb_to_def.py                  # Converts ODB to DEF inside Docker
    templates/
      index.html                   # Single-page web UI
  run_ui.sh                        # Launch script
```

---

## Web UI

### Starting the UI

```bash
bash flow/run_ui.sh
```

Then open `http://localhost:5000` in a browser.

### Features

**Design selector** — lists every design found under `flow/designs/` that has a `config.mk`. Select a platform and design to get started.

**Run Flow** — runs the full ORFS flow (`make`) inside Docker and streams the output line-by-line in the log panel.

**Run DRC** — runs the design rule check (`make drc`) and streams output.

**Predict Congestion** — runs the U-Net model on the design's post-placement cell density grid. If the grid has not been extracted yet, it is extracted automatically from the ODB file first. Results are displayed as 10 per-layer heatmaps (metal1–metal10).

**Suggest Floorplan** — runs the GNN on the netlist graph to suggest macro placement. Only meaningful for designs with hard macro blocks (see [Supported Designs](#supported-designs)). The output is a Tcl script of `place_inst` commands written to `flow/ml/data/<platform>_<design>_suggested.tcl`.

**View Layout** — renders the chip layout interactively in the browser canvas. Supports pan, zoom, layer toggles, and stage selection (Synthesis through Final).

### Log Panel Controls

- **Run** buttons stream live subprocess output as Server-Sent Events.
- A green checkmark and `Done (exit 0)` indicates success; a red `Failed (exit N)` indicates an error.

---

## Congestion Prediction

### How it works

1. After detailed placement (`3_5_place_dp`), a 64×64 grid of normalized cell density values is extracted from the ODB file.
2. The U-Net encoder-decoder takes this grid as input and produces a 10-channel output — one predicted congestion map per routing layer.
3. Values range from 0 (no congestion) to 1 (fully congested). The "hot" colormap is used: black = low, white/yellow = critical.

### Reading the results

Each of the 10 charts corresponds to one metal layer:

- **metal1 / metal2 / metal3** — local routing layers; highest congestion is expected here.
- **metal4+** — longer wires, clock distribution; generally less congested for standard designs.
- **metal7–10** — mostly empty for small designs.

Bright hotspots in the same region across multiple layers indicate a placement density problem. The fix is to increase the die area, reduce the utilization target, or add a density constraint in the congested region.

### Running prediction manually

```bash
cd flow
# Extract placement grid (inside Docker):
util/docker_shell openroad -python /work/ml/congestion/data_collection/extract_placement.py \
    --odb /work/results/nangate45/gcd/base/3_5_place_dp.odb \
    --out /work/ml/data/nangate45_gcd_placement.npy \
    --grid 64

# Run prediction on host:
python3 ml/congestion/model/predict.py \
    --placement ml/data/nangate45_gcd_placement.npy \
    --checkpoint ml/congestion/model/checkpoints/best.pt \
    --out ml/data/nangate45_gcd_predicted_congestion.npy \
    --image ml/data/nangate45_gcd_predicted_congestion.png
```

---

## Floorplan Suggestion

### How it works

1. The post-synthesis ODB (`1_synth.odb`) is parsed into a netlist graph: nodes are cells with features (area, is_macro, is_sequential, is_buffer, fanin, fanout); edges are nets weighted by inverse fanout.
2. The GraphSAGE GNN reads this graph and outputs normalized (x, y) coordinates for each macro node.
3. The coordinates are scaled by the requested die dimensions and written to a Tcl file of `place_inst` commands.

### Important limitations

- This feature **only applies to designs that contain hard macro cells** (SRAM, ROM, large IP blocks). Standard-cell-only designs (adder4, gcd, aes, etc.) will report "no macros found."
- The model was trained on a small dataset (~6 designs). Treat suggestions as a starting point to refine, not a final placement.
- Always review the generated Tcl script before sourcing it in a floorplan script.

### Running suggestion manually

```bash
cd flow
# Extract netlist graph (inside Docker):
util/docker_shell openroad -python /work/ml/floorplan/data_collection/extract_netlist_graph.py \
    --odb /work/results/nangate45/ariane133/base/1_synth.odb \
    --out /work/ml/data/nangate45_ariane133_graph.npz

# Suggest placement on host:
python3 ml/floorplan/integration/suggest_floorplan.py \
    --graph ml/data/nangate45_ariane133_graph.npz \
    --checkpoint ml/floorplan/model/checkpoints/best.pt \
    --die-w 2000 --die-h 2000 \
    --out ml/data/nangate45_ariane133_suggested.tcl
```

---

## Layout Viewer

The layout viewer is an HTML5 Canvas renderer built into the web UI. It parses DEF files (converting from ODB via `openroad -python` when needed) and renders the chip interactively in the browser.

### Controls

| Input | Action |
|---|---|
| Scroll wheel | Zoom in/out |
| Click and drag | Pan |
| Double-click | Fit entire die to view |
| Layer checkboxes | Toggle individual routing layers |
| All / None buttons | Show or hide all layers at once |
| Stage dropdown | Switch between flow stages (Synthesis, Placement, Route, etc.) |

### What is rendered

| Element | When visible |
|---|---|
| Die boundary | Always |
| Power/ground routes (special nets) | Always |
| Standard cells | When zoom > ~0.02 pixels/DBU |
| Signal routing | When zoom > ~0.3 pixels/DBU |
| Pins | When zoom > ~0.1 pixels/DBU |
| Cell names | When zoom > ~20 pixels/DBU |

Macros are rendered in green; standard cells in dark blue. Each metal layer has a distinct color (metal1 = blue, metal2 = red, metal3 = green, etc.).

---

## Training the Models

### Congestion model

```bash
cd flow

# Step 1: collect training data (runs 12 designs through placement and global route)
bash ml/congestion/data_collection/batch_run.sh

# Step 2: train
python3 ml/congestion/model/train.py \
    --data-dir ml/data \
    --checkpoint-dir ml/congestion/model/checkpoints \
    --epochs 50
```

Training on CPU takes several hours. With a GPU, it completes in under 30 minutes. The best checkpoint (lowest validation loss) is saved as `checkpoints/best.pt`.

### Floorplan model

```bash
cd flow

# Step 1: collect training data (7 macro-containing designs)
bash ml/floorplan/data_collection/batch_run.sh

# Step 2: train
python3 ml/floorplan/model/train.py \
    --data-dir ml/data \
    --checkpoint-dir ml/floorplan/model/checkpoints \
    --epochs 100
```

---

## Supported Designs

### Congestion prediction

Works with any design that has been run through at least the detailed placement stage (`3_5_place_dp`). The placement grid is extracted automatically on first use.

### Floorplan suggestion

Only applies to designs with hard macro blocks. The following built-in designs qualify:

| Design | Platform | Macros |
|---|---|---|
| ariane133 | nangate45 | SRAM macros |
| ariane136 | nangate45 | SRAM macros |
| black_parrot | nangate45 | SRAM macros |
| bp_be_top | nangate45 | SRAM macros |
| bp_fe_top | nangate45 | SRAM macros |
| mempool_group | nangate45 | Many RAM macros |
| microwatt | sky130hd | SRAM macros |

All other built-in designs (adder4, gcd, aes, ibex, jpeg, etc.) are standard-cell-only and do not benefit from floorplan suggestion.
