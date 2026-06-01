# ML Integration Setup Guide

This document covers setting up and using the two ML features in `ml/`:

1. **Congestion Prediction** — CNN that predicts routing congestion from post-placement data
2. **Automated Floorplan Generation** — GNN that suggests macro placement from netlist topology

---

## Prerequisites

- ORFS Docker environment working (see `LINUX_SETUP.md`)
- Python 3.10+ on the host
- (Optional) CUDA-capable GPU — CPU training works for the current dataset size

---

## Install Python Dependencies

```
pip install -r ml/requirements.txt
```

For the floorplan GNN, also install PyTorch Geometric:

```
pip install torch-geometric
```

---

## Part 1 — Congestion Prediction

### Step 1: Generate Training Data

Run all built-in designs through placement and global routing, then extract the numpy arrays:

```
bash ml/congestion/data_collection/batch_run.sh
```

This runs from the repo root. For each design it:
1. Runs the ORFS flow through `3_5_place_dp` (detailed placement) and `5_1_grt` (global routing) inside Docker
2. Extracts a 64×64 cell-density grid (placement input)
3. Extracts per-layer routing usage fractions (congestion label)

Outputs land in `ml/data/` as `<platform>_<design>_placement.npy` and `<platform>_<design>_congestion.npy`.

To run a single design manually:

```
# Extract placement grid (inside Docker)
cd flow
util/docker_shell bash -c \
  "openroad -python /work/ml/congestion/data_collection/extract_placement.py \
    --odb /work/results/nangate45/gcd/base/3_5_place_dp.odb \
    --out /work/ml/data/nangate45_gcd_placement.npy \
    --grid 64"

# Extract congestion vector (on host)
python3 ml/congestion/data_collection/extract_congestion.py \
  --log flow/logs/nangate45/gcd/base/5_1_grt.log \
  --out ml/data/nangate45_gcd_congestion.npy
```

### Step 2: Train the Model

```
python3 ml/congestion/model/train.py \
  --data ml/data \
  --epochs 100 \
  --batch 8 \
  --out ml/congestion/model/checkpoints
```

The best checkpoint is saved to `ml/congestion/model/checkpoints/best.pt`.

### Step 3: Run Inference

```
python3 ml/congestion/model/predict.py \
  --placement ml/data/nangate45_gcd_placement.npy \
  --checkpoint ml/congestion/model/checkpoints/best.pt \
  --out ml/data/nangate45_gcd_predicted_congestion.npy \
  --image ml/data/nangate45_gcd_predicted_congestion.png
```

### Step 4: Enable the ORFS Flow Hook (Optional)

To automatically predict congestion after detailed placement during a flow run, add to your design's `config.mk`:

```makefile
export POST_DETAIL_PLACE_TCL = $(FLOW_HOME)/../ml/congestion/integration/orfs_hook.tcl
```

The hook runs after `3_5_place_dp` and prints a warning if predicted max congestion exceeds 50%.

---

## Part 2 — Automated Floorplan Generation

### Step 1: Generate Training Data

```
bash ml/floorplan/data_collection/batch_run.sh
```

This runs designs with hard macros (ariane133, black_parrot, mempool_group, microwatt, etc.) through synthesis and floorplan, then extracts:
- `<tag>_graph.npz` — netlist graph (node features + edge connectivity)
- `<tag>_floorplan.npz` — normalised macro placement coordinates

### Step 2: Train the Model

```
python3 ml/floorplan/model/train.py \
  --data ml/data \
  --epochs 200 \
  --out ml/floorplan/model/checkpoints
```

### Step 3: Suggest Placement for a New Design

First extract the netlist graph for your design (inside Docker):

```
cd flow
util/docker_shell bash -c \
  "openroad -python /work/ml/floorplan/data_collection/extract_netlist_graph.py \
    --odb /work/results/nangate45/mydesign/base/1_synth.odb \
    --out /work/ml/data/nangate45_mydesign_graph.npz"
```

Then run the suggestion script on the host:

```
python3 ml/floorplan/integration/suggest_floorplan.py \
  --graph ml/data/nangate45_mydesign_graph.npz \
  --checkpoint ml/floorplan/model/checkpoints/best.pt \
  --die-w 2000 --die-h 2000 \
  --out flow/designs/nangate45/mydesign/suggested_macros.tcl
```

Review and edit `suggested_macros.tcl`, then source it from your floorplan Tcl script:

```tcl
source flow/designs/nangate45/mydesign/suggested_macros.tcl
```

---

## File Structure

```
ml/
  requirements.txt
  data/                         # generated training data (.npy, .npz)
  congestion/
    data_collection/
      extract_placement.py      # ODB -> placement grid (run via openroad -python)
      extract_congestion.py     # GRT log -> congestion vector (run on host)
      batch_run.sh              # generate all congestion training data
    model/
      unet.py                   # U-Net architecture
      train.py                  # training loop
      predict.py                # inference script
      checkpoints/              # saved model weights
    integration/
      orfs_hook.tcl             # ORFS Tcl hook (after detailed placement)
      predict_congestion.py     # called by the hook via openroad -python
  floorplan/
    data_collection/
      extract_netlist_graph.py  # ODB -> netlist graph (run via openroad -python)
      extract_floorplan.py      # ODB -> macro positions (run via openroad -python)
      batch_run.sh              # generate all floorplan training data
    model/
      gnn.py                    # GNN architecture (requires torch-geometric)
      train.py                  # training loop
      checkpoints/              # saved model weights
    integration/
      suggest_floorplan.py      # inference -> Tcl placement script
```
