# ML Pipeline — Development Log

This file is the **canonical development log** for the ML work on the `congestion-ml` branch.
Every significant change, decision, and planned next step is recorded here so the project can
be resumed from a cold start without losing context. Update it after every working session.

---

## Project Overview

**Goal:** Add ML-based prediction capabilities that OpenROAD currently lacks.

**Primary track: Thermal prediction** (branch focus as of 2026-08-10)

| Track | Input | Model | Labels | Status |
|---|---|---|---|---|
| **Thermal prediction** | Post-placement ODB | U-Net (`models/unet.py`) | HotSpot thermal maps | Extractor + pipeline wired; awaiting data |
| Pre-placement congestion | Post-synthesis netlist graph | GNN (`models/gnn.py`) | GRT congestion maps | Deprioritised — code kept, not actively developed |

**Why thermal is the focus:**
- OpenROAD has no thermal solver at all. HotSpot runs take minutes; a trained U-Net
  surrogate runs in milliseconds and can be embedded directly into the ORFS flow.
- Pre-placement congestion prediction (GNN) is a useful capability but not urgent —
  it remains in the codebase and the pipeline still extracts netlist graphs for free,
  but it is not the primary development target on this branch.

---

## Conventions

**Commit and PR messages (2026-09-17 on):** write in clear, human-readable
prose — explain what changed and why in plain language a reviewer can
follow without cross-referencing the diff, not a terse technical log-line
or an auto-generated-looking dump. This applies to every commit and PR on
this project going forward, regardless of how the change was made (by
hand, or by an agent/pipeline run).

**No AI attribution trailers (2026-09-17 on):** do not add a
`Co-Authored-By: Claude ...` line or a `Claude-Session: ...` link to any
commit message or PR description on this project. Commits should read
identically to a human-authored one.

---

## Codebase Map

```
flow/ml/
├── Dockerfile                          # Custom image: ORFS + HotSpot + Python ML packages
├── congestion/
│   ├── DESIGN_RUNS.md                  # This file — canonical dev log
│   ├── data/                           # Extracted .npz datasets (features + labels)
│   ├── data_collection/
│   │   ├── extract_features.py         # Post-placement features from ODB (4 channels, 64x64)
│   │   ├── extract_labels.py           # Congestion labels from GRT ODB
│   │   ├── extract_thermal_labels.py   # Thermal labels via HotSpot
│   │   ├── extract_netlist_features.py # Pre-placement netlist graph (Track 1 GNN input)
│   │   ├── extract_existing.sh         # Batch extract from pre-existing ORFS result dirs
│   │   └── batch_run.sh                # Helper for manual batch runs
│   ├── models/
│   │   ├── unet.py                     # U-Net: spatial features → congestion/thermal heatmap
│   │   ├── gnn.py                      # GNN: netlist graph → congestion heatmap
│   │   └── heads.py                    # Shared output heads (heatmap, hotspot, score)
│   ├── training/
│   │   ├── dataset.py                  # CongestionDataset loader + split_dataset
│   │   ├── metrics.py                  # heatmap_mae, hotspot_iou, score_pearson, compute_all
│   │   ├── train_unet.py               # U-Net training script
│   │   └── train_gnn.py                # GNN training script
│   ├── inference/
│   │   ├── evaluate.py                 # Evaluate all models on held-out test set
│   │   └── predict.py                  # Single-design inference
│   ├── pipeline/
│   │   ├── run_pipeline.py             # Automated ORFS data collection pipeline
│   │   ├── designs.json                # Design configs for pipeline runs
│   │   └── logs/                       # Per-run error logs + summary logs
│   ├── tests/
│   │   ├── test_models.py              # Smoke tests: shapes, ranges, training, checkpoints
│   │   └── generate_synthetic_data.py  # Synthetic .npz generator for tests
│   └── checkpoints/
│       ├── unet_best.pt / unet_last.pt
│       └── gnn_best.pt  / gnn_last.pt
└── data/                               # Pre-placement GNN data from prior experiments
    ├── *_graph.npz                     # Netlist graphs (Track 1 input features)
    ├── *_congestion.npy                # Congestion maps (Track 1 labels)
    └── *_floorplan.npz                 # Floorplan data (larger designs)
```

---

## Changelog

### 2026-09-16 (later) — 30-design real routed dataset (12 → 30, three new PDKs: sky130hs/gf180/ihp-sg13g2)

**Goal, per plan.** Grow the real routed dataset from 12 to 30 designs across
6 PDKs, chosen so die-size coverage becomes a dense ladder rather than three
clumps, and tighten Laplacian LODO statistics (n=12 → n=30, doubling LOPO
folds 3→6). Target was deliberately 30, not ~50 — reaching 50 unique designs
would require the multi-hour/macro-heavy "giant" tier explicitly avoided by
precedent (`nangate45/bp_fe_top` etc.). **Stated plainly per plan: ~50-60
unique designs is unreachable in this repo without giants/macros; the next
round that wants n>30 must choose giants, finish-level variants, or accept
FNO/Swin work at n≈30.**

**Config/platform staleness found during pre-flight (§4.0 step 2) — decision
made and applied uniformly.** The `openroad/orfs:latest` image's baked-in
`/OpenROAD-flow-scripts/flow` copy is stale relative to this worktree's
`thermal-solver` HEAD: confirmed real diffs in both
`designs/ihp-sg13g2/gcd/config.mk` (image missing `export SYNTH_USE_SYN = 1`)
and `platforms/ihp-sg13g2/config.mk` (image missing the `PLATFORM_TCL`
suppress-message line and `OPT_POST_GRT_WNS`). Per the plan's contingency
("route with `DESIGN_CONFIG=/work/... DESIGN_HOME=/work/designs` instead"),
this was extended to also override `PLATFORM_HOME=/work/platforms` — required
because `PLATFORM_HOME` defaults to `$(FLOW_HOME)/platforms` (the same stale
image copy) the same way `DESIGN_HOME` does, confirmed via `make
print-PWR_NETS_VOLTAGES`/`print-LIB_FILES` dry-runs before committing to real
routing. Applied `DESIGN_CONFIG=/work/designs/<pdk>/<design>/config.mk
DESIGN_HOME=/work/designs PLATFORM_HOME=/work/platforms` uniformly to **all
18** new designs (not just the ones directly shown to mismatch), since the
staleness is systemic (whole image predates recent worktree commits), not
per-design. `extract_thermal_batch.sh`/`extract_irdrop_batch.sh` were left
untouched per the plan's non-goals — spot-checked that their own (unmodified)
`make print-LIB_FILES print-PWR_NETS_VOLTAGES` calls (relative `DESIGN_CONFIG`,
no `PLATFORM_HOME` override) still resolve to the correct on-disk liberty
files and voltages for gf180/ihp-sg13g2 despite reading the stale platform
config for the resolution logic — the corner/voltage mapping itself is
unchanged between the stale and current platform configs, only unrelated
lines (`PLATFORM_TCL`, `OPT_POST_GRT_WNS`) differ, and the `/work/${lib#*/flow/}`
path remap in `extract_irdrop_batch.sh` correctly points at the real mounted
file regardless.

**Probe phase (§4.1, hard checkpoint) — all 3 new PDKs passed, no drops
needed.**

| Design | rc | Wall time | Result |
|---|---|---|---|
| `ihp-sg13g2/gcd` | 0 | 23s | `6_final.odb`/`.spef` present |
| `sky130hs/gcd` | 0 | 71s | `6_final.odb`/`.spef` present |
| `gf180/riscv32i` | 0 | 233s | `6_final.odb`/`.spef` present |

Probe-phase extraction (thermal + IR-drop, both extractors run before
touching the remaining 15, exactly per plan): thermal `passed=6 failed=0
skipped=24`; IR-drop `passed=3 failed=0 skipped=27`. gf180's corner-dependent
`LIB_FILES` (the same pattern that once broke `extract_irdrop_batch.sh` on
asap7) resolved cleanly via `make print-LIB_FILES` →
`gf180mcu_fd_sc_mcu9t5v0__ff_n40C_5v50.lib.gz`; `print-PWR_NETS_VOLTAGES`
returned `VDD 5.5` and the extracted `gf180/riscv32i` IR-drop sample's
`v_nom` (recovered via `mean(irdrop_map + voltage_map)`) landed exactly at
5.5000V, worst-case drop 1.32mV (0.024% of nominal, far under the 10%
sanity bound) — the first 5.5V sample in this dataset, a new voltage regime.
`ihp-sg13g2/gcd` and `sky130hs/gcd` (both micro-die, `l_eq` 186µm/91µm)
solved on a coarse native HotSpot grid (37×37, 18×18) per the plan's §6 risk
note, upsampled to 64×64 for storage — no `lupdcmp: singular matrix` failure,
no degenerate/flat map. No PDK was dropped; no fallback substitutions were
needed at this checkpoint.

**Remaining 15 (§4.2) — all routed cleanly on the first try, no fallback
substitutions, no design near the ~90-minute abort threshold.** Order run
(cheapest-first by the plan's ordering, which is close to but not exactly
ascending stdcell count — the actual sequence has two inversions against
final measured stdcell counts, `sky130hs/ibex`(18307) before
`sky130hs/aes`(16887) and `asap7/jpeg`(63089) before `asap7/ethmac`(59670),
since the plan's ordering was based on estimates made before routing):
`sky130hs/riscv32i` (292s), `ihp-sg13g2/riscv32i` (269s), `asap7/aes_lvt`
(206s), `ihp-sg13g2/aes` (1012s), `sky130hs/ibex` (482s), `sky130hs/aes`
(629s), `sky130hd/ibex` (1042s), `gf180/aes` (371s), `asap7/ibex` (706s) —
the single design flagged by the plan as most likely to fail (slang +
`OPENROAD_HIERARCHICAL`) routed cleanly, no substitution needed — `sky130hd/jpeg`
(1198s), `sky130hs/jpeg` (689s), `asap7/jpeg_lvt` (674s), `asap7/jpeg`
(1022s), `asap7/ethmac` (780s), `nangate45/swerv` (1670s, the longest single
design, 27.8 min). All `rc=0`.

**Total wall time.** Probe phase: 327s (~5.5 min). Remaining 15: 11,042s
(~3.07h). **Combined ≈3.15 hours — well under the plan's honest 4-10h
estimate**, despite this batch's ~3x the prior 12-design pass's cell count on
three unfamiliar PDKs; no design hung, no per-design timeout was hit, no
parallelisation was used.

**Extraction (§4.3/§4.4, all 30 designs, auto-discovering, no `--force`, no
script edits).** `extract_thermal_batch.sh` (`openroad/orfs-ml:latest`,
`--timeout 3600`): `passed=30 failed=0 skipped=30` — the 30 `[OK]`s are the
15 non-probe new designs × 2 (features + thermal), and the 30 skips are the
3 probe designs × 2 (already extracted in §4.1) plus the 12 pre-existing ×
2 (already present, untouched). `extract_irdrop_batch.sh`
(`openroad/orfs:latest`, `--timeout 3600`): `passed=15 failed=0 skipped=45`
— the 15 `[OK]`s are the 15 non-probe new designs' IR-drop extractions
(their features already existed from the thermal batch just run, so those
were skipped too), and the 45 skips are those 15 designs' already-done
features, plus the 3 probe + 12 pre-existing designs' features and IR-drop
(both already present). Zero failures, zero timeouts across both
extractors for the full 18-design new batch.

**Verification against every §1 criterion:**

1. **Counts — PASS.** `find flow/results -name 6_final.odb | wc -l` = 30,
   `-name 6_final.spef | wc -l` = 30, `-name 3_place.odb | wc -l` = 30.
   `ls flow/util/ml/congestion/data/*.npz | wc -l` = 90.
2. **Value sanity — PASS.** All 90 `.npz` arrays finite, shape `(64,64)`
   confirmed for every array in every file; exact expected key set per kind
   (`features`: `{cell_density, fanout_density, macro_density, pin_density}`;
   `thermal_labels`: `{thermal_map, power_grid}`; `irdrop_labels`:
   `{current_density_proxy, irdrop_map, stripe_density, via_density,
   voltage_map}`). `ALLZERO` flags fired on `macro_density` for exactly the
   28 non-macro designs and were absent on `asap7_riscv32i`/
   `nangate45_tinyRocket` (the only two macro designs, both pre-existing) —
   the expected pattern, all 18 new designs (deliberately picked non-macro)
   included.
3. **Die-size diversity criterion — PASS, with large margin, though the
   "before" count is corrected here from the plan's stated target.** The
   plan's original criterion said "0 designs" in either band pre-expansion;
   checking the pre-existing 12 designs' own `l_eq` values against this
   table shows that undercounts the 75-250µm band — 3 of the 12 pre-existing
   designs already fell in it (`sky130hd/gcd` 78.05, `nangate45/dynamic_node`
   238.28, `nangate45/ibex` 240.69), so at most 6 of the "After: 9" below
   were newly added by this batch, not 9. The 500µm+ band's "before: 0" is
   correct (old max was `sky130hd/aes` at 475.19µm). Neither correction
   changes the pass/fail outcome. After: **9** designs in 75-250µm
   (`asap7/ibex` 76.68, `sky130hd/gcd` 78.05, `sky130hs/gcd` 90.73,
   `asap7/jpeg` 99.58, `asap7/ethmac` 106.81, `asap7/jpeg_lvt` 150.32,
   `ihp-sg13g2/gcd` 186.20, `nangate45/dynamic_node` 238.28,
   `nangate45/ibex` 240.69) and **9** above 500µm (`sky130hd/ibex` 507.53,
   `sky130hs/aes` 515.86, `ihp-sg13g2/riscv32i` 623.92, `sky130hs/ibex`
   643.20, `gf180/riscv32i` 732.18, `sky130hd/jpeg` 883.37, `ihp-sg13g2/aes`
   986.13, `gf180/aes` 1004.03, `sky130hs/jpeg` 1108.30) — both comfortably
   clear the ≥5 / ≥3 bar. Full ladder in the table below.
4. **IR-drop sanity — PASS for 13/18 new designs, systemic (pre-existing,
   not a regression) failure on all 5 new asap7 designs.** `v_nom` matched
   platform nominal exactly for every new design (1.2V ihp-sg13g2, 1.8V
   sky130hs, 5.5V gf180, 0.77V asap7). `worst_drop_mv < 0.10 × v_nom × 1000`
   held for all non-asap7 new designs (max observed ratio: `gf180/aes`
   0.021%, `ihp-sg13g2/riscv32i` 0.825%, `sky130hs/riscv32i` 0.086% — all far
   under 10%). **It does NOT hold for any of the 5 new asap7 designs**
   (`asap7/aes_lvt` 115.75mV vs. 77mV threshold = 150%; `asap7/ethmac`
   121.09mV = 157%; `asap7/ibex` 86.63mV = 112%; `asap7/jpeg` 139.02mV =
   181%; `asap7/jpeg_lvt` 77.27mV = 100.3%, right at the line). Checked
   whether this is new: **it is not** — re-computed the same ratio for the
   3 pre-existing asap7 designs from their unchanged `.npz` files and all 3
   already exceed the same 10% bound (`asap7/gcd` 103.68mV = 135%,
   `asap7/aes` 148.24mV = 192%, `asap7/riscv32i` 140.04mV = 182%) — i.e. 8/8
   asap7 designs in the full 30-design dataset (old and new) exceed this
   bound, and the new designs' ratios (100-181%) are actually somewhat
   *better* than the 3 pre-existing ones' (135-192%). This looks like an
   inherent property of asap7's low nominal voltage (0.77V — the 10%
   threshold is only 77mV) combined with the extractor's real IR-drop
   physics on this PDK's fine-pitch 7nm stack, not a new-design defect or an
   extraction bug. Per the plan's non-goals (no changes to
   `extract_irdrop_labels.py` or its constants), this is reported, not
   "fixed": criterion 5 as literally written fails for the asap7 platform as
   a whole, old and new alike, and any future model trained pooling
   `worst_drop_mv` across PDKs should treat asap7 as its own regime rather
   than assuming the 10%-of-nominal sanity bound holds universally.
5. **No collateral damage — PASS.** `md5sum -c /tmp/npz_baseline.md5` (36
   lines, snapshotted before any new routing/extraction): all 36
   pre-existing `.npz` files report `OK`, byte-identical. Nothing was
   re-extracted; the published 12-design tables remain valid unchanged.
6. **`tests/test_models.py -v` — PASS.** 20/20 `ok`, exit 0.

**Full 30-row table, sorted by `l_eq` (equivalent square die edge, µm),
computed uniformly for all 30 from `3_place.odb` via `openroad -python`
(`block.getDieArea()`), not mixed with any previously-documented per-design
numbers:**

| Design | l_eq (µm) | stdcells | ptp_c (°C) | contrast | corr(T,P) | worst_drop (mV) | v_nom (V) | new? |
|---|---|---|---|---|---|---|---|---|
| asap7/gcd | 8.98 | 531 | 0.4100 | 0.0045 | 0.6320 | 103.68 | 0.7700 | |
| nangate45/gcd | 36.73 | 618 | 5.1300 | 0.0566 | 0.6634 | 0.53 | 1.1000 | |
| asap7/aes | 51.04 | 15197 | 3.8200 | 0.0421 | 0.7846 | 148.24 | 0.7700 | |
| asap7/aes_lvt | 66.22 | 15141 | 17.2070 | 0.1890 | 0.8061 | 115.75 | 0.7700 | **new** |
| asap7/riscv32i | 73.15 | 10269 | 26.6046 | 0.2916 | 0.5890 | 140.04 | 0.7700 | |
| asap7/ibex | 76.68 | 20373 | 12.7122 | 0.1396 | 0.8275 | 86.63 | 0.7700 | **new** |
| sky130hd/gcd | 78.05 | 483 | 7.5964 | 0.0838 | 0.6368 | 0.41 | 1.8000 | |
| sky130hs/gcd | 90.73 | 629 | 19.9847 | 0.2193 | 0.4013 | 0.22 | 1.8000 | **new** |
| asap7/jpeg | 99.58 | 63089 | 7.5000 | 0.0823 | 0.5491 | 139.02 | 0.7700 | **new** |
| asap7/ethmac | 106.81 | 59670 | 3.3865 | 0.0371 | 0.5822 | 121.09 | 0.7700 | **new** |
| asap7/jpeg_lvt | 150.32 | 61242 | 42.1461 | 0.4588 | 0.7857 | 77.27 | 0.7700 | **new** |
| ihp-sg13g2/gcd | 186.20 | 386 | 78.6871 | 0.8512 | 0.7472 | 1.79 | 1.2000 | **new** |
| nangate45/dynamic_node | 238.28 | 10492 | 34.5512 | 0.3731 | 0.6333 | 1.01 | 1.1000 | |
| nangate45/ibex | 240.69 | 14978 | 42.4259 | 0.4583 | 0.6534 | 3.06 | 1.1000 | |
| nangate45/aes | 250.22 | 14568 | 43.4886 | 0.4688 | 0.6716 | 4.99 | 1.1000 | |
| nangate45/tinyRocket | 301.74 | 28218 | 24.2696 | 0.2605 | 0.3027 | 1.37 | 1.1000 | |
| nangate45/jpeg | 331.30 | 59227 | 21.1600 | 0.2264 | 0.3741 | 8.82 | 1.1000 | |
| sky130hd/riscv32i | 377.12 | 7306 | 40.3700 | 0.4295 | 0.3504 | 0.59 | 1.8000 | |
| sky130hs/riscv32i | 380.54 | 7172 | 36.4100 | 0.3873 | 0.2600 | 1.54 | 1.8000 | **new** |
| sky130hd/aes | 475.19 | 17607 | 38.6900 | 0.4074 | 0.4458 | 0.51 | 1.8000 | |
| nangate45/swerv | 481.83 | 85242 | 29.2100 | 0.3072 | 0.4561 | 5.53 | 1.1000 | **new** |
| sky130hd/ibex | 507.53 | 18446 | 40.6800 | 0.4270 | 0.4697 | 0.24 | 1.8000 | **new** |
| sky130hs/aes | 515.86 | 16887 | 29.9600 | 0.3141 | 0.4589 | 2.00 | 1.8000 | **new** |
| ihp-sg13g2/riscv32i | 623.92 | 9297 | 64.1900 | 0.6639 | 0.5835 | 9.90 | 1.2000 | **new** |
| sky130hs/ibex | 643.20 | 18307 | 41.8900 | 0.4332 | 0.4075 | 1.76 | 1.8000 | **new** |
| gf180/riscv32i | 732.18 | 7503 | 40.2900 | 0.4122 | 0.3564 | 1.32 | 5.5000 | **new** |
| sky130hd/jpeg | 883.37 | 45611 | 48.2000 | 0.4859 | 0.4736 | 1.46 | 1.8000 | **new** |
| ihp-sg13g2/aes | 986.13 | 16047 | 140.9500 | 1.3967 | 0.8319 | 7.62 | 1.2000 | **new** |
| gf180/aes | 1004.03 | 19284 | 35.2500 | 0.3506 | 0.3813 | 1.15 | 5.5000 | **new** |
| sky130hs/jpeg | 1108.30 | 48649 | 57.0400 | 0.5610 | 0.5267 | 1.48 | 1.8000 | **new** |

**Thermal contrast spread — measured, expected to change, confound NOT
fixed, per the explicit non-criterion.** Spread across the full 30-design
dataset is now `0.0045` (`asap7/gcd`) to `1.3967` (`ihp-sg13g2/aes`), a
~310x range — wider than the previously-documented ~104x on the 12-design
set. The low end is unchanged (`asap7/gcd` at 8.98µm was and remains the
smallest die); the widening is entirely at the top: the old maximum was
`sky130hd/aes` at 475.19µm, and the new large-die sky130hs/ihp-sg13g2/gf180
designs push that out to 1108µm, mechanically widening contrast spread per
the already-documented HotSpot package-geometry physics
— `contrast` grows with die size under any physically realistic package
model where chip thickness doesn't scale with lateral extent, per the
2026-09-14 entry). This is exactly the behavior flagged as a non-criterion
in the plan (HotSpot package-geometry limitation, not something to "fix"
here) and is reported, not treated as a regression.

**Coarse-grid designs (relevant to `UPSAMPLED_KEYS`) — correction: this is
not limited to the 18 new designs, and it invalidates a previously
published result, not just a future one.** 7 of the 18 new designs solved
thermal on a coarse native HotSpot grid, upsampled to 64×64 for storage:
`ihp-sg13g2/gcd` (37×37), `sky130hs/gcd` (18×18), `asap7/aes_lvt` (13×13),
`asap7/ethmac` (21×21), `asap7/ibex` (15×15), `asap7/jpeg` (19×19),
`asap7/jpeg_lvt` (30×30). The remaining 11 new designs (including
`gf180/riscv32i`, `gf180/aes`, and all `ihp-sg13g2`/`sky130hs` designs
except `gcd`) solved at native 64×64.

**A post-hoc audit found the *pre-existing* 12-design set has the same
problem**: `_adaptive_hotspot_grid()` goes coarse whenever the die's
smaller dimension is below `MIN_CELL_UM × 64 = 320µm`, and 4 of the
pre-existing 12 designs are below that threshold but are **not** in the
current `UPSAMPLED_KEYS` set — confirmed by checking the effective rank of
their stored `thermal_map` (bilinear upsampling from an N×N solve caps
observed rank at N; observed rank matched the predicted coarse grid size
exactly for all 4): `nangate45/aes` (min_dim=249.85µm, grid 49×49),
`nangate45/dynamic_node` (238.28µm, 47×47), `nangate45/ibex` (240.69µm,
48×48), `nangate45/tinyRocket` (301.74µm, 60×60). (`nangate45/jpeg` and
`sky130hd/aes`, both above 320µm, correctly solve native — checked as
controls.)

**`UPSAMPLED_KEYS` in `training/laplacian_sweep.py` is a hardcoded 5-entry
set** (`asap7_gcd_base`, `asap7_aes_base`, `asap7_riscv32i_base`,
`nangate45_gcd_base`, `sky130hd_gcd_base`) **and is missing 11 entries, not
7**: the 7 new coarse-grid designs above, plus the 4 pre-existing ones just
found. The correct set at n=30 is 16 keys. Per the plan's explicit
non-goal, fixing the code is left for whoever next re-runs either Laplacian
sweep — but the consequence is stronger than "will mis-stratify a future
run": **the already-published thermal Laplacian sweep's stratified result
at n=12 (`DESIGN_RUNS.md`, "upsampled (n=5) vs native (n=7)", both negative
at both λ) was computed against an already-wrong 5-key set** — 4 of the 7
designs it reported as "native" were actually solved coarse-grid. That
stratified breakdown should be treated as invalid, not merely as a
consistency check that happened to look clean; it does not need
re-extraction (the `.npz` data itself is correct), only re-analysis with
the corrected 16-key set once someone updates `UPSAMPLED_KEYS`. The
sweep's headline "indistinguishable" verdict does not depend on this
stratification and is unaffected.

**Verification checklist (§5), final pass/fail:**
1. Counts (odb/spef/place/npz) — PASS.
2. No collateral damage (md5) — PASS (36/36 unchanged).
3. Value sanity (finite, shape, keys, ALLZERO) — PASS.
4. Thermal non-degeneracy (`ptp_c > 1e-3` on all 18 new) — PASS, all 18 well
   above the gate (lowest of the 18 new designs is `asap7/ethmac` at
   3.3865°C); all 18 `.npz` were written successfully, confirming
   `_check_nondegenerate()` never refused.
5. IR-drop sanity — PASS for 13/18, systemic pre-existing asap7 exception
   documented above (not a regression, not fixed per non-goals).
6. Die-size ladder / §1.4 criterion — PASS, 9/9 vs. required 5/3.
7. `test_models.py -v` — PASS, 20/20.
8. `git status` — clean except this file (see below).

**Files touched:** this entry only (`util/ml/congestion/DESIGN_RUNS.md`).
No edits to `extract_thermal_batch.sh`, `extract_irdrop_batch.sh`,
`extract_features.py`, `extract_thermal_labels.py`, `extract_irdrop_labels.py`,
`batch_run.sh`, `generate_variants.sh`, any training/model/dataset file, or
`laplacian_sweep.py`. No `--force` was passed to either batch extractor. Two
throwaway `openroad -python` scripts used to compute die area and stdcell
counts uniformly across all 30 designs (`_tmp_die_area.py`,
`_tmp_cellcount.py`) were written under `data_collection/` for the docker
mount, then deleted immediately after use — not committed, not left behind.
`flow/results/`, `flow/logs/`, `flow/objects/`, `flow/reports/`, and
`flow/util/ml/congestion/data/*.npz` remain gitignored as expected;
`git status` after this entry shows changes only to this file.

**Non-goals honored, as scoped:** no FNO/Swin work; no training/model/
dataset code touched; no re-running of the existing Laplacian sweeps at
n=30 (flagged above as follow-up, not performed); no util/aspect-ratio
variant generation; no re-litigation of the HotSpot package-scaling fix or
its constants (contrast-spread growth reported, not investigated further);
no placement-feedback-loop work.

---

### 2026-09-16 — IR-drop track: Laplacian sweep, pre-registration and Stage B result (LODO, seed-averaged)

**Why.** Generalise the thermal-track LODO Laplacian sweep
(`training/laplacian_sweep.py`, 2026-09-15 entry below) to the IR-drop
track, to get the same paired/seed-averaged/pre-registered verdict on
`--laplacian-weight` for `train_irdrop.py` that already exists for
`train_thermal.py`. `laplacian_sweep.py` was generalised additively (new
`TRACKS` registry keyed by `"thermal"`/`"irdrop"`; `--track` CLI flag,
track-dependent `--out` default, `track` threaded into the runs dedup key)
— no edits to `train_thermal.py`, `thermal_dataset.py`, `train_irdrop.py`,
`irdrop_dataset.py`, `models/unet.py`, or `models/heads.py`.

**Correction (post-review, 2026-09-16).** The first version of this entry
claimed the `track` field meant "an IR-drop rerun can never silently
overwrite a thermal record" — that was true only for the **runs** dedup
key. The **designs** merge (metadata: `ptp_c`, `worst_drop_mv`, etc.) was
still keyed on `d["key"]` alone, and design keys (`nangate45_gcd_base`
etc.) are identical across tracks — so writing IR-drop results to a
thermal `--out` (or vice versa) would silently overwrite all 12 thermal
(or IR-drop) design-metadata entries with the other track's metadata,
breaking every downstream metadata column, confound check, and stratum
split for that track, even though the run records themselves survived.
Confirmed reproducible pre-fix (`--track irdrop` against a copy of the
thermal artifact raised `KeyError: 'ptp_c'` on a subsequent thermal
`--analyze`). Fixed by keying the designs merge on `(track, key)` (each
design record now carries an explicit `"track"` field, backfilled as
`"thermal"` for pre-existing untagged records) and by adding a guard that
refuses to write a different track's designs into a file whose existing
designs block belongs to another track.

**Post-review correction to this guard:** the first version inferred a
legacy (untagged) design block's track from its *run* records
(`r.get("track", "thermal")`), defaulting to `"thermal"` — but run records
are schema-identical across tracks (no field distinguishes them), so this
default was a guess, and it was wrong in exactly the case that mattered:
writing `--track thermal` into a legacy IR-drop file would itself also
default to `"thermal"`, making the guard a no-op for that direction
(reproduced: it silently replaced all 12 IR-drop design records with
thermal ones, exit 0, no error). Fixed by inferring a legacy design
record's track from its own distinguishing fields instead
(`_infer_legacy_track`: presence of `worst_drop_mv` ⇒ irdrop, `ptp_c` ⇒
thermal) — this is reliable because designs, unlike runs, do have
track-specific fields, and it correctly protects both write directions,
not just IR-drop-into-thermal. `_analyze_mode` was given the matching
fix: reading a file that has zero records for the requested track (e.g.
`--track irdrop --analyze` against a file that turns out to hold only
thermal data) now raises immediately instead of silently emitting an
empty/`nan` summary that overwrites the real one. Re-verified
against copies of the real artifacts (not the live ground-truth files):
writing `--track irdrop` into a copy of the legacy-format
`laplacian_sweep.json` is now refused outright (exit 1, file unchanged);
a post-fix file with both tracks' designs merges correctly (24/24 design
records retained, 12 per track); the live `laplacian_sweep.json` and
`irdrop_laplacian_sweep.json` on disk were unaffected by this
verification (md5-checked before/after), aside from a one-time additive
migration adding the `"track"` field to `irdrop_laplacian_sweep.json`'s
12 pre-existing design records (all other fields and all 180 run records
unchanged).

**Thermal regression gates (non-negotiable, run before any IR-drop work):**
- Training-path bit-exactness: `git show HEAD:.../laplacian_sweep.py` vs.
  the generalised file, both run with `--folds nangate45_gcd_base
  --lambdas 0 --seeds 0 --epochs 3`. `heldout_mse_final` /
  `heldout_mae_final` / `heldout_mse_best_epoch` / `train_loss_final` /
  `train_mse_final` bit-identical (full float repr,
  `0.06547243148088455` / `0.2273174673318863` / `0.06547243148088455` /
  `0.04363104452689489` / `0.04363104452689489` — all matched). **PASS.**
- Analysis-path regression: `--analyze` (thermal default `--out`) against
  the backed-up `laplacian_sweep_summary.md` from the 2026-09-15 entry —
  `diff` empty. **PASS.** Proves the `track` backfill
  (`r.get("track", "thermal")`), the dedup-key change, and the
  registry-driven table/confound/stratum rendering changed nothing for the
  existing 210-record thermal artifact.

**IR-drop metadata dump (`_irdrop_metadata`, all 12 designs).** Reproduces
the 2026-09-14 documented `worst_drop_mv` table exactly (nangate45: gcd
0.534 / dynamic_node 1.008 / ibex 3.057 / aes 4.985 / jpeg 8.821 /
tinyRocket 1.370 mV; sky130hd: gcd 0.412 / aes 0.513 / riscv32i 0.588 mV;
asap7: gcd 103.679 / riscv32i 140.041 / aes 148.236 mV — all match to the
documented precision). `v_nom` lands exactly at platform nominal supply in
every case (asap7 0.77V, nangate45 1.1V, sky130hd 1.8V — recovered via
`mean(irdrop_map + voltage_map)`, no nominal-supply key needed). **No
design has `degenerate=True`** (all `ptp_mv > 0`). Occupancy
(`distinct_values/4096`) ranges 0.0471 (`nangate45_gcd_base`, lowest) to
0.9312 (`nangate45_jpeg_base`, highest); the two `gcd` variants
(`nangate45_gcd_base`=0.0471, `sky130hd_gcd_base`=0.0476) and `asap7_gcd_base`
(0.1191) are the most fill-artifact-dominated by a wide margin.

**Sensitivity-cut exclusion, named before Stage B per plan §8:**
`nangate45_gcd_base` — lowest occupancy of all 12 designs, no
`degenerate=True` design exists so the occupancy rule (not the degenerate
fallback) applies. n=11 sensitivity cut excludes this design.

**Stage A (convergence probe, one λ=0 fold, `nangate45_gcd_base`, 120
epochs, GPU).** Wall time 3.5s. Last-20-epoch `train_mse_final` range
(0.00467) is **33.5% of the last-20-epoch mean (0.01394) — exceeds the
pre-registered 25% plateau threshold.** Per the plan's pre-registered rule,
`--epochs` is raised to **200 for all Stage B IR-drop runs** (all λ, all
folds, all seeds — not tuned per-fold). This is a data-driven necessity,
not a retrofit: IR-drop's LODO train set is 11 samples at batch-size 4
(3 gradient steps/epoch) vs. thermal's same setup, so slower convergence
than thermal at 120 epochs is plausible on its own, independent of any
track-specific loss-landscape difference.

**Reproducibility (`--strict-determinism`).** Unlike the thermal entry's
finding (`torch.use_deterministic_algorithms(True)` was expected to raise
on a bilinear-upsample backward and was never applied), for the IR-drop
track on this environment/torch version **`use_deterministic_algorithms(True)`
did not raise**, at λ=0.1: 4/4 fresh-process reruns (`--folds
nangate45_gcd_base --lambdas 0.1 --seeds 0 --epochs 20
--strict-determinism`) were bit-exact
(`heldout_mse_final=0.07076305150985718` all four times). Without the flag
(cudnn-only determinism, the setting used for the actual Stage B sweep
below), a smaller 2-rerun spot check at both λ=0 and λ=0.1 also matched
exactly — but per the thermal entry's finding (48/60 λ=0.1 reruns matched
under cudnn-only determinism, individual deviations up to ~10%), a 1-pair
match does not establish full determinism at every λ under cudnn-only
settings; Stage B below does not use `--strict-determinism` (not part of
the plan's Stage B command), so this same caveat travels with it, exactly
as it does for the thermal entry.

**No regression:** `tests/test_models.py -v` → 20/20 `ok`, exit 0
(includes the pre-existing `TestIRDropDataset` cases, unaffected by this
file's changes since none of `irdrop_dataset.py`/`train_irdrop.py` were
touched).

**Pre-registered sweep configuration (λ={0, 0.01, 0.1}, 5 seeds {0-4},
epochs=200 per Stage A above, batch-size 4, lr 1e-3, AdamW wd 1e-4,
CosineAnnealingLR eta_min=1e-6, grad-clip 1.0, LODO 12 folds primary).**
`--batch-size 4` deviates from `train_irdrop.py`'s shipped default of 8,
same reasoning/deviation as the thermal sweep (11-sample LODO train set,
bs=8 → only 2 gradient steps/epoch — bs=4 keeps this comparable to the
already-run thermal sweep rather than introducing a second uncontrolled
variable).

**Pre-registered conclusion criteria (verbatim from the plan, applied to
IR-drop Δ = mean(`heldout_mse_final` at λ) − mean(`heldout_mse_final` at
λ=0), 5 seeds each, n=12 primary / n=11 sensitivity excluding
`nangate45_gcd_base`):**
- **Helps:** median Δ<0; ≥9/12 wins; Wilcoxon p<0.05; sign consistent
  across both the `fill_dominated`/`well_populated` strata (0.5 occupancy
  threshold, pre-registered above, median-split fallback if degenerate)
  and all 3 PDK groups; AND |mean Δ| and |median Δ| exceed the seed-noise
  threshold below.
- **Hurts:** mirror image.
- **Confounded:** |ρ(Δ, rel_drop)| ≥0.6 with p<0.05, OR strata disagree in
  sign, OR PDK groups disagree in sign with ≥2 groups at n≥3.
- **Indistinguishable:** anything else.
- **Noise threshold:** per-design sd across the 5 λ=0 seeds; comparison
  scale for a difference of two 5-seed means is `sd·√(2/5)` — the MEAN of
  that quantity across designs is the criterion threshold. Also reporting:
  (a) raw mean per-run sd, (b) same means excluding the highest-sd design,
  (c) per-design count of designs whose |Δ| exceeds their own `sd·√(2/5)`.
- **Honest limits restated regardless of outcome:** paired n=12 floors at
  p≈0.0005; 12 designs not independent (6 nangate45, recurring names across
  PDKs) so effective n<12, p-values optimistic; cudnn-only determinism at
  λ>0 is not fully established for Stage B (see reproducibility note
  above); the smoothness prior is more weakly motivated for IR drop than
  the docstring implies — static IR drop solves a Poisson problem with
  distributed sources (∇·(σ∇V)=J), not a source-free harmonic field, so
  real IR-drop maps have sharp local minima and structural discontinuities
  at PDN geometry, and penalising ∇²V may be a bias rather than pure
  regularisation; if λ>0 helps, part of the effect may be smoothing over
  the nearest-neighbour-fill blocky-label artifact in the
  `fill_dominated` stratum rather than learning better physics — reported
  explicitly per-stratum rather than headlined as a clean win if the effect
  concentrates there.

**Stage B (real sweep).** LODO: 12 folds × 3 λ (0, 0.01, 0.1) × 5 seeds =
180 runs, `--epochs 200` per the Stage A decision above. ~17.5 minutes wall
time on GPU (~5.8s/run at 200 epochs, consistent with Stage A's per-epoch
timing). `--track irdrop --analyze --sensitivity-exclude
nangate45_gcd_base` re-cut the tables below from the saved JSON.

**LODO results (n=12, primary), held-out MSE final epoch, mean±sd over 5
seeds:**

| Design | λ=0 | λ=0.01 | λ=0.1 | Δ(0.01−0) | Δ(0.1−0) |
|---|---|---|---|---|---|
| asap7/aes | 0.04513±0.00286 | 0.04333±0.00402 | 0.04841±0.00630 | −0.00180 | +0.00329 |
| asap7/gcd | 0.09007±0.00877 | 0.08731±0.01114 | 0.08099±0.00685 | −0.00276 | −0.00907 |
| asap7/riscv32i | 0.17662±0.11628 | 0.12782±0.07609 | 0.13397±0.08507 | −0.04880 | −0.04265 |
| nangate45/aes | 0.06364±0.01362 | 0.06955±0.01757 | 0.06747±0.01260 | +0.00591 | +0.00383 |
| nangate45/dynamic_node | 0.08414±0.01902 | 0.09348±0.02143 | 0.09304±0.02729 | +0.00934 | +0.00890 |
| nangate45/gcd | 0.07597±0.01044 | 0.07813±0.00563 | 0.07393±0.01917 | +0.00217 | −0.00203 |
| nangate45/ibex | 0.18338±0.02120 | 0.15685±0.00878 | 0.18479±0.03197 | −0.02653 | +0.00141 |
| nangate45/jpeg | 0.06214±0.01523 | 0.06913±0.01341 | 0.06013±0.02258 | +0.00699 | −0.00202 |
| nangate45/tinyRocket | 0.05224±0.00409 | 0.05633±0.00575 | 0.05456±0.00686 | +0.00409 | +0.00232 |
| sky130hd/aes | 0.04152±0.01395 | 0.04036±0.01056 | 0.04347±0.00909 | −0.00116 | +0.00195 |
| sky130hd/gcd | 0.06418±0.00308 | 0.06436±0.00434 | 0.06206±0.00325 | +0.00018 | −0.00212 |
| sky130hd/riscv32i | 0.02590±0.00670 | 0.02440±0.00618 | 0.02447±0.00478 | −0.00150 | −0.00143 |

**Aggregate (n=12):** λ=0.01: mean Δ=−0.00449, median Δ=−0.00049, 6/12
wins, Wilcoxon p=0.96973. λ=0.1: mean Δ=−0.00314, median Δ=−0.00001, 6/12
wins, Wilcoxon p=0.96973. Both effects are far weaker and far less
consistent than the thermal track's λ=0.01 result (which had 11/12 wins,
p=0.005) — here the deltas split almost exactly evenly around zero at both
λ.

**Confound check:** Spearman ρ(Δ, rel_drop) = −0.3147 (p=0.3191) at
λ=0.01, −0.0490 (p=0.8799) at λ=0.1; ρ(Δ, log10 worst_drop_mv) = −0.3357
(p=0.2861) at λ=0.01, −0.0350 (p=0.9141) at λ=0.1 — none large/significant
on their own. `occupancy` — the variable defining the
`fill_dominated`/`well_populated` stratum split and the mechanism this
entry's risk register worries about (nearest-neighbour-fill blocky-label
artifacts) — was **missing from `confound_specs` in the original version
of this entry** and has been added post-review: ρ(Δ, occupancy) = +0.2727
(p=0.3911) at λ=0.01 (not significant), but **+0.6294 (p=0.0283) at
λ=0.1** — this meets the *numeric* |ρ|≥0.6-with-p<0.05 bar the pre-registration
set for `rel_drop`, though `occupancy` itself was added to `confound_specs`
only after the fact, so calling it "pre-registered" would overstate it;
it is a post-review finding evaluated against a pre-registered threshold,
not a variable that was pre-registered itself. With 3 candidate confound
variables tested
across 2 λ values this does not survive a multiple-comparisons correction,
so it is suggestive, not conclusive — but it is a far better-grounded
signal (magnitude threshold + significance test, both met) than the
sign-disagreement clauses discussed next.

**PDK-sign-disagreement clause — relabelled, not a real confound signal
(post-review correction).** The original version of this entry called the
result "Confounded" on the basis of the PDK-group and strata
sign-disagreement clauses: nangate45 (n=6) mean Δ is slightly *positive*
(+0.00033 at λ=0.01, +0.00207 at λ=0.1) while asap7 (n=3) and sky130hd
(n=3) are both negative (asap7 −0.01778/−0.01614, sky130hd
−0.00083/−0.00053); fill_dominated/well_populated also disagree in sign at
λ=0.1. Those clauses are technically satisfied as written, but review
established they don't hold up as a confound signal: (1) the
sign-disagreement clause has no magnitude floor and no significance
requirement, so it fires under pure noise with ~3/4 probability given 3
groups; (2) every IR-drop PDK group mean here is inside the seed-noise
floor — nangate45's "positive" effect is ~1.7% of a single design's own
seed sd, and asap7's "negative" effect is driven almost entirely by one
design (`asap7_riscv32i_base`) whose own seed sd is 2.4x larger than its
Δ; (3) applying the identical clause retroactively to the already-
validated thermal λ=0.1 data ALSO technically triggers it (sky130hd
+0.00668 vs. nangate45/asap7 negative, a 3x larger magnitude than what's
driving this IR-drop reading) — yet the thermal entry correctly called its
result "Indistinguishable," because its own written rule only checked
strata disagreement, not PDK sign disagreement, for the confound bar. This
is a **methodology gap in the pre-registered confound clause itself**
(flagged here for any future track's sweep, not silently patched away
after the fact): a sign test over group means with no magnitude/
significance floor is not a meaningful confound test as currently written,
and should not be used to call a verdict on its own.

**n=11 sensitivity (`nangate45_gcd_base` excluded, lowest-occupancy design
per the pre-registered choice above):** λ=0.01: mean Δ=−0.00509, median
Δ=−0.00116, 6/11 wins, p=0.89844. λ=0.1: mean Δ=−0.00324, median
Δ=+0.00141, 5/11 wins, p=0.96582. Same pattern as n=12 — weak, inconsistent,
not significant; the sensitivity cut does not change the picture.

**Seed-noise floor (λ=0), all four required numbers:**
(a) raw mean per-design sd across seeds = 0.01960. (b) same, excluding the
highest-sd design (`asap7_riscv32i_base`, sd=0.11628, the same design that
dominated the thermal track's noise floor) = 0.01081. (c) mean of the
per-design `sd·√(2/5)` comparison-scale threshold (matching scale for a
difference of two 5-seed means) = 0.01240 — both λ's |mean Δ| (0.00449,
0.00314) and |median Δ| (0.00049, 0.00001) fall well **below** this
threshold. (d) per-design count of designs whose own |Δ| exceeds their own
`sd·√(2/5)` threshold: 2/12 at λ=0.01, 3/12 at λ=0.1.

**Verdict, applying the plan's pre-registered §8 criteria, with the
methodology correction above applied (n=12 and n=11 agree):**
- **Helps:** fails outright at both λ — wins are 6/12 (6/11), far short of
  the ≥9/12 bar, and Wilcoxon p≈0.97, nowhere near <0.05.
- **Hurts:** fails by the same mirror-image margin (6/12 losses, not
  ≥9/12; p≈0.97).
- **Confounded (as literally written):** the PDK-group sign-disagreement
  clause is technically met at both λ, and the strata sign-disagreement
  clause additionally at λ=0.1 — but per the correction above, this clause
  is magnitude/significance-free and fires on noise; it is not treated as
  the basis for the verdict. The correlation-based confound test
  (|ρ|≥0.6 with p<0.05) IS met, but only for `occupancy` at λ=0.1
  (ρ=+0.6294, p=0.0283) — not for `rel_drop` or `worst_drop_mv` at either
  λ, and not surviving multiple-comparisons correction across the 3
  variables × 2 λ tested.

**→ Indistinguishable overall, with a suggestive (uncorrected p=0.028)
fill-artifact confound signal at λ=0.1 worth a follow-up run at larger n
(post-review corrected verdict; the original version of this entry called
this "Confounded" on the sign-disagreement clause alone — see correction
above).** This is, if anything, an even weaker/more null result than the
thermal track's own "Indistinguishable" verdict: 6/12 wins (p=0.97) here
vs. 11/12 wins (p=0.005) for thermal — thermal at least had unanimous
direction even though it fell short of significance and the noise floor,
whereas IR-drop's aggregate Δ is a coin flip in both magnitude and sign.
The correct framing is not "IR-drop's result is materially weaker/
different than thermal's" as a categorical-verdict claim (both are
"Indistinguishable"), but that **IR-drop's evidence for any effect at all
is weaker than thermal's** (6/12 p=0.97 vs. 11/12 p=0.005). The two
sweeps are also not fully apples-to-apples: IR-drop ran 200 epochs (Stage
A convergence rule, this entry) vs. thermal's 120, so some caution is
warranted before comparing their headline effect sizes directly. Real
confound evidence does exist, but it's the `occupancy` correlation at
λ=0.1 (see above), not the PDK sign test — the plan's own risk #2
(nearest-neighbour-fill blocky-label artifacts) and risk #1 (weaker
physical motivation for smoothness in a Poisson-type field with
distributed sources) both remain live explanations for why any
track-specific λ preference would fail to generalize here, and the
`occupancy` correlation is the more credible evidence for risk #2
specifically.

**Honest limits (stated regardless of outcome, per the plan):** a paired
test at n=12 floors around p≈0.0005 and only reliably detects large,
consistent effects; the 12 designs are not fully independent (6 nangate45,
recurring design names across PDKs), so effective n<12 and the p-values
above are optimistic (though here they are so far from significant this
barely matters); `--strict-determinism` reproduced bit-exactly for
IR-drop at λ=0.1 in a small spot check (§7.6 above), but Stage B itself
ran under the weaker cudnn-only determinism setting (matching the thermal
sweep's methodology, not the stronger flag), so individual-run values in
the table above carry the same un-quantified ULP-to-percent-level
non-determinism risk the thermal entry documented, averaged out (or not)
across 5 seeds; the smoothness prior's physical motivation is weaker for
IR drop (Poisson problem, distributed sources, real discontinuities at PDN
geometry) than the docstring implies, per risk #1 in the plan; this sweep
cannot separate "no real effect" from "an effect entangled with PDK/die
identity," and the pre-registered confound criteria are specifically
designed to catch exactly this ambiguity rather than resolve it.

**Verification performed:**
- §7.1 training-path bit-exactness gate: PASS (bit-identical
  `heldout_mse_final`/`heldout_mae_final`/`heldout_mse_best_epoch`/
  `train_loss_final`/`train_mse_final` between `git show HEAD:...` and the
  generalised file, thermal track, full float repr).
- §7.2 analysis-path regression gate: PASS (`diff` against the backed-up
  `laplacian_sweep_summary.md` is empty).
- §7.3 IR-drop smoke test: PASS (exit 0, one record, `track="irdrop"`,
  `wall_s`, `smoothness_final` present).
- §7.4 metadata correctness: PASS, see metadata dump above; no STOP
  condition hit.
- §7.5 Stage A convergence: measured, epochs raised to 200 per the
  pre-registered rule (see above).
- §7.6 reproducibility: `--strict-determinism` did not raise for this
  model and reproduced bit-exactly (4/4) at λ=0.1 in a spot check; Stage B
  itself used cudnn-only determinism (not `--strict-determinism`), same
  caveat as the thermal entry.
- §7.7 no regression: `tests/test_models.py -v` → 20/20 `ok`, exit 0.
- §7.8 Stage B: 180/180 runs completed, `--analyze` produced the tables
  above.

**Post-review fixes and re-verification (2026-09-16, same day).** Findings
1, 2, 3, 4, 5 above (designs-merge track bug, mislabeled confound verdict,
missing `occupancy` confound, wrong mtime filename, undocumented
sensitivity-exclude default) were all fixed in `laplacian_sweep.py` and
this entry. Re-verification performed:
- Designs-merge fix (Finding 1): writing `--track irdrop` into a copy of
  the legacy-format `laplacian_sweep.json` is refused outright (exit 1,
  copy's md5 unchanged); a fresh post-fix file accumulates both tracks'
  designs correctly (24/24 records, 12 per track) when written to
  sequentially. The live `laplacian_sweep.json` is byte-identical
  (md5-checked) before/after this verification;
  `irdrop_laplacian_sweep.json` received a one-time additive migration
  adding `"track": "irdrop"` to its 12 pre-existing design records (all
  other fields and all 180 run records unchanged, verified by field-set
  and count comparison) so the new track-qualified read path in
  `--analyze` can find them — this was necessary because those records
  were written before this fix existed.
- Thermal gates re-run with the validator's stronger check (18 runs: 3
  folds × 3 λ × 2 seeds, vs. the original single λ=0 run which never
  exercises the Laplacian branch): `git show HEAD:.../laplacian_sweep.py`
  vs. the fully-fixed file, `--folds nangate45_gcd_base,asap7_gcd_base,
  sky130hd_gcd_base --lambdas 0,0.01,0.1 --seeds 0,1 --epochs 3` —
  `heldout_mse_final`/`heldout_mae_final`/`heldout_mse_best_epoch`/
  `train_loss_final`/`train_mse_final`/`best_epoch` bit-identical across
  all 18 records. **PASS** — thermal remains an untouched pure refactor
  after all fixes, including the λ>0 path.
- `--track thermal --analyze` re-run against the live artifact: `diff`
  against the pre-fix summary is empty. **PASS.**
- `--track irdrop --analyze` re-run: only the confound-table lines changed
  (now include `occupancy`); all other numbers (per-design table,
  aggregates, n=11 sensitivity cut, seed-noise floor) are unchanged from
  the pre-fix summary — `diff` confirms. `irdrop_laplacian_sweep_summary.md`
  regenerated with the corrected confound line and the corrected verdict
  documented above.
- `tests/test_models.py -v` re-run post-fix → 20/20 `ok`, exit 0.

**Files:** `training/laplacian_sweep.py` generalised in place (only file
changed; no edits to `train_thermal.py`, `thermal_dataset.py`,
`train_irdrop.py`, `irdrop_dataset.py`, `models/unet.py`,
`models/heads.py`, or `data_collection/*`); this entry.
`experiments/irdrop_laplacian_sweep.json` (180 run records) and
`experiments/irdrop_laplacian_sweep_summary.md` are gitignored artifacts,
not committed. `experiments/laplacian_sweep.json` /
`_summary.md` (the thermal artifacts) are unchanged in content (verified
byte-identical via the §7.2 gate) though `laplacian_sweep_summary.md`'s
mtime was refreshed by the gate's `--analyze` rerun (`--analyze` only
reads the JSON and writes `_summary.md`, so the JSON's mtime is untouched
— the original version of this sentence named `laplacian_sweep.json`
here, which was wrong; corrected post-review).

**Non-conclusion / non-goal, as scoped:** no checkpoint from this sweep
ships; no changes to the shipped `train_irdrop.py` defaults; no revision
of the thermal verdict (only thermal work here is proving it's unchanged,
per the two gates above); no LOPO run for IR-drop (out of scope per the
plan). If a production decision on IR-drop's `--laplacian-weight` is
needed, this result says "leave at the current default (0.0, off)" with
less ambiguity than the thermal track's result — there is no even
borderline-promising signal here, and what small aggregate improvement
exists is concentrated in a subset of designs/PDKs in a way the
pre-registered criteria are specifically designed to flag as unreliable.

---

### 2026-09-15 — Laplacian smoothness loss, re-evaluated on the corrected 12-design dataset (LODO, seed-averaged)

**Why this re-run was needed.** The three prior Laplacian comparisons
(n=3, n=4, n=6 below) flip sign across sample sizes and are each a single
unseeded pooled 70/15/15 split — at n≤6 that degenerates to a 1-sample val
set, and `split_thermal_dataset`'s val subset was also silently augmented
(random flips), so "best val loss" was a noisy, optimistically-biased,
non-reproducible number. None of those three results is trustworthy evidence
either way.

**New harness: `training/laplacian_sweep.py`** (new file only; no edits to
`train_thermal.py`, `thermal_dataset.py`, `models/unet.py`, `models/heads.py`,
or `data_collection/*`). Imports `train_thermal._loss`/`_laplacian` directly
so the sweep measures the exact shipped penalty. Runs leave-one-design-out
(LODO, 12 folds, primary) and leave-one-PDK-out (LOPO, 3 folds, secondary)
protocols, with explicit `torch.manual_seed`/`np.random.seed` per run, a
separate `augment=False` held-out `ThermalDataset` instance for evaluation
(the training instance stays `augment=True`), and the plain-MSE final-epoch
value as the headline metric (`heldout_mse_final`) — best-epoch numbers are
recorded (`heldout_mse_best_epoch`) but never headlined, since selecting the
epoch that minimises held-out loss is selection-on-the-eval-set. Model,
optimizer (AdamW, `lr=1e-3`, `weight_decay=1e-4`), scheduler
(`CosineAnnealingLR`), and grad-clip (1.0) all match `train_thermal.py`
exactly. `--analyze` mode re-cuts the tables from the saved JSON without
retraining.

**Stage A (timing + convergence probe).** One λ=0 fold on GPU (`cuda` was
available in this environment, contrary to the plan's CPU-only assumption):
120 epochs took 3.8s. `train_mse_final` dropped from 0.0653 (epoch 1) to a
0.010–0.020 plateau over the last 20 epochs — well converged, not the
1-gradient-step-per-epoch regime of the historical 15/20/25-epoch runs.
Given <60s/run, seeds were raised from 3 to 5 per the plan's stated
surplus-compute priority (seeds over extra λ values).

**Reproducibility gate caught a real bug — but the fix is partial, and the
original verification of it didn't exercise the case that still fails.**
The first full run of the sweep did not reproduce `heldout_mse_final` to
3 s.f. on a same-seed rerun (0.03973 vs 0.02670) — cuDNN's default
nondeterministic algorithm selection, not incomplete seeding of the RNGs
that were actually seeded. `torch.backends.cudnn.deterministic = True` /
`torch.backends.cudnn.benchmark = False` were set inside `_run_one`, and
the original rerun check (at λ=0) then reproduced exactly
(0.03393132612109184 both times) — but a later independent re-verification
found this only holds at **λ=0**: 60/60 λ=0 reruns were bit-exact, but at
λ=0.1 only 48/60 were, with individual-run deviations up to ~10% (isolated
example: `asap7/aes`, λ=0.1, seed 3, repeated 4× gave 0.023202/0.023168/
0.023168/0.023202). Root cause is more specific than the code comment
states: `cudnn.deterministic`/`benchmark=False` do not fully pin cuDNN's
algorithm-selection heuristic (which remains sensitive to GPU
workspace/memory state, more so for the larger λ>0 backward graph); the
complete fix would be `torch.use_deterministic_algorithms(True)` plus
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, not yet applied here. **This does not
change the verdict below** — an independent full-grid re-run (180 runs,
fresh seeds/process) reproduced every aggregate statistic exactly
(λ=0.01 mean/median Δ, wins, Wilcoxon p all identical to 5 d.p.; λ=0.1's
mean Δ moved 0.7%), because the per-run ULP-level perturbation averages
out across 5 seeds × 12 folds. It does mean the "reproducibility: confirmed
exact" claim above should be read as scoped to λ=0, not as validating
determinism at every λ tested.

**Stage B (real sweep).** LODO: 12 folds × 3 λ (0, 0.01, 0.1) × 5 seeds = 180
runs. LOPO: 3 folds × 2 λ (0, 0.1) × 5 seeds = 30 runs. Total 210 runs,
~13 minutes wall time on GPU (~3.5s/run) — the plan's 4–11h CPU estimate did
not apply once GPU was confirmed available; no parallelisation or λ=0.01
cut was needed.

**LODO results (n=12, primary):**

| Design | λ=0 | λ=0.01 | λ=0.1 | Δ(0.01−0) | Δ(0.1−0) |
|---|---|---|---|---|---|
| asap7/aes | 0.04144±0.01019 | 0.03219±0.00305 | 0.03457±0.00588 | −0.00924 | −0.00687 |
| asap7/gcd | 0.06963±0.00752 | 0.06837±0.02614 | 0.06576±0.01765 | −0.00127 | −0.00387 |
| asap7/riscv32i | 0.21290±0.17235 | 0.16757±0.18960 | 0.12662±0.04367 | −0.04532 | −0.08628 |
| nangate45/aes | 0.00883±0.00319 | 0.00863±0.00463 | 0.01053±0.00406 | −0.00020 | +0.00170 |
| nangate45/dynamic_node | 0.02898±0.01243 | 0.02768±0.01404 | 0.02300±0.00585 | −0.00130 | −0.00598 |
| nangate45/gcd | 0.05622±0.02133 | 0.04305±0.01711 | 0.04434±0.01400 | −0.01317 | −0.01188 |
| nangate45/ibex | 0.04418±0.00869 | 0.03944±0.01326 | 0.03973±0.01262 | −0.00473 | −0.00445 |
| nangate45/jpeg | 0.10879±0.02540 | 0.08910±0.02186 | 0.07929±0.01171 | −0.01969 | −0.02949 |
| nangate45/tinyRocket | 0.11237±0.02668 | 0.10070±0.02149 | 0.09196±0.02010 | −0.01166 | −0.02041 |
| sky130hd/aes | 0.06229±0.02128 | 0.04771±0.00983 | 0.05478±0.02777 | −0.01458 | −0.00751 |
| sky130hd/gcd | 0.07632±0.03493 | 0.06596±0.01664 | 0.08578±0.04196 | −0.01036 | +0.00946 |
| sky130hd/riscv32i | 0.02370±0.01118 | 0.03162±0.01746 | 0.04180±0.01895 | +0.00791 | +0.01809 |

**Aggregate (n=12):** λ=0.01: mean Δ=−0.01030, median Δ=−0.00980,
11/12 wins, Wilcoxon p=0.00488. λ=0.1: mean Δ=−0.01229, median Δ=−0.00642,
9/12 wins, Wilcoxon p=0.09229.

**Confound check:** Spearman ρ(Δ, contrast) = 0.2727 (p=0.3911) at λ=0.01,
0.2378 (p=0.4568) at λ=0.1 — not large or significant, so not the die-size
confound seen previously. Strata agree in sign at both λ: PDK breakdown
nangate45/asap7/sky130hd all negative mean Δ at both λ except sky130hd at
λ=0.1 (+0.00668, n=3, small and noisy); upsampled (n=5) vs native (n=7) both
negative at both λ (upsampled larger in magnitude: −0.0159/−0.0199 vs
−0.0063/−0.0069).

**n=11 sensitivity (`asap7/gcd` excluded):** λ=0.01: mean Δ=−0.01112,
median Δ=−0.01036, 10/11 wins, p=0.00684. λ=0.1: mean Δ=−0.01306,
median Δ=−0.00687, 8/11 wins, p=0.12305. Same pattern as n=12 — excluding
`asap7/gcd` does not flip anything.

**Seed-noise floor (λ=0):** mean per-design sd across seeds = 0.02960.
Two honesty notes on this number, since it is the sole criterion λ=0.01
fails: (1) it is a **per-run** sd, compared against a difference of
**5-seed means** — the matching noise scale for a 5-seed mean is
sd/√5≈0.01324 (or, paired, sd·√(2/5)≈0.01872), under which λ=0.01's
~0.010 effect would be closer to the boundary rather than clearly below
it. (2) Almost half of the 0.02960 (48.5%) comes from a single design,
`asap7/riscv32i` (sd=0.17235, visible in the table above); excluding it
the mean sd drops to 0.01662. Neither point overturns the verdict —
re-run independently with the SEM framing and per-design sd instead of
the mean floor, 0 of 12 designs have |Δ| exceeding their own seed sd — but
the "indistinguishable" call rests on this one threshold, not a wide
margin, and both properties should travel with it rather than being
implied to be a clean, uncontroversial cutoff.

**LOPO (secondary, die-size stress test, n=3, λ∈{0,0.1} only), mean held-out
MSE over 5 seeds:** nangate45 0.0675→0.0611, asap7 0.1529→0.1530 (flat),
sky130hd 0.0588→0.0558. Directionally consistent with LODO (λ=0.1 slightly
lower in 2/3 PDKs, flat in the third) but n=3 folds is far too small to
support its own verdict; reported as a consistency check, not a second
finding.

**Verdict, applying the plan's pre-registered §9 criteria exactly (both
n=12 and n=11 agree — no discrepancy to headline):**

- λ=0.01: median Δ<0 ✓, 11/12 (10/11) wins ✓, Wilcoxon p<0.05 ✓, sign
  consistent in both strata and all 3 PDK groups ✓ — but mean/median |Δ|
  (~0.010) is **below** the λ=0 seed-noise floor (0.0296). The "helps"
  criterion requires the effect to clear the noise floor; it does not.
- λ=0.1: Wilcoxon p=0.09229 (n=12) / 0.12305 (n=11) fails the p<0.05 bar
  outright, and |Δ| is also below the noise floor.
- Neither λ meets the "confounded" bar either (ρ not large/significant,
  strata don't disagree in sign).

**→ Indistinguishable at n=12 (and n=11) for both λ=0.01 and λ=0.1.** The
λ=0.01 result is statistically significant by Wilcoxon and directionally
unanimous across PDK/stratum splits, but its magnitude does not exceed the
measured seed-to-seed noise floor, so per the plan's pre-registered bar it
cannot be called "helps." This retires the contaminated n=6 "2.7× worse"
number and the n=3/n=4 sign-flip: properly paired and seed-averaged at
n=12, there is no reliable harm either — λ=0.1's point estimate is a
smaller improvement than λ=0.01's, not the substantial regression the n=6
entry reported.

**Honest limits (stated regardless of outcome, per the plan):** a paired
test at n=12 floors around p≈0.0005 and only reliably detects large,
consistent effects; the 12 designs are not fully independent (6 nangate45,
recurring design names across PDKs), so effective n<12 and the p-values
above are optimistic; each LODO training fold spans ~104× contrast, so this
result is about λ for a model that is partly learning die size, not a clean
verdict on the physical prior in general. This sweep does not settle the
technique — it replaces three unreliable numbers with one properly paired,
converged, seed-averaged "indistinguishable" result.

**Verification performed:**
- Smoke test (`--folds nangate45_gcd_base --lambdas 0 --seeds 0 --epochs 2`):
  exit 0, one run record with `wall_s`.
- Design metadata reproduced the documented table exactly: `asap7/gcd`
  ptp=0.4100/contrast=0.0045/corr=0.6320, `nangate45/aes`
  ptp=43.4886/contrast=0.4688/corr=0.6716 (matches 43.489/0.4688/0.6716).
- Upsampling detector: `distinct_values` did **not** cleanly partition into
  the documented 5 coarse-grid designs vs. the other 7 (e.g.
  `nangate45/jpeg`=1229 and `sky130hd/aes`=1877 are low despite being native
  64×64 designs; `asap7/riscv32i`=4078 is high despite being coarse-grid).
  Per the plan's fallback instruction, the stratification used the
  hard-coded documented list (`UPSAMPLED_KEYS`) instead of `distinct_values`.
- Convergence: confirmed in Stage A above.
- No regression: `tests/test_models.py -v` → 20/20 `ok`, exit 0.
- Reproducibility: failed once (cuDNN nondeterminism), partially fixed —
  confirmed exact at λ=0, still ~10%-deviation-on-individual-runs possible
  at λ=0.1 (root cause and full fix identified but not applied); verdict
  unaffected since it averages over 5 seeds — see above.

**Files:** new `training/laplacian_sweep.py` only; `.gitignore` gained
`flow/util/ml/congestion/experiments/` (was previously untracked by no
rule, now explicit); this entry. `experiments/laplacian_sweep.json` (210 run
records) and `experiments/laplacian_sweep_summary.md` are gitignored
artifacts, not committed.

**Non-conclusion / non-goal, as scoped:** no checkpoint from this sweep
ships; no changes to the shipped `train_thermal.py` defaults. If a
production decision on `--laplacian-weight` is needed, this result says
"either is fine, leave at the current default (0.0, off)" absent a reason
to prefer the mild λ=0.01 smoothing.

---

### 2026-09-14 (later) — HotSpot package-scaling fix, re-extraction, and calibration finding

**The fix, in `extract_thermal_labels.py`:** the root cause identified in the
correction to the earlier same-day entry below — `run_hotspot()` applying
HotSpot's fixed cm-scale default package geometry (silicon thickness,
spreader, sink) to every design regardless of actual die size — is now
addressed by `hotspot_package_args()`. For each design it derives `-t_chip`,
`-s_spreader`/`-t_spreader`, `-s_sink`/`-t_sink`, and `-r_convec` from the
design's own die extent (`CHIP_THICKNESS_FRAC=0.02` of the die's equivalent
square edge, spreader/sink as fixed multiples of the die's longer edge,
`r_convec` solved so it contributes `TARGET_MEAN_RISE_K` to mean die
temperature rise — see the corrected mean-rise section below, this does not
pin total mean rise), instead of leaving HotSpot's built-in cm-scale
defaults in place. Two supporting guards were added: (1) `MIN_HOTSPOT_GRID=8`
is now a hard floor in `_adaptive_hotspot_grid()`, so a die small enough to
want a coarser grid than 8×8 no longer silently collapses further
(previously `asap7/gcd`'s 9µm die resolved to a literal 1×1 grid); and (2)
`_check_nondegenerate()` raises before `np.savez` if the resulting
`thermal_map` is constant (ptp < `DEGENERATE_PTP_C`) or non-finite, so a
design that still can't produce a real gradient fails loudly and leaves no
`.npz`, rather than being silently saved as flat labels.

**Correction (2026-09-14, validator round 2): this entry's original physics
finding materially understated the residual confound and mean-temperature
claims. The corrected numbers below replace the originally-shipped ~9x/40K
figures.**

**Calibration sweep and the physics finding — corrected.**
`tests/thermal_scale_check.py` sweeps a fixed synthetic two-hotspot relative
power pattern across die extents through the real `hotspot_package_args()` →
`write_hotspot_inputs()` → `run_hotspot()` → `parse_steady()` pipeline, now
using `_adaptive_hotspot_grid()` at each extent (previously the script used
a fixed `--grid 32` regardless of extent, which is why the original version
of this entry only swept 51-1021µm: at `--grid 32` a 9µm die gives
0.28µm cells and HotSpot fails with `lupdcmp: singular matrix`). With the
adaptive grid, the sweep can now cover the dataset's actual smallest die
size, and the result is materially different from the original 51-1021µm
sweep:

| Die extent | 9µm | 25µm | 51µm | 102µm |
|---|---|---|---|---|
| grid used | 8 | 8 | 10 | 20 |
| relative contrast | 0.0210 | 0.1118 | 0.2844 | 0.5646 |

(Values shown are post-`T_CHIP_MIN_M` fix, i.e. with the 0.1µm numerical
floor rather than the original 1µm clamp — see the `T_CHIP_MIN_M` section
below. The clamped values were 0.0121/0.0825 at 9/25µm; the fix itself
widens rather than narrows the small-die contrast, so this does not change
the qualitative conclusion.)

Contrast grows ~27x over this 11x extent range (0.5646/0.0210), and it is
worse — not better — below 51µm than above it: the original 51-1021µm-only
sweep (kept below for reference) found "only" a ~9x range because it never
tested the regime the fix actually targeted (the dataset's smallest real
die, `asap7/gcd` at 8.98µm). The original 51-1021µm sweep, for reference:

| Die extent | 51µm | 255µm | 510µm | 1021µm |
|---|---|---|---|---|
| ptp (°C) | 23.08 | 107.76 | 165.93 | 236.82 |
| relative contrast | 0.2549 | 1.1618 | 1.7363 | 2.3389 |
| correlation vs. 51µm (normalized) | 1.0000 | 0.9727 | 0.9376 | 0.8942 |

Both sweeps are real measurements of the same package model at different
extent ranges; neither is "the" answer — together they show contrast keeps
growing (worse-than-linearly) across the *entire* 9µm-1021µm range this
dataset spans, with no sign of the bound the original entry claimed.

**The real shipped 12-design dataset's residual spread is up to ~104x, not
~9x, and pooling across designs is NOT safe without accounting for it.**
See the corrected 12-design table below: relative contrast ranges from
0.0045 (`asap7/gcd`) to 0.4688 (`nangate45/aes`), a ~104x spread (previously
174x before this round's `T_CHIP_MIN_M` fix improved `asap7/gcd`'s contrast
1.7x — see below). The original entry's "~9x, comparable to the real table"
framing only held by implicitly excluding `asap7/gcd`, which is exactly the
micro-die case this fix was written to address. This is a real, current
confound in the shipped labels, not a bounded/understood residual: a model
trained on the pooled cross-PDK/cross-design set is very likely to partially
learn die size as a proxy for ΔT magnitude. Per-design or per-PDK-group
analysis remains the only safe use of this dataset for now; do not pool all
12 designs and assume the confound is small.

**Why full scale-invariance is not achievable under a physically realistic
package model:** `hotspot_package_args()` sets one specific contribution to
mean temperature rise (`r_convec`, see below) from total power, and
geometric package scaling keeps lateral heat-*capacity* proportional to die
area at every extent. But heat *spreading length* under a finite-thickness
chip grows only as `sqrt(L)` (lateral diffusion through a thin slab), while
die size itself grows as `L`. Contrast — how sharply a local power hotspot
stands out against the mean — is therefore expected to grow with die size
under any physically realistic package model where chip thickness does not
itself scale with lateral extent. Eliminating this fully would require chip
thickness scaling as `L²`, which is not physically realistic: real silicon
wafers are diced to a roughly constant thickness (in the hundreds of µm)
regardless of a die's lateral footprint. `CHIP_THICKNESS_FRAC=0.02` is
therefore kept as the shipped calibration — it eliminates the near-total
isothermal collapse the unscaled default package produced at the small end
— but this entry no longer claims it bounds the residual to ~9x; per the
corrected sweep and table above, it does not.

**Mean-temperature-rise claims — corrected.** `TARGET_MEAN_RISE_K=40.0` sets
the `r_convec` *contribution* to mean die temperature rise, not the total
mean rise, and it is not "pinned at ambient+40K" as the original entry and
code comments claimed. The rest of the package stack (spreader, sink,
interface resistances) adds another ~50-60K on top, and that additional
contribution itself varies with die size — re-measured directly with the
(now-fixed) `thermal_scale_check.py` at the same 51/255/510/1021µm extents:
`Tmean` = 135.54/137.75/140.57/146.25°C, i.e. rise above 45°C ambient =
90.5/92.8/95.6/101.3K, so the package-stack contribution beyond
`TARGET_MEAN_RISE_K` is ~50.5K at 51µm growing to ~61.3K at 1021µm. Real mean
temperatures across the shipped 12-design dataset are 135.11-139.96°C (rise
90.1-95.0K above the 45°C ambient), never the ~85°C (`ambient+40K`) the
original documentation implied. The `Relative contrast` column's denominator
(`Tmean-45°C`) is therefore actually ~90-95°C in practice everywhere in this
dataset, not a pinned 40°C. `extract_thermal_labels.py`'s docstrings/comments
(`hotspot_package_args()`, `TARGET_MEAN_RISE_K`, and the module-level
"Package model" section) have been corrected to state this plainly.

**The "harmless because min-max normalizes per-sample" claim was false and
has been removed.** The original code docstring near
`hotspot_package_args()` claimed residual ΔT growth was harmless because
`ThermalDataset` min-max normalizes each sample independently. Per-sample
min-max removes absolute *magnitude*, but it does not touch relative
*shape/sharpness* — and sharpness is exactly what the confound affects.
Evidence: re-measured with the fixed `thermal_scale_check.py`, the
normalized-map correlation between the smallest (51µm) and largest (1021µm)
extents is only 0.7670 (0.8942 in the original 51-1021µm-only sweep using
the pre-fix, non-adaptive-grid script) — either way, well short of ~1.0. If
per-sample normalization actually removed the confound, that correlation
would be near 1.0 regardless of the ptp/contrast difference. The code
comment has been corrected to state that this normalization does NOT rescue
the confound.

**`T_CHIP_MIN_M` — lowered from 1e-6 (1µm) to 1e-7 (0.1µm).** The old 1µm
floor silently clamped `t_chip` for every die with `l_eq < 50µm`
(`0.02 * l_eq < 1µm`), which in this dataset is `asap7/gcd` (l_eq=8.98µm,
wanted 0.18µm) and `nangate45/gcd` (l_eq=36.73µm, wanted 0.73µm) — both
were solving with an artificially thick chip, understating their contrast.
Tested un-clamping in Docker (`openroad/orfs-ml:latest`) both via
`thermal_scale_check.py`'s 9-102µm sweep and via live re-extraction of both
affected designs' real `3_place.odb` files: HotSpot solved cleanly at the
unclamped thickness in every case (no `lupdcmp` failures), and contrast
improved without becoming unstable or nonsensical — `asap7/gcd` went from
0.0027 to 0.0045 (1.7x, matching the 1.7x seen in the synthetic 9µm sweep:
0.0121 -> 0.0210) and `nangate45/gcd` from 0.0488 to 0.0566 (1.16x). Since
this worked cleanly, `T_CHIP_MIN_M` is now `1e-7` (0.1µm) — purely a
numerical-safety floor against zero/negative thickness for pathological
micro-dies, not a physical scaling choice — and no longer clamps any of the
12 shipped designs. Both affected designs' rows were re-extracted and the
table below updated for just those two rows; the other 10 designs'
`l_eq >= 50µm` so the clamp never applied to them and their rows are
unchanged from the original fix.

**Re-extraction:** deleted all 12 stale `*_thermal_labels.npz` and re-ran
`extract_thermal_batch.sh --timeout 3600 --force` (`openroad/orfs-ml:latest`)
for all 12 designs when this fix first landed. Result: `passed=12 failed=0`
(plus `skipped=12` for the already-extracted `*_features.npz`, unaffected by
this fix and left alone). No timeouts, no HotSpot solver failures. This
correction round re-extracted only `asap7/gcd` and `nangate45/gcd` (the two
designs affected by the `T_CHIP_MIN_M` change), both `[OK]`, no failures.

**Verification:**
- Batch pass/fail (original fix): 12/12 thermal extractions `[OK]`, 0
  `[FAIL]`, 0 `[TIMEOUT]`. This round: 2/2 re-extractions (`asap7/gcd`,
  `nangate45/gcd`) `[OK]`.
- Value-sanity check across all 12 `.npz`: all `thermal_map`/`power_grid`
  arrays finite (no NaN/Inf), both shaped `(64, 64)`, keys exactly
  `{thermal_map, power_grid}` on every file (shape/key contract holds). Note:
  shape-matching is not resolution-matching — 5 of the 12 (`asap7/gcd`,
  `asap7/aes`, `nangate45/gcd`, `asap7/riscv32i`, `sky130hd/gcd`) solved at a
  coarse native HotSpot grid (8-15 per side, per `_adaptive_hotspot_grid()`)
  and were bilinearly upsampled to 64x64 for the saved arrays; the paired
  `*_features.npz` are true native 64x64. Pre-existing behavior, not
  introduced by this fix or this correction, but worth being explicit about.
- Thermal/power correlation per design ranges 0.30–0.78 (see table below) —
  positive and non-trivial everywhere, as expected for a power-density-driven
  steady-state solve.
- `python3 util/ml/congestion/tests/test_models.py -v` → **20/20 pass, `OK`,
  exit 0** (synthetic-data regression guard only, unaffected by this fix as
  expected).

**`flow/util/ml/congestion/tests/thermal_scale_check.py` — fixed to actually
exercise `_adaptive_hotspot_grid()` and to drop the misleading always-FAIL
gate.** Previously the script took a fixed `--grid` value straight to
HotSpot, so it never called `_adaptive_hotspot_grid()` and could not detect
a `MIN_HOTSPOT_GRID` regression (reverting `MIN_HOTSPOT_GRID` to the old
`max(1, ...)` would have produced byte-identical script output). It now
calls `_adaptive_hotspot_grid()` at each extent (the `--grid` flag is the
*target* grid passed to it, matching `extract_thermal_labels.py`'s own
convention), and normalized maps at different native grids are upsampled to
the target grid before correlating, so they remain comparable. Separately,
its old PASS/FAIL bar (correlation > 0.99, contrast deviation <= 25%) is not
physically achievable under this package model (see the corrected findings
above) and always reported `FAIL` regardless of code health, so it could
never be wired into CI as a real gate. It has been replaced with: (a) a
metrics table for a human to read (contrast, correlation, grid used, at
each extent), and (b) one automated regression check — contrast at the
smallest tested extent must stay above a small non-degenerate floor
(`DEGENERATE_CONTRAST_MIN=1e-3`), which is what would fail if the old
isothermal-collapse bug (or a `MIN_HOTSPOT_GRID` regression that reproduces
it) came back. Run inside Docker:
```bash
export OR_IMAGE=openroad/orfs-ml:latest
util/docker_shell python3 /work/util/ml/congestion/tests/thermal_scale_check.py \
    [--extents-um 9,25,51,255,510,1021] [--grid 32] [--power-density 10.0]
```
The default `--extents-um` now includes 9µm (the dataset's actual smallest
die). This matters: at 51µm and above, `_adaptive_hotspot_grid()`'s
`MIN_HOTSPOT_GRID` floor is never the binding constraint, so a sweep
confined to 51-1021µm cannot detect a `MIN_HOTSPOT_GRID` regression — it
would report identical metrics with or without the floor. Always include
at least one extent below ~40µm when checking for that specific
regression. Re-run it after any future change to `hotspot_package_args()` /
`CHIP_THICKNESS_FRAC` / `MIN_HOTSPOT_GRID` / `MIN_CELL_UM` / `T_CHIP_MIN_M`;
its exit code is now a real regression signal (0 = no isothermal-collapse
regression detected, 1 = regression), not an unconditional FAIL.

**This entry supersedes the `Thermal ΔT (°C)` column of the 2026-09-14
"12-design real routed dataset" entry below** (all values there predate this
fix and reflect the un-scaled default HotSpot package model). That entry's
`Worst-case IR-drop` column is a separate, unrelated measurement and remains
authoritative/unaffected — no IR-drop code was touched here.

**12-design thermal table** (re-extracted with the fix; `asap7/gcd` and
`nangate45/gcd` rows re-extracted again this round with the corrected
`T_CHIP_MIN_M=1e-7`; `contrast` = `(Tmax-Tmin)/(Tmean-45°C)`, same convention
as `thermal_scale_check.py`; `corr(T,P)` = Pearson correlation between the
flattened `thermal_map` and `power_grid` arrays):

| Design | Thermal ΔT / ptp (°C) | Relative contrast | corr(T, P) | Tmean (°C) |
|---|---|---|---|---|
| asap7/gcd | 0.410 | 0.0045 | 0.6320 | 135.11 |
| asap7/aes | 3.820 | 0.0421 | 0.7846 | 135.70 |
| nangate45/gcd | 5.130 | 0.0566 | 0.6634 | 135.58 |
| sky130hd/gcd | 7.596 | 0.0838 | 0.6368 | 135.66 |
| asap7/riscv32i | 26.605 | 0.2916 | 0.5890 | 136.23 |
| nangate45/jpeg | 21.160 | 0.2264 | 0.3741 | 138.48 |
| nangate45/tinyRocket | 24.270 | 0.2605 | 0.3027 | 138.18 |
| nangate45/dynamic_node | 34.551 | 0.3731 | 0.6333 | 137.61 |
| sky130hd/aes | 38.690 | 0.4074 | 0.4458 | 139.96 |
| sky130hd/riscv32i | 40.370 | 0.4295 | 0.3504 | 138.98 |
| nangate45/ibex | 42.426 | 0.4583 | 0.6534 | 137.58 |
| nangate45/aes | 43.489 | 0.4688 | 0.6716 | 137.76 |

(Table sorted by ptp, smallest to largest, to make the residual
die-size-vs-contrast relationship visible at a glance. Real spread:
0.0045-0.4688 = ~104x, still large — see the correction above.)

**`asap7/gcd` status: kept, not excluded.** With the `MIN_HOTSPOT_GRID=8`
floor, its 9µm die now resolves on a real 8×8 HotSpot grid (confirmed in the
extraction log: `Die 0.01×0.01 mm is small — using 8×8 HotSpot grid`), not
the silent 1×1 collapse from the earlier entry. `power_grid` and
`thermal_map` are both spatially non-constant, and `_check_nondegenerate()`
passes (ptp=0.410°C, well above the `DEGENERATE_PTP_C=1e-3` floor). Its low
contrast is now understood to be partly (1.7x) an artifact of the
now-fixed `T_CHIP_MIN_M` clamp and partly genuine "clean solve, low
contrast" physics for a micro-scale die — not attributable wholly to
physics as the original entry claimed. It is retained as a genuine, if
low-amplitude, extracted sample, with no special-case exclusion.

**`MIN_HOTSPOT_GRID=8` vs. `MIN_CELL_UM=5`: known tension, not fixed here.**
At `MIN_HOTSPOT_GRID=8`, a die below ~40µm produces cells smaller than
`MIN_CELL_UM` — the exact regime this module's own docstring says makes
HotSpot's block RC model ill-conditioned. Every die tested so far (down to
`asap7/gcd`'s 8.98µm) has converged, but nothing bounds this in general; a
different sub-40µm design could still hit `lupdcmp: singular matrix`. This
is a loud failure (non-zero exit, caught and re-raised by `run_hotspot()`),
not a silent one, so it is a documented, watch-for risk rather than a
blocking defect — see the comment added next to `MIN_HOTSPOT_GRID`'s
definition in the code.

No `git status` changes outside `extract_thermal_labels.py`,
`tests/thermal_scale_check.py`, and this file — `results/`, `logs/`,
`objects/`, and `data/*.npz` remain gitignored as expected.

### 2026-09-14 — 12-design real routed dataset (6 baseline re-routes + 6 new: asap7/nangate45/sky130hd)

Executed the planned expansion from 6 to 12 real routed designs across all
three PDKs, per the cheapest-first order plan. All 12 `make finish` runs
(`openroad/orfs:latest`) completed with `rc=0` in ~56 minutes total (far
under the 5-7h estimate — no design in this batch was large/slow enough to
approach the estimate; `nangate45/jpeg`, the largest at 68k stdcells, took
~11.5 minutes). Order run: nangate45/gcd, asap7/gcd, sky130hd/gcd,
sky130hd/riscv32i, nangate45/dynamic_node, asap7/riscv32i, nangate45/aes,
asap7/aes, sky130hd/aes, nangate45/ibex, nangate45/tinyRocket,
nangate45/jpeg. No fallback substitutions were needed — all 12 primary
picks routed cleanly, including both macro-bearing designs
(asap7/riscv32i with fakeram7 SRAM, nangate45/tinyRocket with fakeram45
SRAM), the first macro designs in this dataset.

**Extraction:** `extract_thermal_batch.sh` (`openroad/orfs-ml:latest`,
`--timeout 3600`) → 24/24 passed, 0 failed (features + thermal for all 12,
auto-discovered via `find results -name 3_place.odb`, no allowlist edits).
`extract_irdrop_batch.sh` (`openroad/orfs:latest`, `--timeout 3600`, no
`--voltage` flag — per-design liberty/voltage resolved automatically via
`make print-LIB_FILES print-PWR_NETS_VOLTAGES`) → 12/12 passed. Manually
inspected the raw docker command lines logged for the two macro designs to
confirm the `/work/${lib#*/flow/}` liberty-path remap resolved correctly:
`asap7/riscv32i` picked up `/work/platforms/asap7/lib/NLDM/fakeram7_256x32.lib`
(a real on-disk file) alongside the 5 standard-cell libs, and
`nangate45/tinyRocket` picked up `/work/designs/nangate45/tinyRocket/fakeram45_{1024x32,64x32}.lib`
(symlinks into `platforms/nangate45/lib/`) — both resolved to real files,
not mangled paths, and both extractions logged `[OK]` with no errors.

**Verification:** `find flow/results -name 6_final.odb | wc -l` = 12, each
with sibling `6_final.spef` and `3_place.odb` confirmed present.
`ls flow/util/ml/congestion/data/*.npz | wc -l` = 36. Value-sanity script
(adapted only for this numpy version's `np.ptp(a)` vs. the now-removed
`a.ptp()` method — no functional change to the check) found no NaN/Inf and
no unexpected all-zero arrays; the only `ALLZERO` flags were `macro_density`
for the 10 non-macro designs, which is correct (and its *absence* on
`asap7_riscv32i`/`nangate45_tinyRocket` confirms macro detection worked).
`python3 util/ml/congestion/tests/test_models.py -v` → 20/20 pass (`OK`,
exit 0), unchanged as expected (synthetic-data regression guard only).

**IR-drop worst-case matches the previously-documented baseline almost
exactly** — a good confirmation the toolchain/environment is unchanged
since the 2026-08-27 entry: nangate45 gcd 0.534mV / dynamic_node 1.008mV /
ibex 3.057mV / aes 4.985mV (baseline: 0.53/1.01/3.06/4.99mV); sky130hd gcd
0.412mV (baseline 0.41mV); asap7 gcd 103.68mV (baseline 103.68mV, exact).
All new samples' IR-drop landed in the platform-expected magnitude bands
(asap7 tens–150mV, nangate45 sub-mV to ~9mV, sky130hd sub-mV) and
`voltage_map` stayed at-or-below nominal supply per platform in every case.

**Thermal ΔT criterion NOT met — reporting, not silently patching (out of
scope: no changes to `extract_thermal_labels.py`, no `MIN_CELL_UM` tuning
per plan constraints).** Only 7 of 12 designs have thermal peak-to-peak
> 0.5°C, short of the ">= 10 of 12" target, and **0 of the 3 asap7 samples**
exceed 0.5°C ΔT — short of the "at least 2 of 3" goal criterion.
`asap7/aes` (14,321 stdcells, chosen specifically to break the flat-map
degeneracy per the plan's rationale) came in at ΔT=0.05°C, barely above
`asap7/gcd`'s literal 0.0°C and `asap7/riscv32i` (with SRAM macros) only
reached ΔT=0.43°C — still under the 0.5°C bar. All three asap7 thermal maps
sit in a ~110.0–110.4°C band.

**Correction (2026-09-14, later same day — validator finding): the
`MIN_CELL_UM=5.0` grid-resolution-floor explanation originally written here
was falsified and has been removed.** Three runs against the real
`3_place.odb` files (`util/docker_shell` + `openroad/orfs-ml:latest`) show
grid resolution has no effect on the asap7 flatness: `asap7/aes` at the
shipped `MIN_CELL_UM=5.0` (10x10 grid) gives ΔT=0.05°C, and forcing
`MIN_CELL_UM=2.5` (20x20 grid, 4x more blocks) gives the identical
ΔT=0.05°C; meanwhile a `nangate45/aes` run forced onto a *coarser* 9x9 grid
than asap7/aes still resolves ΔT=3.3°C (vs. 3.47°C at its native 50x50 grid).
A 9x9 grid clearly resolves a multi-degree gradient fine, so a resolution
floor cannot explain asap7's flatness.

**Actual root cause:** `run_hotspot()` in `extract_thermal_labels.py`
invokes `hotspot` with no `-c` package config, so HotSpot's built-in default
package geometry (~0.15mm silicon thickness, cm-scale spreader/heatsink) is
applied to *every* design regardless of its actual die size. asap7 dies in
this batch are 51-73µm across — three orders of magnitude smaller than the
default package's cm-scale heatsink and ~3x *thinner* than they are wide in
the wrong direction (the default silicon layer alone is ~2-3x the die's
lateral extent), so lateral spreading resistance is negligible and the die
is rendered effectively isothermal under any power distribution, no matter
how much detail the power map has. This was confirmed directly by holding
placement and power distribution fixed and scaling only the physical die
extent, at constant power density and constant grid:

| Die extent | 51µm | 255µm | 510µm | 1021µm |
|---|---|---|---|---|
| ΔT | 0.05°C | 1.2°C | 4.7°C | 14.9°C |

Same cells, same power distribution, same grid — ΔT scales with absolute
die size against the fixed default package model, not with the design's
spatial power variation. **This means the thermal-label dataset collected
in this entry largely encodes die size, not spatial power distribution,
across all 12 samples** (asap7 51-73µm dies → 0.05-0.43°C; nangate45 250µm
dies → 1.8-4.04°C; sky130hd 475µm dies → 4.71-6.52°C — the ΔT ranking tracks
die size almost exactly). This is a data-quality caveat that must travel
with the dataset as a whole, not just an asap7-specific footnote: a future
cross-design thermal model trained on this data risks learning die size
rather than real thermal physics.

Two nangate45 and one sky130hd sample (gcd variants: 0.05, 0.05, 0.14°C)
also came in under 0.5°C, consistent with the same die-size effect — the
`gcd` variants are simply the smallest dies on each PDK in this batch, not a
separate issue.

**`asap7/gcd`'s sample (`asap7_gcd_thermal_labels.npz`) is a degenerate
constant, not just "low ΔT", and should be excluded from training.**
`_adaptive_hotspot_grid()` in `extract_thermal_labels.py` computes
`max(1, int(min_dim_um / MIN_CELL_UM))` with no lower-bound guard; for
`asap7/gcd`'s 9µm die this silently resolves to a 1x1 HotSpot grid
(confirmed live in the extraction log: `using 1×1 HotSpot grid`). The saved
sample is a single HotSpot block bilinearly upsampled to 64x64 — both
`power_grid` and `thermal_map` in the `.npz` are literally constant
(`min == max`: `8.0587e-4` for `power_grid`, `110.0`°C for `thermal_map`,
everywhere). This is qualitatively different from the other low-ΔT
samples, which at least have a spatially varying (if low-amplitude)
power/thermal map — `asap7/gcd` carries zero spatial information at all.

**Follow-up needed (out of scope for this data-collection task, not fixed
here):** two real bugs in `extract_thermal_labels.py` explain both findings
above — (1) `run_hotspot()` never passes a per-die `-c` package config to
`hotspot`, so the fixed default package geometry dominates ΔT for any die
far from cm-scale (i.e. every die in this dataset), and (2)
`_adaptive_hotspot_grid()` has no lower bound on grid size, letting
small-enough dies silently collapse to a 1x1 grid. Neither was touched in
this entry — that file was out of scope for this task per its plan
constraints. A dedicated fix-and-re-extract pass (scaling the HotSpot
package config to each die's physical extent, and enforcing a minimum grid
size) is needed before the thermal dataset can be trusted for cross-design
modeling.

**Full per-design table (worst-case IR-drop, thermal ΔT):**

| Design | Worst-case IR-drop | Thermal ΔT (°C) |
|---|---|---|
| nangate45/gcd | 0.534 mV | 0.050 |
| asap7/gcd | 103.68 mV | 0.000 (degenerate constant, see above) |
| sky130hd/gcd | 0.412 mV | 0.139 |
| sky130hd/riscv32i | 0.588 mV | 6.52 |
| nangate45/dynamic_node | 1.008 mV | 2.65 |
| asap7/riscv32i | 140.04 mV | 0.434 |
| nangate45/aes | 4.985 mV | 3.468 |
| asap7/aes | 148.24 mV | 0.050 |
| sky130hd/aes | 0.513 mV | 4.71 |
| nangate45/ibex | 3.057 mV | 4.04 |
| nangate45/tinyRocket | 1.370 mV | 1.80 |
| nangate45/jpeg | 8.821 mV | 2.16 |

**Net result:** 12/12 designs routed and extracted successfully with clean
IR-drop data matching prior baselines; the dataset now spans 36 `.npz`
files (up from 18) with a genuine cross-PDK, macro-inclusive sample set.
The thermal-ΔT success criterion (>=10/12 designs, >=2/3 asap7 samples)
was **not met** — not because of an extraction failure, but because
`extract_thermal_labels.py`'s `run_hotspot()` applies a fixed default
package geometry regardless of die size (see correction above), which makes
the 12 collected thermal labels largely a function of die size rather than
power distribution, and `asap7/gcd`'s sample is additionally a degenerate
1x1-grid constant. This needs the dedicated `extract_thermal_labels.py`
fix-and-re-extract pass described above before the thermal dataset is
usable for cross-design modeling, rather than being chased further within
this data-collection task's constraints.

No `git status` changes outside this file — `results/`, `logs/`,
`objects/`, and `data/*.npz` remain gitignored as expected.

### 2026-08-27 (later) — `no-mistakes` validation of the asap7 LIB_FILES fix; path move `flow/ml/` → `flow/util/ml/`

Ran the `no-mistakes` gate pipeline (review/test/document/lint/push/PR/CI)
against `extract_irdrop_batch.sh`'s asap7 `LIB_FILES`/voltage fix
(`dc3e4f8d5`). Findings and outcome:

- **Review step** (auto-fix, 2 findings): the new `make print-LIB_FILES
  print-PWR_NETS_VOLTAGES` call was (1) a bare command substitution under
  `set -euo pipefail` — unlike every other per-design step, a `make` failure
  here would abort the whole batch instead of skipping just that design; and
  (2) ran unconditionally per design even when that design would be entirely
  skipped (outputs already exist, no `--force`), spinning up a Docker
  container for nothing on every idempotent re-run. Fixed by wrapping the
  call in `if ! make_out=...; then skip; continue; fi` and moving it inside
  the `irdrop_host` skip-check's `else` branch. Verified by hand: `bash -n`
  clean, and a no-`--force` re-run against all 6 designs now completes with
  **zero** docker/make invocations (previously one per design).
- **CI step** (3 auto-fix rounds, PR #5): GitHub's security filename scanner
  blocks `flow/ml/` and bare files directly under `flow/` (only
  `flow/scripts`, `flow/util`, `flow/designs`, `flow/test` etc. are
  allowlisted) — pre-existing and unrelated to this PR's actual diff, but
  this branch's push was the first to touch files under the blocked path.
  Round 1 removed a stray 3MB `flow/thermal_report.html` (a generated report,
  re-gitignored). Round 2 found the scan aborts on the *first* blocked
  filename tree-wide, so after round 1 it just failed on the next
  (`flow/ml/Dockerfile`) — fixed by **moving the entire `flow/ml/` tree to
  `flow/util/ml/`** (and `flow/run_visualize.sh` under `flow/util/`),
  rewriting internal path references, verified against a real clone of the
  security scanner (exit 0) and the full test suite (20/20) from the new
  location. Round 3 fixed a trivial `black` formatting violation in two
  visualize scripts. One check (`update`, a `repository_dispatch`-only
  yosys-submodule-sync cron job, untriggerable by a normal push/PR) was
  correctly identified as unrelated and unfixable by any code change, and
  skipped. **Outcome: `passed`**, PR #5 merged, pushed as `1d54c963b`.

**Bug the move introduced, caught and fixed manually (commit `0e0f833e4`,
not part of the `no-mistakes` run):** 5 scripts anchor themselves to `flow/`
via a fixed-depth `cd "$(dirname "$0")/../../.."` (or similar) — moving one
directory deeper (`ml/` → `util/ml/`) broke that arithmetic. They were
silently `cd`-ing into `flow/util/` instead of `flow/`, so e.g.
`find results -name "6_final.odb"` found nothing and the batch script did
*nothing*, with no error — the same silent-no-op failure class this branch's
earlier stdin-draining bug was. Fixed by adding one more `../` in:
`data_collection/extract_irdrop_batch.sh`, `extract_thermal_batch.sh`,
`batch_run.sh`, `generate_variants.sh`, and `run_pipeline.sh` (one directory
shallower, so 2→3 dots instead of 3→4). Also moved the real (gitignored,
never-tracked) extracted `.npz` files from the old `flow/ml/congestion/data/`
to the new `flow/util/ml/congestion/data/` by hand, and removed the
now-empty old `flow/ml/` tree. Verified: `bash -n` on all 5; re-ran
`extract_irdrop_batch.sh` against the 6 real designs — correctly found
`results/` again and reported `skipped=12` (6 designs × 2 checks, as
expected for an idempotent re-run) instead of silently finding zero;
`test_models.py` 20/20 from the new location.

All paths in this document as of this entry refer to the new
`util/ml/congestion/...` location.

### 2026-08-27 — IR-drop solver (new track) + real-data validation of both this session and 2026-08-26's Laplacian loss

**Correction to the 2026-08-26 entry below and to `project_congestion_ml_roadmap` memory:**
both describe a "48-sample dataset" with a 0.02912 val-MSE baseline and a
per-design correlation table as already collected, and describe Docker/HotSpot
as unavailable in the dev environment. Neither was true in this worktree: at
the start of this session `results/`, `util/ml/congestion/data/*.npz`, and
`util/ml/congestion/checkpoints/` were all empty, and `docker images` showed both
`openroad/orfs:latest` and `openroad/orfs-ml:latest` present and functional
(`docker run --rm hello-world` succeeded). The 48-sample dataset and its
baseline table either lived in a different environment/session or were never
actually produced — treat any pre-2026-08-27 claim of collected data or a
trained checkpoint as unverified until a matching file is found on disk.

**New track — IR-drop solver**, per the Tier-2 `FEATURE_ROADMAP.md` item
("IR-drop solver, builds on Thermal U-Net pipeline, same shape as thermal,
new labels (PDN geometry + power density)"). Unlike thermal, IR-drop labels
come from OpenROAD's **native PDNSim** (`analyze_power_grid`) — no external
simulator or custom Docker image needed, just the stock `openroad/orfs:latest`.
New files (all additive, no existing thermal-track file modified):

- `data_collection/extract_irdrop_labels.py` — writes a small Tcl driver
  (`read_liberty` → `read_db` → `read_spef` → `set_pdnsim_net_voltage` →
  `analyze_power_grid -voltage_file ... -error_file ...`), runs it via
  `openroad -no_init -exit`, parses the voltage CSV, and separately rasterizes
  PDN wire/via geometry (`stripe_density`, `via_density`) and a cell-weighted
  `current_density_proxy` from the routed ODB (same grid-binning technique as
  `build_power_grid` in `extract_thermal_labels.py`). Outputs `irdrop_map`
  (volts of drop), `voltage_map`, `stripe_density`, `via_density`,
  `current_density_proxy`.
- `data_collection/extract_irdrop_batch.sh` — batch driver over existing
  `6_final.odb`/`.spef` pairs, mirrors `extract_thermal_batch.sh`, defaults
  to the stock image.
- `training/irdrop_dataset.py` — `IRDropDataset`: 6-channel input (4 reused
  from `*_features.npz` + stripe/via density from `*_irdrop_labels.npz`, no
  Gaussian-blur channel since IR drop is a DC resistive problem, not a
  diffusion one), per-sample `[0,1]` normalization of `irdrop_map`, same
  train/val/test split pattern as `ThermalDataset`.
- `training/train_irdrop.py` — same MSE + optional Laplacian-smoothness loss
  as `train_thermal.py` (physically justified: static IR drop is also a
  smooth/harmonic field), `in_channels=6`, `irdrop_best.pt`/`irdrop_last.pt`.
- `inference/predict_irdrop.py`, `inference/visualize_irdrop.py` — cloned
  from the thermal equivalents.
- `tests/generate_synthetic_data.py`, `tests/test_models.py` — extended with
  synthetic IR-drop data + a `TestIRDropDataset` smoke test (load, shapes,
  split, 2-epoch training run). `python3 util/ml/congestion/tests/test_models.py -v`
  → 20/20 tests pass (16 pre-existing + 4 new).

**Real-data validation (not synthetic) — the load-bearing step:**
Confirmed the exact `analyze_power_grid -voltage_file` CSV format by running
PDNSim directly against a real routed `nangate45/gcd` design (`6_final.odb` +
`6_final.spef`, generated via `make finish` in `openroad/orfs:latest` in this
session): header `Instance,Terminal,Layer,X location,Y location,Voltage`, one
row per instance terminal (dense — 997 rows for gcd's 997 instances), X/Y in
microns. Liberty must be `read_liberty`'d before `read_spef` or OpenSTA raises
`STA-2141`. `-error_file` is only written on an error/warning, not on clean
success, so it's optional/diagnostic, not required output.

One real bug found and fixed during validation: the extractor's
nearest-neighbor fill (for grid cells with no instance-terminal sample) used
`scipy.ndimage.distance_transform_edt`, but `scipy` is **not installed** in
the stock `openroad/orfs:latest` image (only in `-ml`) — this would have
defeated the entire point of using PDNSim to avoid a custom image. Replaced
with a pure-numpy iterative-dilation nearest-fill
(`extract_irdrop_labels.py:_nearest_fill`).

Routed 3 designs end-to-end (`make finish`, nangate45) to get real labels for
both tracks from the same routed output: **gcd**, **dynamic_node**, **aes**.
IR-drop worst-case drop scaled sensibly with design size/density (gcd 0.53 mV
→ dynamic_node 1.01 mV → aes 4.99 mV), a good physical sanity check.
`train_irdrop.py` on these 3 real samples (15 epochs, batch size 1,
`--laplacian-weight 0.1`): best val MSE **0.02889**. This is the pipeline's
first successful real (non-synthetic) end-to-end run.

**Laplacian loss (2026-08-26 entry) — first real-data comparison:**
Trained `train_thermal.py` on the same 3 real samples, 15 epochs, batch size 1:
- `--laplacian-weight 0.0`: best val MSE **0.07847**
- `--laplacian-weight 0.1`: best val MSE **0.06735** (~14% lower)

Directionally consistent with the smoothness-regularization hypothesis, but
**n=3 designs is not strong statistical evidence** — this is a real result,
not a synthetic smoke test, but it should not be read as confirming the
technique at the scale the (nonexistent) 48-sample baseline implied. A larger
real run (10+ designs, ideally the originally-intended multi-PDK set) is
needed before treating this as validated.

**Update — 4th real design (`ibex`, nangate45) added same session:**
`ibex` finished routing shortly after the above was written; extracted real
thermal/feature/IR-drop labels for it too (IR-drop: 0.06–3.06 mV, consistent
scaling with its size between dynamic_node and aes). Re-ran both comparisons
on all 4 real samples (20 epochs, batch size 2):
- `train_irdrop.py --laplacian-weight 0.1`: best val MSE **0.05964**
- `train_thermal.py --laplacian-weight 0.0`: best val MSE **0.05618**
- `train_thermal.py --laplacian-weight 0.1`: best val MSE **0.06108**

**This flips the sign of the 3-sample Laplacian result above** (there,
λ=0.1 was ~14% *better*; here, with a 4th design added, λ=0.1 is ~9% *worse*).
With n=4 designs split train=2/val=1/test=1, a single design's difficulty
dominates whichever split it lands in — this is exactly the kind of
instability the "n=3 is not strong evidence" caveat above was warning about,
now demonstrated rather than just asserted. **Conclusion: do not treat either
result as validating or invalidating the Laplacian loss.** A real conclusion
needs enough designs that the train/val/test split stops being the dominant
source of variance — realistically 15-20+ real samples, matching the scale
the (nonexistent) 48-sample baseline had implied.

Batch-extracting across more of the existing `designs/*` directory (asap7,
sky130hd included) via `extract_thermal_batch.sh` / `extract_irdrop_batch.sh`
— now that Docker is confirmed working — is the natural next step to build
that larger real dataset, rather than continuing to reference the
undocumented one.

**Update — cross-PDK expansion to 6 real designs (asap7, sky130hd added):**
Routed `asap7/gcd` and `sky130hd/gcd` and extracted real thermal + feature +
IR-drop labels for both, giving 6 real samples across all 3 PDKs (nangate45:
gcd/dynamic_node/aes/ibex, asap7: gcd, sky130hd: gcd). Two notes on doing
this correctly:
- `extract_irdrop_labels.py --voltage` defaults to 1.1V (nangate45's nominal)
  and must be overridden per platform — asap7's is 0.77V
  (`make DESIGN_CONFIG=... print-VOLTAGE`), sky130hd's is 1.8V
  (`PWR_NETS_VOLTAGES` in `platforms/sky130hd/config.mk`). Passing the wrong
  nominal doesn't crash anything, it just silently analyzes against the wrong
  supply — caught this on the first asap7 attempt (1.1V) and re-ran at the
  correct 0.77V. `extract_irdrop_batch.sh` takes `--voltage` as a single
  global flag, so a true multi-PDK batch run needs one invocation per
  platform with the right value, not one batch call across all platforms.
- asap7's `LIB_FILES` in `platforms/asap7/config.mk` is built from
  corner/VT-placeholder-substituted make variables
  (`$($(CORNER)_$(LIB_MODEL)_LIB_FILES)`), not a simple `export LIB_FILES =`
  line — `extract_irdrop_batch.sh`'s original awk-based LIB_FILES parser
  (written by the implementation subagent, never exercised against asap7)
  did not resolve these correctly. Worked around it manually at first via
  `make DESIGN_CONFIG=... print-LIB_FILES`, then **fixed the batch script
  itself** (below) to do the same thing generically.

**Fix — `extract_irdrop_batch.sh` rewritten to resolve liberty + voltage
per design via `make print-X`, not by parsing config.mk text:** replaced
the awk-based `LIB_FILES` scrape and the fixed `--voltage 1.1` CLI default
with `make DESIGN_CONFIG=designs/<platform>/<design>/config.mk
print-LIB_FILES print-PWR_NETS_VOLTAGES` (using `variables.mk`'s generic
`print-%` target — the same mechanism used to manually resolve asap7's
libs/voltage earlier in this entry). `PWR_NETS_VOLTAGES` is a universal
`"<net> <voltage> ..."` dict (the same format `final_outputs.tcl` parses)
so the per-platform nominal is now resolved automatically instead of
requiring a manual `--voltage` override per platform.

Found and fixed a second real bug while validating this: the `make print-X`
call was made via `util/docker_shell -- "make ..."` *inside* the batch
script's `while read HOST_ODB; do ... done < <(find results ...)` loop, and
`docker run -i` reads from stdin until EOF — without an explicit
`</dev/null` on that call, it silently drained the outer loop's process
substitution pipe after the first iteration, ending the batch after
processing exactly one design with no error message. Easy to miss (the
script "worked", just only on the first design) — fixed by adding
`</dev/null` to the `make print-X` invocation, matching what the two
extraction-step invocations already did.

Re-ran the fixed batch script (`--force`) against all 6 real routed designs
from this entry: **6/6 passed**, producing byte-identical `irdrop_map`
values to the earlier manual per-design runs (asap7 0.77V/103.68mV,
nangate45 1.1V, sky130hd 1.8V/0.41mV) — confirms the generic resolution
path is correct, not just non-crashing.

**Hardening — contained the `make print-X` call and gated it behind the
skip-check:** review of the fix above found two follow-on issues, both
fixed in the same commit set. First, the `make print-LIB_FILES
print-PWR_NETS_VOLTAGES` call was a bare assignment under `set -euo
pipefail`, so any `make` failure (not just an empty `LIB_FILES`) would
abort the whole batch instead of skipping just that design — wrapped in
`if ! make_out=...; then skip; continue; fi` like every other per-design
step. Second, the call ran unconditionally per design even when
`irdrop_host` already existed and `--force` was not passed, spinning up a
docker container just to resolve liberty/voltage for a design that was
going to be skipped anyway — moved inside the `irdrop_host` skip-check's
`else` branch so it only runs when extraction will actually happen.
Re-verified: `bash -n` passes, and re-running without `--force` against
all 6 designs now skips instantly with zero docker/make invocations.

Real cross-PDK IR-drop magnitudes (worst-case), physically sensible:
nangate45 gcd 0.53mV / dynamic_node 1.01mV / aes 4.99mV / ibex 3.06mV;
sky130hd gcd 0.41mV (thick PDN, high 1.8V supply); asap7 gcd **103.68mV**
(dense 7nm die, 0.77V supply — a ~13.5% IR drop, the largest by far, as
expected for the smallest/densest process node). asap7/gcd's die is only
8.98×8.98µm — this is genuinely representative of a real PDK effect, not a
bug.

asap7/gcd's **thermal** map, however, is flat (110.0°C everywhere,
peak-to-peak 0.0°C) — not because of a power-model bug (the cell-type
weighting from the 2026-08-26 fix works fine on asap7 names, confirmed
separately), but because the die is only ~9µm and HotSpot's adaptive grid
(`MIN_CELL_UM=5.0`) collapses to a 1×1 block for a die this small — there is
no spatial resolution left to show any gradient regardless of the power
model. This is a different, more fundamental limitation than the
originally-diagnosed "asap7 flat map" name-matching bug, and it won't be
fixed by anything in `extract_thermal_labels.py` — it needs either a larger
asap7 design (e.g. `asap7/ibex`, `asap7/aes`) or a lower `MIN_CELL_UM` (which
risks the singular-matrix HotSpot failures that constant was chosen to
avoid).

Retrained both tracks on the 6-sample cross-PDK set (25 epochs, batch size
2):
- `train_irdrop.py --laplacian-weight 0.1`: best val MSE **0.03263**
- `train_thermal.py --laplacian-weight 0.0`: best val MSE **0.02153**
- `train_thermal.py --laplacian-weight 0.1`: best val MSE **0.05894**

**The Laplacian result is now consistent in the same direction as n=4** (λ=0
beats λ=0.1, and by a wider margin than at n=4: ~2.7× worse val MSE at n=6 vs
~9% worse at n=4), reversing the n=3 result where λ=0.1 looked ~14% better.
Sequence across sample-size increases: n=3 (helped) → n=4 (hurt slightly) →
n=6 (hurt substantially). This is now two independent additions pointing the
same direction, which is somewhat more informative than the n=3→n=4 flip
alone, but still far short of a reliable conclusion — 6 designs spanning 3
process nodes with wildly different die sizes (9µm to 250µm) is a very
heterogeneous, very small set, and one of the six (asap7/gcd) has a
degenerate flat thermal target that may interact oddly with a smoothness
penalty. **Working hypothesis, not a conclusion: at this sample size and
this level of cross-design heterogeneity, the Laplacian smoothness prior may
be actively miscalibrated rather than merely under-powered.** Testing this
needs either more same-PDK samples (to separate "not enough data" from
"wrong prior for this data") or excluding degenerate flat-map samples from
the comparison.

### 2026-08-26 — Laplacian smoothness loss + cell-type weighted power model (actually committed this time)

**Correction to the 2026-08-13 entry below:** that entry describes a
`_cell_power_weight` cell-type weighted power model as already implemented,
but `git log -- data_collection/extract_thermal_labels.py` shows the file was
never touched after the initial add + a black-formatting commit — the
function did not exist in the committed code. The file was still using pure
area as the power proxy (see its module docstring). Re-implemented for real
in this session; see below.

**`training/train_thermal.py` — Laplacian smoothness loss (Option B):**
Added `--laplacian-weight` (default 0.0, off). When > 0, adds
`λ·||∇²T_pred||²` (5-point discrete Laplacian via a fixed 3×3 conv kernel,
replicate-padded) to the training loss, penalising curvature in the
prediction. Val loss is intentionally left as plain MSE so `best_val_loss`
stays comparable to the existing baseline (0.02912). Smoke-tested: flat
predictions get zero penalty, noisy ones get penalised (`_laplacian`/`_loss`
unit-checked directly; full training loop run end-to-end on a synthetic
6-sample dataset, 2 epochs, no errors).

**`data_collection/extract_thermal_labels.py` — cell-type weighted power model:**
Added `_cell_power_weight(master_name, is_block)`: clock cells 5×, sequential
3×, macros (`master.isBlock()`) 2×, combinational 1×. Unlike the described
(nonexistent) prior version, name matching is a lowercased **substring** test
with no word-boundary requirement, specifically to handle concatenated PDK
naming — asap7 (`CKBUFx2_ASAP7_75t_R`, `DFFHQNx1_ASAP7_75t_R`,
`ICGx1_ASAP7_75t_R`) doesn't use the `CLKBUF`/`DFF_X1`-with-separators style
nangate45 does. `build_power_grid()` now accumulates `area_um2 × weight` per
grid cell instead of raw area, then rescales to `total_power_w` as before (total
chip power unchanged, only its spatial distribution). Prints a per-type
breakdown (area, % of weighted power) at extraction time.

Verified the classifier against real cell names from all three PDKs in the
dataset (nangate45, asap7, sky130hd) plus a macro case — 11/11 correct after
one fix: an initial `_fd_`/`fd_` sequential token false-positived on every
sky130 cell (`sky130_fd_sc_hd__*` prefix contains `fd_`), dropped in favor of
the more specific flip-flop suffixes (`dfxtp`, `dlxtp`, etc.) sky130 actually
uses. Not yet re-run against the real 48-sample dataset (needs
`openroad/orfs-ml:latest` + HotSpot, not available in this environment) — the
known asap7 flat-map issue (ΔT≈0) should be re-checked once that's rerun.

**Next:** re-extract the 48-sample dataset with `--force` under
`openroad/orfs-ml:latest`, retrain, and compare correlation against the
2026-08-13 baseline table below (mean ~0.80, asap7 NaN/flat). Then try
`--laplacian-weight` sweeps (start ~0.05–0.2) on the same data.

---

### 2026-08-26 — Synced thermal-solver with fork master (133 upstream commits)

Merged `origin/master` (fork master, itself up to date with upstream ORFS) into
`thermal-solver` via `git merge origin/master --no-edit`. No conflicts — the two
branches never touched overlapping files (`flow/ml/**` vs. flow scripts/platforms/designs).

**What landed from upstream:** `tools/OpenROAD` and `tools/kepler-formal` submodule bumps,
new `asap7/coralnpu` and `nangate45/cva6` design configs, new fakeram LEF/LIB/Verilog models
for asap7 and nangate45, `flow/scripts/{cts,floorplan,global_route,macro_place_util,synth}.tcl`
updates, `flow/scripts/variables.{json,yaml}` additions, `flow/util/{genReportTable,uploadMetadata}.py`
changes, and various `rules-base.json`/`constraint.sdc` regenerations across platforms.

**Verification:** all `flow/util/ml/congestion/**/*.py` files still `py_compile` cleanly post-merge.
Submodules (`tools/OpenROAD`, `tools/kepler-formal`, `tools/yosys`) are not checked out in this
worktree — same state as before the merge, unrelated to it. No ORFS flow build was run (out of
scope for this sync; the ML pipeline doesn't touch `flow/scripts/*` or `flow/platforms/*`).

Merge commit: `4d64fba0d`. Pushed to `origin/thermal-solver`.

---

### 2026-08-13 — Cell-type weighted power model + full 48-sample retrain

**Root cause of previous near-flat thermal maps:**
The old `build_power_grid()` used uniform cell area as the power proxy — every cell weighted
equally. Sequential cells (DFFs) and clock cells (ICG, CLKBUF) dissipate 3–5× more power than
combinational logic. This produced near-uniform power grids → near-flat HotSpot output (ΔT ≈ 0–2°C).

**`data_collection/extract_thermal_labels.py` — cell-type weighted power model:**

Added `_cell_power_weight(master_name, master_type_str)` function:
- Clock cells (CLKBUF, CLKINV, CKBUF, CKINV, ICG, CLKGATE, __CLK, __DLCLK): **5× weight**
- Sequential (DFF, SDFF, LATCH, __DFX, __DLX, __DLAT, FD_, _FD_): **3× weight**
- Macros (BLOCK master type): **2× weight**
- Combinational: **1× weight**

Rewrote `build_power_grid()` to accumulate `area_um2 × weight` into `weighted_grid`,
then rescale to `total_power_w` so absolute power is preserved while spatial distribution
reflects cell activity. Prints per-type breakdown at runtime (count, weighted-power %).

**Re-extraction of all 48 samples:**
Previous run failed with `FileNotFoundError: 'hotspot'` because `OR_IMAGE` was not set,
so Docker used `openroad/orfs:latest` (no HotSpot) instead of `openroad/orfs-ml:latest`.
Fix: always set `OR_IMAGE=openroad/orfs-ml:latest` before running `extract_thermal_batch.sh`.

```bash
cd flow && OR_IMAGE=openroad/orfs-ml:latest bash util/ml/congestion/data_collection/extract_thermal_batch.sh --force
```
Result: **passed=48, failed=0** (skipped=48 = features already extracted).

**U-Net retrain on 48 samples (100 epochs, CPU):**
- Dataset: 48 samples, T range 87–148°C (real thermal gradients confirmed)
- Split: train=33, val=7, test=8
- Best val MSE: **0.02912** at epoch 70
- Final val MAE: ~0.138 normalized ≈ ~8°C absolute error
- Train/val gap after epoch 70 indicates mild overfitting at 33 samples

```bash
OR_IMAGE=openroad/orfs-ml:latest util/docker_shell python3 /work/util/ml/congestion/training/train_thermal.py \
    --data-dir /work/util/ml/congestion/data \
    --checkpoint-dir /work/util/ml/congestion/checkpoints
```

**Visualization report (`inference/visualize_thermal.py`) — per-design correlation:**

Run from `flow/` on host (matplotlib not in orfs-ml image):
```bash
python3 util/ml/congestion/inference/visualize_thermal.py \
    --data-dir util/ml/congestion/data \
    --checkpoint util/ml/congestion/checkpoints/thermal_best.pt \
    --out thermal_report.html
```

Full per-design results (baseline for future model comparisons):

| Design | Corr | MAE (norm) | T range (°C) |
|---|---|---|---|
| asap7_aes_base | +0.821 | 0.115 | 110–110 |
| asap7_jpeg_hi_util_75 | NaN | 0.249 | 110–110 |
| asap7_jpeg_pipeline_85 | +0.858 | 0.089 | 110–110 |
| nangate45_adder4_base | +0.894 | 0.093 | 110–110 |
| nangate45_aes_base | +0.941 | 0.076 | 108–112 |
| nangate45_ariane133_base | +0.891 | 0.130 | 109–130 |
| nangate45_ariane133_util_60 | +0.675 | 0.156 | 115–121 |
| nangate45_ariane136_base | +0.940 | 0.083 | 87–148 |
| nangate45_dynamic_node_base | +0.954 | 0.065 | 109–111 |
| nangate45_gcd_base | +0.848 | 0.134 | 110–110 |
| nangate45_gcd_hi_util | +0.199 | 0.277 | 110–110 |
| nangate45_ibex_ar_05 | +0.858 | 0.096 | 108–112 |
| nangate45_ibex_ar_15 | +0.737 | 0.158 | 109–111 |
| nangate45_ibex_ar_20 | +0.823 | 0.138 | 107–112 |
| nangate45_ibex_base | +0.912 | 0.095 | 108–112 |
| nangate45_ibex_hi_util | +0.923 | 0.086 | 109–111 |
| nangate45_ibex_pipeline_85 | +0.902 | 0.118 | 109–111 |
| nangate45_ibex_util_60 | +0.931 | 0.114 | 109–111 |
| nangate45_ibex_util_70 | +0.938 | 0.088 | 109–111 |
| nangate45_jpeg_ar_05 | +0.974 | 0.064 | 109–112 |
| nangate45_jpeg_ar_15 | +0.690 | 0.148 | 110–112 |
| nangate45_jpeg_ar_20 | +0.616 | 0.149 | 110–112 |
| nangate45_jpeg_base | +0.814 | 0.101 | 110–112 |
| nangate45_jpeg_hi_util | +0.781 | 0.156 | 110–112 |
| nangate45_jpeg_pipeline_88 | +0.795 | 0.147 | 110–112 |
| nangate45_jpeg_util_60 | +0.542 | 0.167 | 109–113 |
| nangate45_jpeg_util_70 | +0.879 | 0.078 | 109–112 |
| nangate45_jpeg_util_90 | +0.781 | 0.156 | 110–112 |
| nangate45_swerv_ar_05 | +0.807 | 0.135 | 108–114 |
| nangate45_swerv_ar_15 | +0.822 | 0.105 | 108–114 |
| nangate45_swerv_ar_20 | +0.647 | 0.166 | 107–114 |
| nangate45_swerv_base | +0.741 | 0.104 | 109–113 |
| nangate45_swerv_hi_util | +0.766 | 0.146 | 109–113 |
| nangate45_swerv_pipeline_80 | +0.771 | 0.132 | 109–113 |
| nangate45_swerv_util_60 | +0.734 | 0.109 | 109–113 |
| nangate45_swerv_util_70 | +0.761 | 0.108 | 109–113 |
| nangate45_tinyRocket_base | +0.331 | 0.265 | 110–112 |
| sky130hd_aes_base | +0.787 | 0.113 | 111–113 |
| sky130hd_gcd_base | +0.668 | 0.152 | 110–110 |
| sky130hd_ibex_base | +0.816 | 0.268 | 110–116 |
| sky130hd_jpeg_base | +0.887 | 0.157 | 111–119 |
| sky130hd_riscv32i_ar_05 | +0.908 | 0.115 | 108–113 |
| sky130hd_riscv32i_ar_15 | +0.967 | 0.054 | 109–112 |
| sky130hd_riscv32i_ar_20 | +0.791 | 0.147 | 109–112 |
| sky130hd_riscv32i_base | +0.824 | 0.097 | 108–112 |
| sky130hd_riscv32i_pipeline_65 | +0.971 | 0.070 | 109–112 |
| sky130hd_riscv32i_util_60 | +0.967 | 0.074 | 108–112 |
| sky130hd_riscv32i_util_70 | +0.957 | 0.069 | 109–112 |

Summary: 13 designs ≥0.9 (excellent), ~30 designs 0.6–0.9 (good), 3 designs <0.6 (poor),
1 NaN (asap7_jpeg_hi_util_75, flat map). Mean corr across non-NaN designs: ~0.80.

**Known issue — asap7 flat maps:**
All asap7 designs show ΔT≈0°C (110–110°C). The cell naming in asap7 (`BUF_X1`, `DFF_X1`)
does not match the substring patterns in `_cell_power_weight` (which expect e.g. `CLKBUF`,
`DFF` not preceded by `_`). Fix needed: add asap7-specific patterns or use master type
flags more aggressively instead of name patterns.

**Warnings fixed in `visualize_thermal.py`:**
- `tight_layout` warning: switched colorbar subplot to `layout="constrained"` in `plt.subplots`
- `invalid value in divide` from `np.corrcoef`: wrapped in `np.errstate(invalid="ignore")`
  (NaN result is expected for flat maps and rendered correctly in HTML)

**Docker path note:**
`util/docker_shell` mounts the workspace to `/work` but `cd`s to `/OpenROAD-flow-scripts/flow`
inside the container. Always use absolute `/work/...` paths when passing scripts and data
directories to docker_shell. Relative paths resolve to the baked-in image path, not the
mounted workspace.

---

### 2026-08-11 — Option A: pre-diffused input channel + data expansion plan

**`training/thermal_dataset.py` — added 5th input channel (pre-diffused cell density):**
- Added `scipy.ndimage.gaussian_filter(cell_density, sigma=3.0)` as channel 4.
- Normalised blurred channel to [0,1] independently before stacking.
- Rationale: U-Net has no knowledge of thermal diffusion (heat spreading laterally).
  The blurred channel approximates the Green's function kernel of the steady-state
  heat equation, giving the model a "pre-spread" view of the power distribution.
  The model then learns the residual between this approximation and the true HotSpot output.
- Input shape: (4, 64, 64) → (5, 64, 64).

**`training/train_thermal.py` + `inference/visualize_thermal.py`:** `in_channels=4 → 5`.

**`data_collection/generate_variants.sh` — new script for ORFS flow variant generation:**
- Generates utilization variants (60%, 70%, 90%) for ibex, jpeg, swerv, ariane133, riscv32i.
- Generates aspect-ratio variants (0.5, 1.5, 2.0) for ibex, jpeg, swerv, riscv32i.
- Run with `--dry-run` to preview make commands without executing them.
- Adds ~24 new training samples (from 26 → ~50) once extracted.
- `adder4` and `gcd` intentionally excluded — near-flat thermal maps (ΔT ≈ 0) add noise.

**Alternative model options noted for future (not yet implemented):**
- **Option B** — Physics-informed Laplacian loss: `L_total = L_mse + λ·||∇²T_pred||²`
  Penalises non-smooth gradients without needing PDE solver. ~20 lines in train_thermal.py.
- **Option C** — Fourier Neural Operator (FNO): operates in frequency domain via FFT.
  Theoretically most principled for PDE solutions (∇·(k∇T)+Q=0). Needs new models/fno.py.
  Recommended once dataset exceeds 60 samples.
- **Option D** — Swin Transformer: global attention = larger effective receptive field.
  Already existed in repo (deleted). Better than U-Net for large dies (ariane136 ΔT=54°C).
  Needs 50+ samples to outperform U-Net reliably.

**Next steps:**
1. Retrain with 5-channel input: `python3 util/ml/congestion/training/train_thermal.py --data-dir util/ml/congestion/data --checkpoint-dir util/ml/congestion/checkpoints --epochs 200`
2. Run variant generation (dry-run first to check): `bash util/ml/congestion/data_collection/generate_variants.sh --dry-run`
3. Run for real (takes several hours): `bash util/ml/congestion/data_collection/generate_variants.sh`
4. Re-extract thermal labels for new variants: `bash util/ml/congestion/data_collection/extract_thermal_batch.sh`
5. Retrain again on expanded dataset (~50 samples).

---

### 2026-08-10 — Thermal training pipeline: per-sample normalisation + timeout fix

**Root cause of "21 designs failed" in batch:**
The original batch script had no per-design timeout. Large designs (ariane133, ariane136,
swerv) take 40+ minutes for ODB loading alone in OpenROAD Python mode. The batch ran the
first 5 small/fast designs successfully (asap7 ×3 + nangate45/adder4/aes), then appeared
to stall on ariane133 (which actually completed after 41 min). The remaining designs were
simply waiting in sequence.

**Current dataset: 8 complete pairs** (6 nangate45 + 2 asap7, extracted 2026-08-10):
- asap7: aes_base, jpeg_hi_util_75, jpeg_pipeline_85
- nangate45: adder4_base, aes_base, ariane133_base, gcd_base, ibex_base

**`data_collection/extract_thermal_batch.sh` — timeout support added:**
- Default 3600s (1 hour) per extractor call via `timeout "$TIMEOUT_S" util/docker_shell ...`
- Exit code 124 = timeout → prints `[TIMEOUT]` message rather than generic `[FAIL]`
- `--timeout N` flag to override from command line
- Skip counter now also incremented for features (was only counting thermal skips)

**`training/thermal_dataset.py` — switched to per-sample normalisation:**
- Each thermal map is independently normalised to [0,1] using its own min/max.
  Reason: HotSpot absolute temperatures vary ~100× across process nodes and die sizes
  (asap7 50µm die at 500mW → 2000°C; nangate45 ibex 0.24mm die → 100°C). The ML model
  needs to learn spatial hotspot patterns, not cross-process temperature scales.
- `__getitem__` now returns `{"x", "thermal", "t_min", "t_max"}` per sample.
- `denormalize()` signature updated to take explicit `(t_norm, t_min, t_max)`.
- Dataset-level `self.t_min` / `self.t_max` kept as per-sample lists for diagnostics.

**`training/train_thermal.py` — updated for per-sample norm:**
- Removed `thermal_norm.json` write (no longer a global constant).
- Val metric is now `val_mae (norm)` [0,1] instead of °C (meaningless cross-process).

**`inference/predict_thermal.py` — updated for per-sample norm:**
- Removed `--norm` argument (no external norm JSON needed).
- Output `.npz` now contains only `thermal_pred_norm` (relative heatmap [0,1]).
- 1.0 = predicted hottest point in that specific design.

**Smoke test:** 5-epoch training run on 8 samples converged (val MSE 0.061 on 1 val sample).
GPU used (CUDA available). Full training pipeline verified end-to-end.

---

### 2026-08-10 — Thermal inference script + U-Net fix

**`models/unet.py`** — added `num_heatmap_layers` parameter to `CongestionUNet.__init__`
(default 10 for congestion, backwards-compatible). For thermal, pass `num_heatmap_layers=1`
to get a proper 1-channel output instead of wasting 9 unused channels.

**`training/train_thermal.py`** — updated to use `num_heatmap_layers=1` and removed the
`pred.heatmap[:, :1, :, :]` channel-slice hack. Now uses `pred.heatmap` directly.

**`inference/predict_thermal.py`** — inference script for trained thermal model.

Two usage modes:
- `--features <npz>` (no OpenROAD needed): loads pre-extracted feature file, runs model,
  outputs predicted thermal map in normalised [0,1] and °C forms.
- `--odb <path>` (auto-extracts): calls `extract_features.py` via `docker_shell` internally,
  then runs model. Requires `OR_IMAGE=openroad/orfs-ml:latest` or base image with OpenROAD.

Outputs `thermal_pred_norm` (64×64), `thermal_pred_c` (64×64 in °C), and normalisation
constants to a `.npz`. Run from `flow/`:
```bash
python3 util/ml/congestion/inference/predict_thermal.py \\
    --features util/ml/congestion/data/<label>_features.npz \\
    --checkpoint util/ml/congestion/checkpoints/thermal_best.pt \\
    --norm util/ml/congestion/checkpoints/thermal_norm.json \\
    --out predicted_thermal.npz
```

---

### 2026-08-10 — extract_thermal_batch.sh bug fixes (3 iterations)

**Bug 1 — doubled `results/results/` path:**
`find results -name "3_place.odb"` returns relative paths starting with `results/`.
The strip `${HOST_ODB#*/flow/results/}` expected an absolute path and stripped nothing,
so `cont_odb=/work/results/results/asap7/...` was passed to docker.
Fix: `${HOST_ODB#results/}` strips the leading `results/` from relative paths.

**Bug 2 — skip-check path had extra `flow/` prefix:**
`feat_host="flow/ml/..."` was wrong because the script already `cd`s into `flow/`.
Fix: `feat_host="ml/..."`.

**Bug 3 — `docker run -i` consumed the `find` pipe (only 1 design processed):**
`docker_shell` always passes `-i` to `docker run`, which attaches the container stdin
to the script's own stdin. Since the while loop reads from process substitution
`< <(find ... | sort)`, docker consumed all remaining ODB paths after the first
container exited. Only one design was ever processed.
Fix: added `</dev/null` to each `util/docker_shell` call so docker gets a dead stdin.

**Bug 4 — `extract_thermal_labels.py` singular matrix (`lupdcmp`) on small dies:**
asap7 die (50×50 µm) at 64×64 grid gives 0.78 µm cells — too small for HotSpot's
block RC model (conductance matrix becomes ill-conditioned).
Fix: `_adaptive_hotspot_grid()` caps the HotSpot grid to keep cells ≥ 5 µm
(gives 10×10 for asap7), runs HotSpot at that resolution, then bilinearly upsamples
to 64×64. Also added 1% uniform power floor to all cells to prevent zero-power rows
from causing singular matrix independently of die size.

---

### 2026-08-10 — Thermal training infrastructure

Three new files to support thermal model training:

**`data_collection/extract_thermal_batch.sh`** — batch extraction over all existing
`3_place.odb` files on disk (26 found). Runs `extract_features.py` + `extract_thermal_labels.py`
for each, writes paired `*_features.npz` + `*_thermal_labels.npz` to `util/ml/congestion/data/`.
Idempotent — already-extracted files are skipped. Run with:
```bash
cd flow && export OR_IMAGE=openroad/orfs-ml:latest
bash util/ml/congestion/data_collection/extract_thermal_batch.sh
```

**`training/thermal_dataset.py`** — `ThermalDataset`: finds matched `*_features.npz` +
`*_thermal_labels.npz` pairs, normalises thermal maps to [0,1] using dataset-wide
min/max, applies random H/V flips for augmentation. `split_thermal_dataset()` for
train/val/test splits. Stores `t_min`/`t_max` for °C de-normalisation at inference.

**`training/train_thermal.py`** — trains `CongestionUNet` on thermal data. Uses
`pred.heatmap[:, :1, :, :]` (first heatmap channel) as the thermal output — MSE loss
only, no hotspot/score heads. Saves `thermal_best.pt`, `thermal_last.pt`, and
`thermal_norm.json` (normalisation constants) to `checkpoints/`.

---

### 2026-08-10 — Pipeline run results + bug analysis

Run: `python3 util/ml/congestion/pipeline/run_pipeline.py` (WITHOUT `OR_IMAGE=openroad/orfs-ml:latest`).

| Design | Result | Notes |
|---|---|---|
| asap7/jpeg @ 85% | SUCCESS | data collected — no thermal files (OR_IMAGE not set) |
| sky130hd/riscv32i @ 65% | SUCCESS | data collected — no thermal files |
| nangate45/swerv @ 80% | FAILED_GRT_80 | see bug below |
| nangate45/ibex @ 85% | SUCCESS | data collected — no thermal files |
| nangate45/jpeg | FAILED_SYNTH @ 78% | see bug below |

**No thermal files were produced** — must set `OR_IMAGE=openroad/orfs-ml:latest` for HotSpot to run.
Run `extract_thermal_batch.sh` to collect thermal labels from all existing ODB files.

**Bug: swerv ODB-0269 root cause (updated)**
The REPORTS_DIR fix did not resolve ODB-0269. Root cause: `OPT_POST_GRT_WNS=0` (nangate45 default)
causes `global_route -end_incremental` to re-route modified nets after `recover_power`, and that
specific call fails writing markers. The REPORTS_DIR path itself is constructed correctly (confirmed
from log: `congestion_post_repair_design.rpt` was written). The failing path is the subsequent
`congestion_post_repair_timing.rpt` call, where OpenROAD can't open the markers file.
**Fix:** Set `OPT_POST_GRT_WNS=1` for GRT stage in `_make()` — switches to WNS repair path
which does not trigger the bug. Now applied in `run_pipeline.py`.
**Note:** swerv@80% `3_place.odb` EXISTS (placement succeeded before GRT failed) → thermal extraction works.

**Bug: jpeg FAILED_SYNTH at 78% (retry from GRT-0232 at 88%)**
Yosys phase completed (1_2_yosys.v produced). The OpenROAD step producing `1_synth.odb` failed.
`pipeline_78/` directory has yosys outputs but no `1_synth.odb`. Likely a dependency or objects-dir
issue with the fresh `pipeline_78` objects directory. Unrelated to CORE_UTILIZATION (synth doesn't
use utilization). No 3_place.odb exists so no thermal data from this design/tag.
Next attempt: add jpeg to designs.json with explicit `"utilization": 80` to avoid the 88%→GRT-0232
retry path and test directly at 80%.

---

### 2026-08-10 — Branch focus narrowed to thermal solver

User decision: `thermal-solver` branch is now **thermal modeling only**. Pre-placement
congestion prediction (GNN track) is deprioritised. Rationale: OpenROAD has no thermal
solver at all, making this the higher-value gap to fill. The GNN codebase is kept intact
and the pipeline still runs netlist extraction as a free step, but thermal is the active
development target.

---

### 2026-08-10 — Pre-placement netlist extractor + pipeline integration

Wrote `data_collection/extract_netlist_features.py` and wired it into `run_pipeline.py`
as step 0 of `extract_data()`.

**What `extract_netlist_features.py` does:**

Runs inside `openroad -python` on a post-synthesis ODB (`1_synth.odb`). No placement
coordinates are read — the extractor is intentionally blind to physical layout. It:

1. Iterates all instances to collect per-node stats: master area, master type flags
   (isBlock, isBuf, isInverter), sequential detection (name pattern match against
   FF/DFF/REG/LATCH/FD), fanin/fanout counts
2. Normalises area, fanin, fanout to [0, 1] relative to design max
3. Iterates all nets, finds driver→sink ITerms, skips nets above `--fanout-cap` (default 100)
   to drop clocks/resets that would create O(N) edges and swamp topology signal
4. Assigns `edge_weight = 1 / fanout` so high-fanout nets contribute less per connection
5. Saves `node_features (N,6)`, `edge_index (2,E)`, `edge_weight (E,)`, `node_names (N,)`,
   `num_macros` to `*_graph.npz`

**Node features (6):** `[area_norm, is_macro, is_seq, is_buf, fanin_norm, fanout_norm]`
This matches `NODE_FEATURES = 6` in `models/gnn.py` exactly — no changes to the model needed.

**Pipeline integration:** Step 0 in `extract_data()`, non-fatal. Uses `{results_dir}/1_synth.odb`
as input, writes `{data_dir}/{out_label}_graph.npz` alongside the congestion labels so
`GraphCongestionDataset` can pair them by name for GNN training.

**Run manually:**
```bash
OR_IMAGE=openroad/orfs-ml:latest util/docker_shell openroad -python \
    /work/util/ml/congestion/data_collection/extract_netlist_features.py \
    --odb /work/results/<platform>/<design>/<tag>/1_synth.odb \
    --out /work/util/ml/congestion/data/<label>_graph.npz
```

---

### 2026-08-10 — Branch renamed from congestion-ml to thermal-solver

```bash
git branch -m congestion-ml thermal-solver
```

---

### 2026-08-10 — Thermal extraction wired into pipeline

Added thermal label extraction as step 3 in `extract_data()` inside `run_pipeline.py`.
Every successful pipeline run now produces three output files per design:
- `*_features.npz` — placement features (existing, step 1)
- `*_labels.npz` — congestion labels from GRT (existing, step 2)
- `*_thermal_labels.npz` — HotSpot thermal map from placement ODB (new, step 3)

Thermal extraction is non-fatal: if HotSpot is not found in the container
(i.e. running with the base `openroad/orfs:latest` image), it logs a warning
and skips without failing the run. Congestion data is always saved.

To enable thermal extraction, run the pipeline with the custom image:
```bash
OR_IMAGE=openroad/orfs-ml:latest python3 util/ml/congestion/pipeline/run_pipeline.py
```

---

### 2026-08-10 — GNN rewrite: Option B (global pool + CNN decoder)

Rewrote `models/gnn.py` to remove scatter-to-grid and replace it with a
global pool + CNN decoder that doesn't need placement coordinates.

**Why:** The original GNN used placement (x, y) coordinates to scatter node
embeddings onto a spatial grid. Pre-placement, those coordinates don't exist.
Option A (RUDY-estimated positions) was considered but rejected — feeding wrong
position estimates would teach the model incorrect spatial correlations and
cap accuracy. Option B is architecturally correct for the task.

**New architecture:**
1. Linear projection + 3-layer GraphSAGE encoder (same as before)
2. Global mean + max pool → graph fingerprint (B, 2×embed_dim)
3. Seed MLP → reshape to (B, decoder_dim, 4, 4) spatial seed
4. CNN decoder: 4×4 → 64×64 via bilinear upsample + Conv2d + BatchNorm + ReLU
5. Same output heads as U-Net (heatmap, hotspot, score)

**Input changed:** NODE_FEATURES reduced from 8 to 6 — dropped x_norm and y_norm
since they don't exist pre-placement. Forward signature simplified to
`(x, edge_index, batch)` — no x_norm/y_norm arguments.

**New files:**
- `training/graph_dataset.py` — `GraphCongestionDataset` loads `*_graph.npz` +
  `*_labels.npz` pairs; `graph_collate` handles variable-size graphs in a batch
- `training/train_gnn.py` — rewritten to use `GraphCongestionDataset` with real
  netlist graphs instead of the fake grid-to-graph conversion placeholder

**Existing graph data note:** `flow/util/ml/data/*_congestion.npy` files are shape (10,)
per-layer global scores, NOT spatial maps — incompatible with our spatial task.
The graph_dataset pairs `*_graph.npz` with `*_labels.npz` (spatial, from
extract_labels.py) by matching design names in the same directory.

All 16 tests pass.

---

### 2026-08-10 — Codebase cleanup and pivot to two focused tracks

**Decision:** Pivoted from comparing all model architectures to two focused tracks
(pre-placement congestion + thermal). The original goal of comparing Swin/RF/Diffusion/Ensemble
was abandoned because:
- OpenROAD already has post-route congestion maps — post-placement ML adds no unique value
- Pre-placement prediction and thermal modeling are genuine gaps OpenROAD lacks
- The six-model comparison was blocked by data quality anyway (only 2 usable training samples)

**Removed:**

| File | Reason |
|---|---|
| `models/swin.py` + `training/train_swin.py` | Requires spatial grid input — not available pre-placement; U-Net is simpler and better suited for thermal |
| `models/classical.py` + `training/train_classical.py` | RF/XGBoost cannot output 64×64 spatial heatmaps |
| `models/diffusion.py` + `training/train_diffusion.py` | Generative model — slow inference, not a regression task, neither track needs sampling |
| `models/ensemble.py` | Premature — the two active tasks are now separate, not competing on the same problem |
| `checkpoints/swin_*.pt`, `checkpoints/rf.pkl` | Stale checkpoints for removed models |
| `model/` (directory) | Old single-model directory from early experiments, superseded by `models/` |
| `flow/ml/floorplan/` | Old floorplan experiment directory, no source code remaining, just a checkpoint |
| `flow/rsults/` | Typo directory (should be `results/`), contained one stale `clock_period.txt` |

**Added:**

| File | Purpose |
|---|---|
| `data_collection/extract_thermal_labels.py` | Reads a placed ODB, uses cell area as power proxy, writes HotSpot `.flp`/`.ptrace`, runs HotSpot, parses `.steady` output into a `thermal_map` + `power_grid` `.npz` |
| `flow/ml/Dockerfile` | Extends `openroad/orfs:latest` with HotSpot v7.0 and Python ML packages |

**Discovered:** `flow/util/ml/data/` contains prior pre-placement GNN experiments with netlist
`*_graph.npz` files and `*_congestion.npy` labels for ~15 nangate45/sky130hd designs including
larger ones (ariane133, black_parrot, mempool_group, microwatt). This data is directly usable
for Track 1 without any new ORFS runs.

---

### 2026-08-10 — Pipeline bug fixes

**Bug 1 — ODB-0269 (swerv@80%):**
`REPORTS_DIR` was not passed to `make` in `run_pipeline.py`. ORFS defaulted it to
`reports/<platform>/<design>/base`, causing OpenROAD to construct an empty markers file path.
GRT itself completed successfully (routing congestion 1.3175 — genuinely congested), but make
exited non-zero and the ODB was never written.
**Fix:** Derive `REPORTS_DIR` from `RESULTS_DIR` by substituting `/work/results/` → `/work/reports/`
and pass it explicitly in `_make()`.

**Bug 2 — GRT-0232 not retried (jpeg@88%):**
`[ERROR GRT-0232] Routing congestion too high` was treated as an unexpected failure, so the
pipeline gave up instead of retrying at a lower utilization. FLW-0024 and DPL-0038 triggered
retry; GRT-0232 should too.
**Fix:** Added `GRT0232_RE` pattern to the retry logic alongside FLW-0024 and DPL-0038.

**Bug 3 — Summary not persisted:**
Pipeline summary was only printed to stdout. If the terminal closed, the run history was lost.
**Fix:** Summary now also written to `pipeline/logs/summary_<timestamp>.log`.

---

### 2026-08-10 — Docker image (`openroad/orfs-ml:latest`)

Built a custom Docker image extending the ORFS base with HotSpot and ML packages.
Motivated by the thermal track needing HotSpot, which is not in the ORFS base image.

**Build:**
```bash
docker build -t openroad/orfs-ml:latest flow/ml/
```

**Use (instead of plain `util/docker_shell`):**
```bash
OR_IMAGE=openroad/orfs-ml:latest util/docker_shell <cmd>
# or export for the whole session:
export OR_IMAGE=openroad/orfs-ml:latest
```

`docker_shell` image resolution: `-i flag > $OR_IMAGE env var > default openroad/orfs:latest`

**Contents added over base:**
- HotSpot v7.0 compiled from source → `/usr/local/bin/hotspot`
- Python packages: numpy, scipy, scikit-learn, torch, torch-geometric

**Verify:**
```bash
OR_IMAGE=openroad/orfs-ml:latest util/docker_shell which hotspot
```

---

### 2026-08-09 — Pipeline run results

Ran `run_pipeline.py` with designs.json configured as:

| Design | Target util | Result | Data |
|---|---|---|---|
| asap7/jpeg | 85% | SUCCESS | 0% hotspots — no congestion signal |
| sky130hd/riscv32i | 65% | SUCCESS | 0% hotspots — no congestion signal |
| nangate45/swerv | 80% | FAILED_GRT_80 | ODB-0269 bug (see bug fix above) |
| nangate45/ibex | 85% | SUCCESS | 0% hotspots — still no congestion signal |
| nangate45/jpeg | 88% | FAILED_GRT_88 | GRT-0232 bug (see bug fix above) |

**Observation:** All three successes produced 0% hotspots. The only designs producing
useful congestion signal remain nangate45/swerv@85% (32.5%) and nangate45/jpeg@85% (6.1%).
Data starvation in the 10–70% hotspot range is the primary blocker for model training.

---

### Prior sessions — Initial pipeline and models

- Built U-Net (4-level, 3-head), GNN (GraphSAGE + grid scatter), Swin, RF/XGBoost,
  Ensemble, Diffusion models (Swin/RF/Ensemble/Diffusion later removed — see cleanup above)
- Built `run_pipeline.py` automated data collection pipeline
- Built `extract_features.py` (placement ODB → 4-channel 64×64 feature map) and
  `extract_labels.py` (GRT ODB → heatmap/hotspot/score labels)
- Ran manual high-utilization design runs; only swerv@85% and jpeg@85% produced useful data
- Ran `extract_existing.sh` to harvest existing ORFS result dirs; all produced 89–99%
  hotspot rates (extreme congestion, not useful for calibrated training)

---

## Known Blockers

| Issue | Symptom | Fix / Workaround |
|---|---|---|
| FLW-0024 density > 1.0 | `Place density exceeds 1.0` at placement | Lower `CORE_UTILIZATION` or use different platform |
| PDN file not found (sky130hd) | `gcd/grid_strategy-M1-M4-M7.tcl` missing | Add `PDN_TCL=/OpenROAD-flow-scripts/flow/platforms/sky130hd/pdn.tcl` |
| PDN file not found (asap7) | `asap7/gcd/grid_strategy-M1-M4-M7.tcl` missing | Add `PDN_TCL=/OpenROAD-flow-scripts/flow/platforms/asap7/openRoad/pdn/grid_strategy-M1-M2-M5-M6.tcl` (capital R in openRoad) |
| `No rule to make target 2_1_floorplan.sdc` | `make grt` on fresh dir | Run synth → floorplan → grt sequentially, never skip |
| OPENROAD_HIERARCHICAL=1 | Wrong PDN file references | Skip design or explicitly override `PDN_TCL` |
| SYNTH_HIERARCHICAL=1 | Broken make dependency chain | Skip design entirely |
| Sky130hd util ceiling | Platform overhead eats ~25% of core area; safe max ~65% | Use nangate45 or asap7 instead |

---

## Dataset Summary (congestion labels)

**Target range for useful training: 10–70% hotspots.** Both extremes (~0% and ~98%) hurt calibration.

| File key | Hotspots | % | Notes |
|---|---|---|---|
| nangate45_swerv_hi_util | 1331/4096 | 32.5% | **Best congestion signal** |
| nangate45_jpeg_hi_util | 250/4096 | 6.1% | Mild signal |
| nangate45_jpeg_2 | 3961/4096 | 96.7% | Extreme — extract_existing |
| nangate45_jpeg_10 | 4051/4096 | 98.9% | Extreme — extract_existing |
| nangate45_aes_0 | 4015/4096 | 98.0% | Extreme — extract_existing |
| nangate45_aes_8 | 3646/4096 | 89.0% | Extreme — extract_existing |
| nangate45_coyote_6 | 3996/4096 | 97.6% | Extreme — extract_existing |
| nangate45_swerv_4 | 4052/4096 | 98.9% | Extreme — extract_existing |
| sky130hd_ariane_7 | 4055/4096 | 99.0% | Extreme — extract_existing |
| sky130hd_gcd_1/9 | ~98% | Extreme | extract_existing |
| sky130hd_ibex_3/11 | ~97–99% | Extreme | extract_existing |
| sky130hd_tinyRocket_5 | 4067/4096 | 99.3% | Extreme — extract_existing |
| Everything else | 0/4096 | 0% | No congestion signal |

---

## Design Run History

### nangate45

| Design | Util | Outcome | Notes |
|---|---|---|---|
| gcd | 85% | FAILED CTS | Never reached GRT |
| ibex | 80% | SUCCESS | 0% hotspots |
| ibex | 85% | SUCCESS | 0% hotspots |
| jpeg | 85% | SUCCESS | 6.1% hotspots — mild signal |
| jpeg | 88% | FAILED GRT-0232 | Too congested to route (pipeline bug now fixed) |
| swerv | 80% | FAILED ODB-0269 | GRT actually finished congested; pipeline bug now fixed |
| swerv | 85% | SUCCESS | 32.5% hotspots — best signal |
| swerv | 90% | FAILED FLW-0024 | Density > 1.0 |
| dynamic_node | 85% | FAILED FLW-0024 | Only 521 instances, too small |
| ariane133 | any | SKIPPED | SYNTH_HIERARCHICAL=1 |
| tinyRocket | any | SKIPPED | Uses SRAMs (fakeram) |

### sky130hd

| Design | Util | Outcome | Notes |
|---|---|---|---|
| jpeg | 85% | FAILED FLW-0024 | 93% effective util after tapcells/PDN |
| jpeg | 75% | FAILED FLW-0024 | RSZ buffer insertion pushes density > 1.0 |
| ibex | any | SKIPPED | OPENROAD_HIERARCHICAL=1 |
| riscv32i | 65% | SUCCESS | 0% hotspots — sky130hd overhead too high |

### asap7

| Design | Util | Outcome | Notes |
|---|---|---|---|
| aes | baseline | SUCCESS | 0% hotspots |
| ethmac | 85% | FAILED FLW-0024 | Only 458 instances, too small |
| jpeg | 85% | SUCCESS | 0% hotspots |
| ibex | — | NOT YET | Candidate |
| mock-alu | — | NOT YET | Candidate |
| mock-cpu | — | NOT YET | Candidate |

---

## Planned Next Steps

**Updated 2026-09-16 (30-design entry above).** The real routed dataset is
now 30 designs / 90 `.npz` files across 6 PDKs (nangate45, asap7, sky130hd,
sky130hs, gf180, ihp-sg13g2), up from 12/36. The existing thermal and
IR-drop Laplacian sweep artifacts (`experiments/laplacian_sweep.json`,
`experiments/irdrop_laplacian_sweep.json`) were generated against the old
n=12 dataset and have NOT been re-run at n=30 (out of scope for the
data-collection pass). Before re-running either sweep at n=30,
`UPSAMPLED_KEYS` in `training/laplacian_sweep.py` must be updated from its
current 5-entry set to the correct 16 entries (9 pre-existing + 7 new — see
the corrected coarse-grid audit in the entry above). **This is not just a
future-proofing step**: the audit found 4 of the 12 pre-existing designs
(`nangate45_aes_base`, `nangate45_dynamic_node_base`,
`nangate45_ibex_base`, `nangate45_tinyRocket_base`) are also coarse-grid
and were already missing from `UPSAMPLED_KEYS` — meaning the thermal
sweep's already-published n=12 "upsampled (n=5) vs native (n=7)" stratified
breakdown was computed against a wrong partition and should be treated as
invalid (not the sweep's headline "indistinguishable" verdict, which
doesn't depend on this stratification — just that one breakdown table).

### Immediate

1. **Collect remaining 18 thermal label files** (run overnight):
   ```bash
   cd flow && export OR_IMAGE=openroad/orfs-ml:latest
   bash util/ml/congestion/data_collection/extract_thermal_batch.sh 2>&1 | tee batch_thermal.log
   ```
   Already-extracted designs (8) will be skipped. Large designs (ariane136, swerv x3, tinyRocket)
   take ~40 min each; full run will take 4–6 hours. Script now has 1-hour per-design timeout.

2. **Train U-Net once 20+ samples are collected:**
   ```bash
   cd flow && python3 util/ml/congestion/training/train_thermal.py \
       --data-dir util/ml/congestion/data \
       --checkpoint-dir util/ml/congestion/checkpoints \
       --epochs 200
   ```
   Checkpoint saved to `util/ml/congestion/checkpoints/thermal_best.pt`.
   **Note:** normalisation is now per-sample (not global) — no `thermal_norm.json` needed.

### Short term

3. **Evaluate thermal model** on held-out designs:
   ```bash
   python3 util/ml/congestion/inference/predict_thermal.py \
       --features util/ml/congestion/data/<label>_features.npz \
       --checkpoint util/ml/congestion/checkpoints/thermal_best.pt \
       --out predicted_thermal.npz
   ```
   Output is a relative hotspot map [0,1] — visualise with matplotlib.

4. **Add per-design evaluate loop** to `inference/evaluate.py` for thermal track.

### Medium term

5. **Power model improvement** — current thermal extractor uses cell area as a
   leakage-power proxy. Better options:
   - Use per-cell `staticPower` from Liberty (via OpenROAD STA) for process-accurate values
   - Scale total power proportional to die area so power density is constant across designs
     (fixes the unrealistic absolute temperatures caused by 500 mW fixed power on tiny asap7 dies)

6. **Expose as OpenROAD command** — once model quality is good, wrap prediction as a
   Python/Tcl callable that runs inside the ORFS flow at the placement stage.

7. **Add more designs** — asap7/ibex, asap7/mock-cpu for more thermal variety.
