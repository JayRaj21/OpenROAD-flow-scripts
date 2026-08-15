# P&R Extension — Development Log

Branch: `pr-extension`  
Started: 2026-08-14  
Author: JayRaj21

This file is the canonical dev log for the `pr-extension` branch.
Every decision, change, and next step is recorded here so work can resume
from a cold start without losing context. Update it after every session.

---

## Branch Context

`pr-extension` carries all the ML prediction work from the earlier `thermal-solver` /
`congestion-ml` branch (see commit history below), plus new P&R stage augmentation work.

### Inherited commits (ML pipeline — do not re-do)

| Commit | Date | Summary |
|---|---|---|
| `e1a1f93` | 2026-08-13 | thermal: replace uniform power model with cell-type-weighted model |
| `a875c09` | 2026-08-12 | gitignore: exclude ML training logs and pipeline log directory |
| `fe37faf` | 2026-08-11 | Add thermal prediction pipeline: HotSpot U-Net, dataset builder, variant generator |
| `1261b7e` | 2026-08-06 | Fix extraction scripts: wrong OpenROAD Python API usage |
| `3b569d0` | 2026-08-06 | Add run_pipeline.sh and extract_existing.sh |
| `c6ee9e9` | 2026-08-06 | Add test suite and fix Swin LayerNorm shape bug |
| `a6286ac` | 2026-08-06 | Add Swin, RF/XGBoost, Ensemble, Diffusion congestion models |
| `10cec03` | 2026-08-06 | Add congestion ML pipeline from scratch (U-Net + GNN, 3 output heads) |

The ML work is fully documented in `flow/ml/congestion/DESIGN_RUNS.md`.
Do not duplicate that content here — read it for ML context.

---

## New Direction: P&R Stage Augmentation

### Decision (2026-08-14)

Goal: augment the Place & Route flow to demonstrate understanding of how P&R works.
Constraint: do not rewrite core algorithms (placement/routing engines).
Approach: add tooling that sits *around* the existing stages — analysis, feedback, and
post-processing — using OpenROAD's Tcl/Python APIs and ORFS hook points.

### Options evaluated (2026-08-14)

Four directions were considered:

| # | Option | Demonstrates | Complexity |
|---|---|---|---|
| 1 | **Timing-driven post-placement cell perturbation** | placement ↔ timing feedback loop | Medium |
| 2 | **Congestion-feedback floorplan parameter tuner** | ML integration into flow | Medium — needs trained model |
| 3 | **Stage-by-stage quality metric aggregator** | quality trajectory across P&R | Low |
| 4 | **CTS skew analysis and buffer profiler** | CTS internals | Medium |

**Decision: start with option 1 (timing-driven perturbation), with option 3 as a
supporting diagnostic layer.**

Rationale:
- Option 1 has a clear success metric (improved WNS/TNS) and directly demonstrates
  the most fundamental P&R trade-off: placement quality drives timing closure.
- Option 3 is lightweight and makes the results of option 1 visible — shows before/after
  HPWL, WNS, TNS, congestion overflow at each checkpoint.
- Option 2 requires a trained congestion model; blocked until ML data collection is done.
- Option 4 is interesting but CTS is self-contained — less central than timing feedback.

---

## Codebase Map

```
flow/
├── scripts/
│   ├── global_place.tcl        # Stage 3_1: global placement
│   ├── detail_place.tcl        # Stage 3_5: detail placement
│   ├── cts.tcl                 # Stage 4_1: clock tree synthesis
│   ├── global_route.tcl        # Stage 5_1: global routing
│   ├── detail_route.tcl        # Stage 5_2: detail routing
│   └── final_report.tcl        # Stage 6: metrics collection
├── ml/
│   └── congestion/             # All ML work (see DESIGN_RUNS.md)
│       ├── data_collection/
│       ├── models/
│       ├── training/
│       └── inference/
└── PR_EXTENSION_DEV_LOG.md     # This file
```

P&R stage checkpoints written to `results/<platform>/<design>/<tag>/`:

| File | Stage | Contents |
|---|---|---|
| `3_1_place.odb` | Global placement | Cell positions (not legalised) |
| `3_5_place.odb` | Detail placement | Legalised, optimised placement |
| `4_1_cts.odb` | Post-CTS | Clock buffers inserted, timing propagated |
| `5_1_grt.odb` | Global routing | Route topology without geometry |
| `5_2_route.odb` | Detail routing | Full geometry |
| `6_final.odb` | Final | Sign-off ready |

---

## Implementation Plan

### Phase 1 — Stage-by-stage metric aggregator (option 3) ✓ DONE

**File: `flow/util/pr_metrics.py`**

Parses existing ORFS `.rpt` and `.log` files (no OpenROAD process needed) and prints
a stage-by-stage table of WNS, TNS, worst slack, Fmax, HPWL, GRT overflow, and power.

**Status: complete. Tested on nangate45/ibex/base and nangate45/adder4/base.**

### Phase 2 — Timing-driven post-placement cell perturbation (option 1) ✓ DONE

**File: `flow/scripts/post_cts_timing_repair.tcl`**

Tcl hook sourced at the `POST_CTS` point inside `cts.tcl`. Runs inside the live
OpenROAD session so all STA and ODB APIs are available.

**Algorithm:**
1. Read WNS via `sta::worst_slack -max`. Exit early if no setup violations.
2. Capture `report_timing -path_count 10 -path_delay max` to a string using `redirect -string`.
3. Parse instance/cell pairs from the timing report using regex on the pin lines.
4. Build an upsize map dynamically from loaded libraries: `TYPE_X<N> → TYPE_X<2N>`.
5. For each unique instance on a critical path (excluding DFFs and clock cells):
   - Call `$inst swapMaster $new_master` via ODB to replace the master in-place.
6. After all swaps: run `detailed_placement` to re-legalise (widths changed), then
   `estimate_parasitics -placement` to update wire models.
7. Report before/after WNS.

**Hook variable:** `POST_CTS_TCL` (not `POST_CTS` — discovered from `util.tcl:source_step_tcl`).

**Wiring it in (per design or globally):**
```makefile
# In flow/designs/<platform>/<design>/config.mk:
export POST_CTS_TCL = $(SCRIPTS_DIR)/post_cts_timing_repair.tcl
```

**Key design decisions:**
- DFFs and clock cells are excluded — swapping them changes hold/setup arcs and
  disturbs the CTS-balanced clock tree.
- Upsize map built from loaded libs at runtime, not hardcoded — works for any PDK
  following the `_X<N>` convention.
- Re-legalisation is run once after all swaps, not per-swap, to avoid redundant work.
- `redirect -string` used to capture timing report without writing a temp file.

**Status: complete and verified end-to-end on nangate45/ibex/base.**

**First successful run (2026-08-14):**
```
INFO [pctr] WNS -0.007 ns — starting cell upsizing on critical paths.
INFO [pctr] Upsize map: 84 candidate transitions loaded.
INFO [pctr]   swapped  _27049_  AND2_X1 -> AND2_X2
INFO [pctr] 1 cell(s) upsized, 0 skipped.
INFO [pctr] WNS: -0.007 ns -> -0.004 ns  (delta +0.003 ns)
```
ODB SHA changed (161557dd vs 76a0e40c baseline), confirming the hook made real design modifications.

---

## API Notes (OpenROAD version in orfs:latest as of 2026-08-14)

The following STA/ODB Tcl API calls were discovered during debugging:

| Call | Status | Notes |
|---|---|---|
| `sta::worst_slack -max` | ✓ works | Returns float |
| `find_timing_paths -path_delay max -sort_by_slack` | ✓ works | Returns list of PathEnd objects |
| `find_timing_paths -path_count N ...` | ✗ not supported | Use `lrange` on result instead |
| `[$path_end path]` | ✓ works | Returns Path object |
| `[$path pin]` | ✓ works | Returns OpenSTA Pin* |
| `[get_full_name $sta_pin]` | ✓ works | Returns "inst_name/port" string |
| `[$path prevPath]` | ✓ works | Returns previous Path* or NULL |
| `redirect -string { ... }` | ✗ not available | Not defined in this build |
| `redirect $file { ... }` | ✗ not available | Not defined in this build |
| `sta::report_path_string` | ✗ not available | Not defined in this build |
| `$block findInst $name` | ✓ works | ODB lookup by instance name |
| `$inst swapMaster $master` | ✓ works | ODB in-place cell swap |
| `detailed_placement` | ✓ works | Re-legalises after width changes |
| `estimate_parasitics -placement` | ✓ works | Wire model update |

---

## Known ORFS Hook Points

ORFS supports pre/post hooks for each stage via variables:
```
PRE_GLOBAL_PLACE / POST_GLOBAL_PLACE
PRE_DETAIL_PLACE / POST_DETAIL_PLACE
PRE_CTS / POST_CTS
PRE_GLOBAL_ROUTE / POST_GLOBAL_ROUTE
PRE_DETAIL_ROUTE / POST_DETAIL_ROUTE
```

Set in design config or `Makefile`:
```makefile
export POST_CTS = $(SCRIPTS_DIR)/my_post_cts_hook.tcl
```

The hook is sourced inside the OpenROAD session that already has the ODB loaded,
so all `odb`, `sta`, `grt`, `dpl` commands are available.

---

## Session Log

### 2026-08-14 — Session start, direction set

- Reviewed P&R stage scripts: `global_route.tcl`, `detail_place.tcl`, `cts.tcl`.
- Reviewed existing ML work in `flow/ml/congestion/`.
- Evaluated four augmentation directions (documented above).
- Decision: deterministic augmentation only — no ML in the P&R extension.
  Rationale: ML earns its place where EDA tools have no answer (thermal, pre-placement
  congestion). For post-CTS timing repair, OpenROAD's STA already has exact ground truth;
  using ML there would replace a precise answer with an approximation.
- Decision: implement metric aggregator (phase 1) then timing perturbation (phase 2).

---

### 2026-08-14 — Phase 1 complete: stage-by-stage metric aggregator

**New file: `flow/util/pr_metrics.py`**

Standalone Python script (no OpenROAD required) that parses existing ORFS report and log
files and prints a stage-by-stage quality trajectory table.

**Metrics collected per stage:**

| Metric | Source | Stage(s) |
|---|---|---|
| WNS (worst negative slack, ns) | `<stage>.rpt` | all |
| TNS (total negative slack, ns) | `<stage>.rpt` | all |
| Worst slack (ns) | `<stage>.rpt` | all |
| Fmax (MHz) | `<stage>.rpt` | all |
| HPWL (half-perimeter wirelength, µm) | `3_3_place_gp.log` | global place |
| GRT overflow | `3_3_place_gp.log`, `5_1_grt.log` | global place, global route |
| Total power (W) | `<stage>.rpt` | last available |

**Usage:**
```bash
# From repo root:
python3 flow/util/pr_metrics.py --platform nangate45 --design ibex --tag base
python3 flow/util/pr_metrics.py --platform nangate45 --design adder4 --tag base

# With explicit paths:
python3 flow/util/pr_metrics.py \
    --reports-dir flow/reports/nangate45/ibex/base \
    --logs-dir    flow/logs/nangate45/ibex/base
```

**Example output (ibex/base — timing-stressed design):**
```
Stage            WNS (ns)   TNS (ns)  Worst slack  Fmax (MHz)    HPWL (um)  GRT overflow
Global place       +0.000     +0.000       +0.020       459.4  331,831,045        1.2875
Resizer            +0.000     +0.000       +0.020       459.4            —             —
Detail place       -0.030     -1.430       -0.030       448.9            —             —
CTS                -0.000     -0.000       -0.000       454.5            —             —
Global route       -0.260    -81.060       -0.260       406.9            —             —
Finish             +0.000     +0.000       +0.000       455.5            —             —
Total power (post-route):  3.1700e-02 W
```

This makes the timing degradation at each stage visible — detail placement introduces hold
violations, global route reveals setup violations at real wire parasitics, final sign-off
recovers them. This baseline is needed to measure the impact of the phase 2 cell swapping hook.

---

---

### 2026-08-14 — Phase 2 verified end-to-end

After resolving several API incompatibilities in the orfs:latest Docker image
(no `redirect`, no `-path_count` flag on `find_timing_paths`, no
`sta::report_path_string`), the hook was rewritten to use direct path object
traversal: `[$path_end path]` → `[$path pin]` → `get_full_name` → ODB lookup.

**Key mechanics confirmed working:**
- `find_timing_paths -path_delay max -sort_by_slack` returns path end objects
- `[$path_end path]` / `[$path prevPath]` traverses path backwards
- `get_full_name [$path pin]` gives "inst_name/port" which we split to get inst name
- `$block findInst $name` converts name to ODB dbInst*
- `$inst swapMaster $new_master` swaps the cell in-place
- `detailed_placement` re-legalises cleanly (0 displacement after upsize)
- `estimate_parasitics -placement` updates wire models

**Result on ibex/base:** 1 cell swapped (AND2_X1 → AND2_X2), WNS -0.007 → -0.004 ns.
Only 1 cell found because repair_timing had already upsized most candidates; the
remaining violation was in a deep path with limited upsize opportunity.

**How to run the hook:**
```bash
rm results/nangate45/ibex/base/4_1_cts.odb      # force rebuild
util/docker_shell make cts \
    DESIGN_CONFIG=designs/nangate45/ibex/config.mk \
    POST_CTS_TCL=/work/scripts/post_cts_timing_repair.tcl
```

---

### 2026-08-14 — Controlled before/after comparison

Ran `util/compare_hook.sh` to compare the full flow from the same `3_place.odb`
checkpoint, with and without the POST_CTS hook. Stage 1–4 numbers were identical
in both runs, confirming a clean controlled comparison.

**Results (`nangate45/ibex/base`):**

```
Stage            Baseline WNS   Hook WNS   Delta
Global place       +0.000         +0.000     —
Resizer            +0.000         +0.000     —
Detail place       +0.000         +0.000     —
CTS                -0.010         -0.010     —    (report written before hook runs)
Global route       -0.020         -0.000   +0.020 ns  ← key improvement
Finish             +0.000         +0.000     —
```

**Global route TNS:** -0.110 ns (baseline) → -0.000 ns (hook)  
**Global route Fmax:** 451.4 MHz (baseline) → 454.0 MHz (hook, +2.6 MHz)  
**Total power:** identical at 3.17e-02 W — upsize did not measurably increase power.

**Interpretation:**  
The hook's 3 ps improvement at CTS (AND2_X1 → AND2_X2 swap on `_27049_`) translated
into 20 ps of recovered slack at global route, eliminating all setup violations before
detail route ran. The gain amplified because upsizing reduces gate delay across the
cell's entire fanout cone; when real wire parasitics were added at global route, the
baseline was marginal enough to be pushed into violation while the hook version had
just enough headroom to absorb them. Both designs closed timing at finish, but the
hook version arrived at detail route with a cleaner slate.

**Tooling added:** `util/compare_hook.sh` — runs both flows and prints tables
back-to-back for repeatable before/after comparison.

---

---

### 2026-08-14 — Phase 2 upgraded: iterative upsizing

**Changed: `flow/scripts/post_cts_timing_repair.tcl`**

The single-pass `run` proc was refactored into an iterative loop:

**New structure:**

- `collect_candidates upsize_arr path_count seen_arr` — finds new upsize candidates
  on the N worst paths, skipping instances already swapped in prior iterations.
- `apply_swaps candidates` — applies ODB swaps, returns {swap_count skip_count}.
- `run {path_count 10} {max_swaps 30} {max_iters 5}` — outer loop:
  1. Collect candidates (deduped via `seen` array)
  2. Apply swaps
  3. Re-legalise + re-estimate parasitics
  4. Re-run STA; stop if WNS ≥ 0, no candidates, or no swaps applied
  5. Repeat up to `max_iters` times

**Why iterative matters:**  
A single pass swaps cells on the *current* critical paths. After those swaps +
re-legalisation, the critical paths may change — a previously non-critical path may
become the new worst path. Each iteration finds new candidates on the updated critical
paths, so the hook converges rather than leaving residual violations untouched.

**The `seen` array spans all iterations**, so a cell that was upsized in iteration 1
(e.g., `AND2_X1 → AND2_X2`) is not considered again in iteration 2 — it is already at
the higher drive level and would need a second upsize (`AND2_X2 → AND2_X4`) to improve
further. This is intentional: one upsize per cell per hook invocation keeps the area
budget predictable.

**Signature unchanged** — still invoked as `pctr::run` with no arguments for the
standard 10-path / 30-swap / 5-iteration defaults.

---

---

### 2026-08-14 — aes comparison revealed post-GRT hook gap; added post_grt_timing_repair.tcl

**Observation from aes comparison:**  
The post-CTS hook correctly skipped on `aes` because CTS timing was met (WNS +0.000).  
The violations in aes (-0.020 at global route, -0.010 at finish) only appear once real
wire parasitics are loaded by the GRT step — they are invisible at CTS time.

**New file: `flow/scripts/post_grt_timing_repair.tcl`**

Same iterative upsizing algorithm as `post_cts_timing_repair.tcl`, wired to the
`POST_GLOBAL_ROUTE_TCL` hook point. One critical difference in the parasitic
re-estimation step:

| Hook | Parasitic call after swaps | Why |
|---|---|---|
| post_cts_timing_repair.tcl | `estimate_parasitics -placement` | GRT not yet run |
| post_grt_timing_repair.tcl | `estimate_parasitics -global_routing` | GRT topology available |

Using `-global_routing` means the hook's STA reflects the actual route topology, so
the WNS reported inside the hook matches the global route report — no artificial
optimism from placement-only estimates.

Namespace: `pgtr` (vs `pctr` for the CTS hook) to avoid name collisions when both
hooks are active in the same session.

**Updated: `flow/util/compare_hook.sh`**

Now accepts both hooks simultaneously by default (CTS + GRT). Flags to disable either:
```bash
# Both hooks (default)
util/compare_hook.sh --platform nangate45 --design aes

# CTS hook only
util/compare_hook.sh --platform nangate45 --design aes --no-grt-hook

# GRT hook only
util/compare_hook.sh --platform nangate45 --design aes --no-cts-hook
```

---

### 2026-08-14 — aes comparison shows post-GRT hook is redundant; architectural insight

**Result:** aes baseline and hook numbers are identical. The GRT hook does not improve timing.

**Root cause — reading `flow/scripts/global_route.tcl`:**

The `POST_GLOBAL_ROUTE_TCL` hook fires at line 151, *after* all of the following have
already run:
1. `global_route` — builds the routing topology
2. `estimate_parasitics -global_routing` — loads real wire RC
3. `repair_design_helper` — fixes max-cap/max-slew violations
4. `repair_timing_helper` — fixes setup/hold with gate sizing, buffer insertion, cell swapping
5. Another `estimate_parasitics -global_routing`
6. `report_metrics 5 "global route"` — writes `5_global_route.rpt`
7. ← **Our hook fires here**

ORFS's built-in `repair_timing` at step 4 is far more capable than our simple upsizing
(it does buffer insertion, VT swaps, and multi-objective repair). By the time our hook
runs, there are few or no candidates left.

**Why ibex worked but aes does not:**  
The post-CTS hook fires *before* the GRT repair. Our upsize reduces gate delay on the
critical path, which becomes the starting point for GRT repair to refine further. That
compounding effect is what eliminated ibex's violations entirely.

For aes, CTS timing is met (+0.000 WNS), so the post-CTS hook correctly skips.
The violations at global route (-0.020 WNS) appear when real parasitics are loaded,
but the built-in GRT repair partially addresses them. The residual violations at finish
(-0.010 WNS) are introduced by *detail routing* — the actual wire geometry after DRC-
legal routing differs from the GRT topology estimate. Post-detail-route violations require
an ECO (Engineering Change Order) flow, not simple cell upsizing.

**Conclusion:**  
The post-GRT hook is architecturally redundant with ORFS's built-in repair. The
post-CTS hook is the correct intervention point: it runs before the built-in GRT repair,
so improvements compound rather than compete.

`post_grt_timing_repair.tcl` is kept for completeness and as a documented dead-end
that explains *why* the post-CTS hook is the right intervention point.

---

## Planned Next Steps

1. ~~Implement `pr_metrics.py`~~ ✓ done
2. ~~Implement `post_cts_timing_repair.tcl` — single-pass~~ ✓ done
3. ~~Controlled before/after comparison on ibex~~ ✓ done
4. ~~Make hook iterative~~ ✓ done
5. ~~Add post-GRT hook — tested, found redundant with built-in repair~~ ✓ done (documented)
6. (Blocked on ML data) Congestion-feedback parameter tuner.
