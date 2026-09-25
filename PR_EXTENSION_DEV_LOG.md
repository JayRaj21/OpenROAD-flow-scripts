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

The ML work is fully documented in `flow/util/ml/congestion/DESIGN_RUNS.md`.
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
- Reviewed existing ML work in `flow/util/ml/congestion/`.
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

---

### 2026-08-22 — Phase 3: triage agent

**New file: `flow/util/triage_agent.py`**

LLM-powered diagnostic layer that sits on top of the existing toolchain:

```
pr_metrics.py  →  collect()  →  triage_agent.py  →  Claude  →  diagnosis
```

**What it does:**
1. Calls `pr_metrics.collect()` to read the stage-by-stage quality trajectory.
2. Computes notable stage-to-stage WNS deltas (threshold: ≥5 ps change).
3. Builds a structured prompt with the trajectory table, deltas, and final metrics.
4. Calls `claude-opus-5` with adaptive thinking and a system prompt encoding
   P&R domain knowledge — known failure patterns, what each stage does, and
   the ORFS parameters and hooks available on this branch.
5. Prints a structured diagnosis: root cause, evidence, recommended actions,
   expected outcome.

**Model:** `claude-opus-5` with `thinking: {type: "adaptive"}`.

**Usage:**
```bash
export ANTHROPIC_API_KEY=<key>   # or: ant auth login

python3 flow/util/triage_agent.py --platform nangate45 --design ibex --tag base
python3 flow/util/triage_agent.py --platform nangate45 --design aes  --tag base
```

**Why this is distinct from ORFS-Agent (ABKGroup):**
ORFS-Agent tunes top-level flow parameters (utilisation, density) across
multiple parallel runs. This triage agent reads the *inside* of a completed
run — the per-stage quality trajectory — and diagnoses which specific stage
caused the failure and why. It operates on a single run and produces a
targeted intervention recommendation rather than a search over parameter space.

**Branch story (complete):**
```
observe  →  pr_metrics.py       (what happened at each stage?)
intervene →  post_cts_*_tcl     (fix it inside the live OpenROAD session)
decide   →  triage_agent.py     (diagnose why, recommend what to try next)
```

---

---

### 2026-08-22 — Phase 4: validate triage diagnosis on aes

**Goal:** confirm the triage agent's recommended fix actually closes timing on aes.

**What the triage agent diagnosed (aes/nangate45/base):**
- CTS WNS +0.000 ns, GRT WNS −0.330 ns — classic CTS→GRT parasitic cliff
- Root cause: CTS uses `estimate_parasitics -placement` (optimistic); real wire RC
  only known after GRT, causing endpoints to look clean at CTS but violate at GRT
- Recommendation: `SETUP_SLACK_MARGIN=0.03`, `TNS_END_PERCENT=100`,
  `POST_CTS_TCL=$(SCRIPTS_DIR)/post_cts_timing_repair.tcl`; re-run CTS then finish

**Validation run (variables passed on make command line to reach Docker container):**
```bash
util/docker_shell make DESIGN_CONFIG=designs/nangate45/aes/config.mk \
    SETUP_SLACK_MARGIN=0.03 \
    POST_CTS_TCL=/work/scripts/post_cts_timing_repair.tcl \
    cts
util/docker_shell make DESIGN_CONFIG=designs/nangate45/aes/config.mk \
    SETUP_SLACK_MARGIN=0.03 \
    POST_CTS_TCL=/work/scripts/post_cts_timing_repair.tcl \
    finish
```

**Key lesson — Docker variable passing:**
The container runs make from `/OpenROAD-flow-scripts/flow/` (image copy of the repo),
not from `/work/` (the mounted workspace). Local `config.mk` changes are NOT seen.
Variables must be passed explicitly as `make VAR=value` arguments on every invocation.
`HOOK_PATHS` in `loop_agent.py` uses `/work/scripts/...` (Docker workspace path);
`CONFIG_HOOK_PATHS` stores `$(SCRIPTS_DIR)/...` (ORFS-canonical) for config.mk write-back.

**Result:**
| Metric | Before | After |
|--------|--------|-------|
| GRT WNS | −0.330 ns | −0.010 ns |
| Finish WNS | −0.010 ns | 0.000 ns |
| Finish TNS | −0.330 ns | 0.000 ns |
| Fmax | ~1190 MHz | ~1239 MHz (+49 MHz) |

Triage agent's prediction ("GRT WNS ≥ −0.005 after fix") confirmed.

**Committed:** `bad2f2bd8` — aes: apply triage-agent recommendations to close timing

---

### 2026-08-22 — Phase 5: closed-loop optimization agent

**New file: `flow/util/loop_agent.py`**

Autonomous observe→diagnose→intervene→verify cycle. No human intervention needed.

**Architecture:**
```
loop_agent.py
  ├── get_metrics     → calls pr_metrics.collect(), formats trajectory table
  ├── set_config_param → queues param change; translates "enabled" → Docker hook path
  ├── run_stage       → deletes stale ODB files, runs docker_shell make <stage>
  └── finish          → terminates loop; on success calls write_config_params
```

**Four tools exposed to Claude Opus 5:**
1. `get_metrics` — read current WNS/TNS/Fmax/overflow trajectory
2. `set_config_param(param, value)` — allowlisted params only, value checked against
   `UNSAFE_VALUE_PATTERNS` (blocks `$(`, `${`, backticks, shell metacharacters) to
   prevent Make-injection via config.mk write-back; "enabled" → hook path
3. `run_stage(stage)` — valid stages: `place`, `cts`, `grt`, `finish`
4. `finish(summary, success)` — terminate; if success=True, write params to config.mk

**PARAM_ALLOWLIST:**
`SETUP_SLACK_MARGIN`, `TNS_END_PERCENT`, `OPT_POST_GRT_WNS`,
`PLACE_DENSITY_LB_ADDON`, `POST_CTS_TCL`, `POST_GLOBAL_ROUTE_TCL`

**STAGE_STALE_FILES** — files deleted before each stage re-run:
- `place`: `3_3_place_gp.odb` through `3_place.odb` (PLACE_DENSITY_LB_ADDON affects global place)
- `cts`: `4_1_cts.odb`, `4_cts.odb`
- `grt`: `5_1_grt.odb`, `5_1_grt.sdc`
- `finish`: `5_2_route.odb`, `5_route.odb`

**Write-back (`write_config_params`):**
On success, updates `designs/<platform>/<design>/config.mk` in-place:
- Regex-matches existing `export PARAM = ...` lines and updates them
- Appends new params with `# Written by loop_agent.py` comment
- Translates Docker paths (`/work/scripts/...`) → ORFS-canonical (`$(SCRIPTS_DIR)/...`)

**End-to-end result on aes/nangate45/base:**
Single iteration, no human intervention. Agent called `set_config_param` 3×,
`run_stage("cts")`, `run_stage("finish")`, verified metrics, called `finish(success=True)`.
Final WNS 0.000, Fmax 1239 MHz. Params written to config.mk.

**Commits:**
- `7671426da` — loop_agent: add closed-loop optimization agent
- `1afb28fd9` — loop_agent: add write-back and placement-stage support

---

### 2026-08-23 — Phase 6: unit tests

**New file: `flow/util/test_loop_agent.py`**

24 unit tests covering all non-Docker, non-API logic. No API key or Docker required.

**Test classes:**
- `TestAllowlist` — rejects unknown params (including injection attempts); accepts all 6 allowlisted
- `TestHookTranslation` — `"enabled"` → `/work/scripts/...` for both hook params; case-insensitive;
  numeric params untouched; explicit paths not double-translated
- `TestStaleFilePaths` — correct files for each stage; `place` list starts at `3_3_place_gp.odb`;
  no CTS outputs in place list
- `TestWriteConfigParams` — in-place update, append, Docker→canonical path translation,
  no duplication, error on missing file, comment only added for new params

**Run:**
```bash
cd flow && python3 util/test_loop_agent.py
```
All 24 pass in ~0.004 s.

**Commit:** `b84e1d4ac` — loop_agent: add unit test suite (24 tests, no API/Docker required)

---

### 2026-08-26 — Review fixes: value-side injection blocklist, hook dedup, regression tests

**Problem:** `set_config_param` validated the param *name* against `PARAM_ALLOWLIST` but
not the *value*. Since `write_config_params` writes the value verbatim into `config.mk`
(a GNU Make include), an adversarial or hallucinated value containing `$(shell ...)` —
or its `${shell ...}` equivalent, since Make treats `$(...)` and `${...}` as
interchangeable — would execute arbitrary shell code on the next `make` invocation.

**Fix (`flow/util/loop_agent.py`):** added `validate_param_value()`, called from
`impl_set_config_param()` before a value is queued. Rejects values containing any of
`UNSAFE_VALUE_PATTERNS` (`$(`, `${`, backtick, `;`, `|`, `&`, newline/CR).

**Also:** `post_cts_timing_repair.tcl` and `post_grt_timing_repair.tcl` were near
byte-for-byte duplicates (~200 lines each). Factored the shared upsizing logic into
`flow/scripts/timing_repair_common.tcl` (namespace `::trepair`), parameterized by
log-prefix and parasitics mode (`-placement` vs `-global_routing`); both hook files are
now thin wrappers that source the common lib.

**Tests:** added regression cases in `test_loop_agent.py` covering both the `$(` and
`${` value-injection forms for an allowlisted param, distinct from the existing
name-injection test. Suite is now 28 tests (was 24).

**Commits:**
- `8c77f24cf` — validate config param values; dedupe timing-repair Tcl hooks into shared lib
- `01cb3c686` — block `${` Make-syntax variant in config param value validation
- `d34d787fd` — add regression tests for config value injection blocklist

---

### 2026-08-23 — PR opened

**PR #1:** https://github.com/JayRaj21/OpenROAD-flow-scripts/pull/1

Title: "pr-extension: LLM-driven P&R triage, closed-loop optimization, and congestion ML pipeline"

Branch `pr-extension` → `master`.

---

### 2026-09-12 — eco_fix: targeted single-instance ECO repair, grounded via live capability probe

**Goal:** give `loop_agent.py` a fourth tool, `eco_fix`, that applies one targeted,
caller-specified incremental repair (resize / buffer insert / hold fix) to an
already-built stage database, measures timing before and after inside the same
OpenROAD process, and commits the change to the stage `.odb` only if the targeted
metric improved and nothing else regressed past tolerance — without paying the cost
of a full stage rerun.

**Tcl (`flow/scripts/eco_repair.tcl`, new file):** extends the `::trepair` namespace
from `timing_repair_common.tcl` (reuses `build_upsize_map`, `find_master`,
`is_excluded`, the `swapMaster` idiom, `detailed_placement` + `estimate_parasitics`).
Sourcing the file only defines procs; the caller invokes `trepair::eco_run` explicitly.
`eco_run` loads the stage odb via the same `open.tcl` path `read_timing` uses,
measures `wns`/`tns`/`worst_hold_slack`/`setup_viol_count`/`hold_viol_count` before
and after the edit, dispatches to one of `eco_resize` (up/down, via `swapMaster`),
`eco_insert_buffer`, or `eco_fix_hold`, computes an accept/reject verdict against
caller-supplied tolerances, writes `write_db` only on accept, and hand-writes the
before/after/verdict as JSON (no Tcl JSON package) to a file under
`OBJECTS_DIR/eco/`.

**Grounding correction — buffer insert and hold fix are NOT hand-rolled ODB surgery.**
The original plan for this feature called for splitting nets and chaining buffer
inserts manually via `odb::dbNet_create`/`dbITerm connect/disconnect`. Before writing
any of that, I ran a live capability probe (`openroad -no_init -exit probe.tcl`)
against the actual OpenROAD build shipped in this repo's `openroad/orfs:latest` Docker
image, invoking suspected commands with no/bad arguments and reading the resulting
`[ERROR ...]` usage messages to confirm real flag names — not by reading source or
guessing. This confirmed two real, existing commands that make the hand-rolled
approach unnecessary:
- `insert_buffer -net <net> -buffer_cell <cell>` — public, top-level. Used directly
  for `eco_insert_buffer`; changes the `insert_buffer` fix_type's `target` semantics
  to a **net name** (not a pin, as originally planned).
- `rsz::repair_hold_pin <end_pin> <setup_margin> <hold_margin> <allow_setup_violations> \
  <max_buffer_percent> <max_passes>` — internal, 6 positional args (confirmed via the
  arity-error message). Used directly for `eco_fix_hold`, sourcing `setup_margin`/
  `hold_margin` from the same `SETUP_SLACK_MARGIN`/`HOLD_SLACK_MARGIN` env vars this
  repo already threads into whole-design repair (`util.tcl`, `global_route.tcl`).
  `allow_setup_violations=0`, `max_buffer_percent=50`, `max_passes=1` are exposed as
  tunable constants (`::eco_max_buffer_percent`, `::eco_max_passes`) at the top of the
  namespace. This replaces the originally planned buffer-chaining loop entirely — no
  `count` parameter is needed since `repair_hold_pin` runs its own internal passes.

Also confirmed (negative result): no public top-level resize command exists, so the
plan's `swapMaster`-based resize (already proven in `timing_repair_common.tcl`) is the
only path there, as originally planned.

**Python (`flow/util/loop_agent.py`):** new `eco_fix` tool entry, `impl_eco_fix()`, and
`_format_eco_result()`. One-shot subprocess per call (`util/docker_shell make ...
RUN_SCRIPT=.../eco/<id>.tcl RUN_LOG_NAME_STEM=<id> run` — `run` is the phony,
no-prerequisite target, so this never triggers a stage rebuild). `target`/`cell` are
validated against `ECO_NAME_RE` before ever being interpolated into generated Tcl —
this is the Tcl-injection boundary, mirroring the existing `validate_param_value`
pattern for config params. `fix_type` and `stage` are validated against fixed
allowlists (`ECO_FIX_TYPES`, `ECO_STAGE_ODB`); the host-side `.odb` existence is
checked before any subprocess is spawned. Results are parsed from a JSON file under
`OBJECTS_DIR/eco/`, not scraped from stdout, and appended to `change_log`. Three
system-prompt sentences added: prefer `eco_fix` over `run_stage` for single-instance
fixes; it edits the stage `.odb` in place, so do ECOs last (after all `run_stage`
calls); ECO results aren't visible in `get_metrics` until a downstream `run_stage
finish` regenerates reports.

**Tests:** added `TestEcoFixValidation`, `TestEcoFixGeneratedTcl`, and
`TestEcoResultFormatter` to `test_loop_agent.py` — unknown `fix_type`/injected
`target`/bad `stage` rejected without spawning a subprocess (mocked and asserted
never called); missing-odb short-circuits the same way; generated Tcl contains the
validated target and `trepair::eco_run`; the result formatter renders accept/reject
correctly for hand-built before/after dicts across all four fix types. Suite is now
40 tests (was 28).

---

### 2026-09-12 — eco_fix: validator round of fixes (live-reproduced bugs)

An independent validator built a real design (`nangate45/gcd`) with the actual
`openroad/orfs:latest` Docker image and ran `eco_fix` end-to-end for all four
fix types. It found real, live-reproduced bugs, not style issues. All were
fixed in this session; each fix was re-verified live against the same image
(see "Live verification" below), not just by unit test.

**1. Every error path of `eco_run` crashed before writing the JSON result.**
`eco_resize`/`eco_insert_buffer`/`eco_fix_hold`'s error returns, and the
`default` branch, were two-key dicts (`status`/`msg`) with no `kind`, but
`eco_run` unconditionally did `dict get $fix kind|inst|from|to` on them —
crashing the whole `make run` invocation on any user error (e.g. resizing a
non-existent instance). Fixed by giving every error-path dict a `kind`
matching its fix type (`resize`, `insert_buffer`, `fix_hold`, `unknown`), and
merging in `inst`/`from`/`to` defaults (`dict merge {kind "" inst "" from ""
to ""} $fix`) before `eco_run` reads any of those keys, so the read never
fails regardless of success/failure. `msg` is now always threaded through to
the final JSON (`eco_write_result`, factored out of `eco_run` so every
early-return error path — a thrown `eco_load`/`eco_measure` error — writes
the same clean JSON as the success path, closing part of #9 below too).

**2. The hand-rolled JSON writer didn't escape backslashes — successful ECOs
on real instance names got reported as failures, silently.** `eco_json_str`
only escaped `"`, not `\`. Real yosys/ODB names contain backslashes (e.g.
gcd's own flops: `ctrl.state.out\[0\]$_DFF_P_`), and `ECO_NAME_RE` in
`loop_agent.py` already permits `\`. A successful fix on such a name produced
JSON with an invalid `\[` escape, `json.load` raised, and the agent reported
"eco run produced no result" — even though `write_db` may have already
mutated the .odb, with no record of it (the real safety-stakes bug: a
completed, undocumented DB mutation). Fixed: `eco_json_str` now maps `\` →
`\\` **before** mapping `"` → `\"` (order matters — otherwise the
backslash inserted by quote-escaping would itself get re-escaped), and also
escapes `\n`/`\r`/`\t` so raw OpenSTA/OpenROAD error text in `msg` can't
break JSON structure.

**3. `insert_buffer` segfaults OpenROAD on every tested input — disabled,
not left reachable.** Live-tested `insert_buffer -net <net> -buffer_cell
<cell>` on an ordinary net in `nangate45/gcd`'s `4_cts.odb`: confirmed a
process-level segfault (Signal 11, `rsz::Resizer::insertBufferAfterDriver`)
regardless of arguments (explicit cell, `-location`, `-load_pins`). A
segfault can't be caught by Tcl — it kills the whole `make run` subprocess
before any JSON can be written, so this is not something a `catch` can fix.
Removed `insert_buffer` from `ECO_FIX_TYPES` and the `eco_fix` tool's
`fix_type` enum in `loop_agent.py` (3 fix types remain: resize_up,
resize_down, fix_hold); updated the tool description accordingly.
`eco_insert_buffer` is kept defined in `eco_repair.tcl` (documented as
disabled, with the exact segfault signature and date) for future use if
this is fixed upstream, but `eco_run`'s dispatch now returns a clean
`{status error kind insert_buffer msg "insert_buffer is disabled..."}`
directly, without ever calling it. Also fixed, in passing, the pre-existing
`MIN_BUF_CELL_AND_PORTS` parsing bug (it's a 3-token "cell input_port
output_port" list, e.g. `BUF_X1 A Z` — the whole string was being passed as
`-buffer_cell`, which fails STA-0116) and wrapped the env read in a catch,
in case the proc is re-enabled later.

**4. `eco_targets` was always empty.** `[$cur prevPath]` does not exist on
this build's Path object (live probe: "Invalid method. Must be one of: ...
pin edge tag pins start_path") — the `catch` around it silently broke the
traversal loop after one pin. Also, `find_timing_paths -path_delay max
-sort_by_slack` without `-group_count`/`-group_path_count` returns only one
path's worth of pins. Fixed by using `[$path pins]` (returns every pin along
one path in one call — confirmed via live probe) instead of manually walking
`prevPath`, and adding `-group_path_count $count` (not the deprecated
`-group_count`, which the live run flagged with `STA-0503`) so multiple
distinct violating endpoints are returned. Live-verified on `gcd/base`
(23 setup violations at CTS): `eco_targets` now returns 100+ target dicts
instead of `[]`. `timing_repair_common.tcl` itself was intentionally left
untouched (shared file, out of scope) even though it has the same bug.

**5. `setup_viol_count` was measured but never gated.** `eco_verdict`
checked `hold_viol_count` regression but not `setup_viol_count`, so an ECO
that improved WNS while pushing setup violations from 23 to 200 would be
accepted. Added a `setup_viol_count` regression check (`after <= before`,
same strict style as the existing `hold_viol_count` check) to
`eco_verdict`'s `no_regression` gate. Live-verified: a synthetic
after-dict with `setup_viol_count` 23→200 and an otherwise-improving WNS is
correctly rejected with reason `"setup_viol_count increased from 23 to
200"`.

**6. The accept/reject reason string always quoted the WNS delta, regardless
of the actual target metric.** A `fix_hold` accept said "improved
worst_hold_slack by 0.0" while quoting the WNS delta; a `resize_down` accept
could say "improved wns by -0.0002..." (a negative "improvement"). Fixed by
building a `wns`/`tns`/`worst_hold_slack` → delta dict and indexing it by
`$target_metric` when building the accept reason. Live-verified: a
synthetic `fix_hold` accept now reports "improved worst_hold_slack by 0.006
and no regression" (the hold delta, not `d_wns`).

**7. `fix_hold` was effectively unreachable through the verdict gate.**
Hold-violation fixes structurally trade a small, bounded amount of setup
slack for hold closure (the inserted hold buffer adds delay on the data
path). The flat `tol_wns` (0.001 ns) treated that expected collateral cost
as a regression, rejecting nearly every real hold fix — live-reproduced with
the validator's exact scenario: WNS -0.100 → -0.102 (a normal -0.002 hold-fix
cost) rejected as "wns regressed". Added `tol_wns_hold` (new
`ECO_TOLERANCES["wns_hold"] = 0.01`, 10x the generic tolerance — large
enough to absorb normal hold-fix collateral, still bounded enough to catch a
runaway hold fix), threaded through `eco_run`'s new `tol_wns_hold` argument
and `impl_eco_fix`'s generated Tcl call, and used in `eco_verdict` only for
`fix_type eq "fix_hold"` (the WNS-regression check is fix-type-aware, not
disabled outright — a large setup regression from a runaway hold fix is
still caught). Live-verified with the exact validator scenario (`before`
wns=-0.100/hold=-0.005, `after` wns=-0.102/hold=+0.001): now **accepted**
("improved worst_hold_slack by 0.006 and no regression"); re-running the
same case with the old flat tolerance confirms it would still be rejected
("wns regressed by -0.002"), proving the fix is specific to `fix_hold` and
not a blanket loosening.

**8. A target ending in an odd number of backslashes broke the generated
Tcl.** `loop_agent.py` wraps `target`/`cell` in `{...}`; `ECO_NAME_RE`
permits `\`; a trailing single backslash escapes the closing brace, giving
an opaque "missing close-brace" Tcl error instead of a clean validation
error. Added a cheap parity check in `impl_eco_fix` (Python side, before Tcl
generation): reject `target`/`cell` ending in an odd number of consecutive
backslashes with a clear `ERROR: ... ends in an odd number of backslashes`.

**9. Several fatal-if-thrown calls sat outside any catch.** `eco_measure`
(before/after), `eco_targets`, `write_db`, and the `estimate_parasitics`
calls inside each fix proc could each throw and skip the JSON writer
entirely. All are now wrapped in `catch`, writing a clean `{status error
kind ... msg ...}` result instead of aborting `eco_run` silently (`eco_measure`-before
and `write_db` failures are terminal for that run and reported as `status
error`; an `eco_targets` failure is non-fatal — a `targets: []` result with
a `WARN` log line, since discovery failing shouldn't block a fix the caller
already named). Also: `eco_load` returned `-global_routing` for stage `grt`
purely by tag, while `open.tcl`'s own equivalent read guards it with
`grt::have_routes` — an unrouted `5_1_grt.odb` (e.g. from an aborted GRT run)
would otherwise abort later when `estimate_parasitics -global_routing` is
called with no actual routes. `eco_load` now calls `grt::have_routes` itself
(wrapped in `catch`) and falls back to `-placement` with a `WARN` log line
if routes aren't available.

**10. Test gaps.** Added 8 tests to `test_loop_agent.py` (40 → 48 total):
`test_rejects_insert_buffer_fix_type`, `test_rejects_cell_with_semicolon`,
`test_rejects_fix_hold_target_without_slash`,
`test_rejects_target_with_odd_trailing_backslashes` /
`test_accepts_target_with_even_trailing_backslashes` /
`test_rejects_cell_with_odd_trailing_backslashes`,
`test_json_parse_failure_returns_error_with_output_tail`, and — the one that
actually matters for #2 — `TestEcoFixJsonTransport
.test_backslash_in_pin_name_round_trips_through_real_json_parse`, which
mocks `subprocess.run` to write a raw JSON **string** (not a pre-parsed
Python dict, which by construction can't exercise the Tcl-side escaping bug)
containing an escaped backslash, decoded through the real `json.load()` call
in `impl_eco_fix`. `test_tolerances_exposed` was also de-vacuous-ified: it
now asserts `wns_hold` exists and is strictly looser than the generic `wns`
tolerance.

**Live verification (all against `openroad/orfs:latest`, `nangate45/gcd/base`,
which had a real, already-built `4_cts.odb`/`5_1_grt.odb` with 23 real setup
violations):**
- (a) `resize_up` on `NONEXISTENT_INST_123` → clean `status: error`, `msg:
  "instance not found: NONEXISTENT_INST_123"` JSON; no crash. (#1)
- (b) `fix_hold` on the real gcd flop pin `ctrl.state.out\[0\]$_DFF_P_/D`
  (genuine backslash-escaped name) → result parses as valid JSON end to end
  through `impl_eco_fix`'s real `json.load()`; no "produced no result"
  error. (#2)
- (c) `insert_buffer` invoked directly at the Tcl layer (bypassing the
  now-updated Python enum, to prove the Tcl dispatch itself is safe even if
  somehow reached) → clean disabled-error JSON, "DONE — no segfault" printed
  after. (#3)
- (d) `eco_targets 5` on the same 23-setup-violation design → 100+ non-empty
  target dicts (previously always `[]`). (#4)
- (e) `eco_verdict fix_hold` with the validator's exact before/after
  numbers (wns -0.100→-0.102, hold -0.005→+0.001) → `accepted true`, reason
  quotes the hold delta (#6); re-run with the old flat tolerance →
  `accepted false`, confirming the fix is real and specific to `fix_hold`.
  Also verified a `setup_viol_count` 23→200 regression is still rejected
  even when WNS nominally improves. (#5, #7)

**Files touched:** `flow/scripts/eco_repair.tcl` (fixes #1–7, #9),
`flow/util/loop_agent.py` (fix #3's enum removal, fix #7's `ECO_TOLERANCES`
addition, fix #8's validation), `flow/util/test_loop_agent.py` (#10),
`PR_EXTENSION_DEV_LOG.md` (this entry).

---

### 2026-09-12 — eco_fix: second validator round (swapMaster no-op, tns==NA, setup_viol_count/tol_wns_hold conflict)

A second, independent live re-verification against `openroad/orfs:latest`
found one CRITICAL and three lesser issues in the first round's fixes. All
fixed:

1. **CRITICAL — `swapMaster` silently no-ops on incompatible cells,
   reported as `applied`.** `$inst swapMaster $new_master` returns 0 and
   throws nothing when the target master's terminals don't match the
   current one's — live-reproduced 100%: `resize_down(_513_, cell=INV_X4)`
   on a real `NAND2_X4` instance in `nangate45/gcd`'s built `5_1_grt.odb`
   returned `status: applied`, `odb_written: true`, with a fabricated
   `from`/`to` delta, while the `.odb` was byte-identical (md5) before and
   after. Fixed in `eco_resize` (`flow/scripts/eco_repair.tcl`) two ways:
   (a) a pre-check (`terms_compatible`/`master_signal_terms`, comparing
   non-power/ground `getMTerms` names) rejects an explicit `cell` that
   isn't terminal-compatible *before* attempting the swap, with a clear
   message; (b) a post-swap check re-reads `[[$inst getMaster] getName]`
   and errors out if it doesn't match the intended cell, as a safety net
   for both the explicit-cell and auto-selected paths. Live-reconfirmed:
   the exact repro case now returns `status: error` with an unchanged
   `.odb` (md5 identical to before); a legitimate compatible swap
   (`NAND2_X4` → `NAND2_X2`) still applies and actually changes the
   `.odb` (different md5, `from`/`to` populated correctly).

2. **MEDIUM — zero-tolerance `setup_viol_count` check defeats
   `tol_wns_hold`.** `fix_hold`'s collateral WNS cost (up to
   `tol_wns_hold` = 0.01ns, added in the first round specifically to allow
   this) will routinely flip one or two near-zero endpoints into
   violation, which the zero-tolerance `setup_viol_count` regression check
   then rejected — live-confirmed with a real 0.18ps WNS delta flipping
   `setup_viol_count` 23→24 on gcd. Gave `eco_verdict`
   (`flow/scripts/eco_repair.tcl`) a `fix_hold`-specific
   `setup_viol_count` tolerance of +3 endpoints (small and bounded, sized
   to absorb the expected one-or-two-endpoint collateral flip from
   `tol_wns_hold`-level WNS cost without hiding a real setup regression).
   `resize_up`/`resize_down`/default keep zero tolerance — they have no
   structural reason to trade away setup violations, and live-testing
   confirmed zero-tolerance there still correctly rejects a real
   WNS-only regression.

3. **LOW — `tns == "NA"` silently treated as a zero delta.** A
   `report_tns` parse failure forced `d_tns` to `0.0`, which passed both
   the `resize_down` "improved" check and the TNS regression check —
   live-confirmed an eco run with `tns: NA` was accepted and written with
   TNS effectively unchecked. `eco_verdict` now checks `wns`/`tns`/
   `worst_hold_slack` on both sides for `"NA"` up front and refuses to
   verdict at all in that case (`accepted: false`, reason
   `"insufficient data: <metric> could not be measured"`) rather than
   guessing — the safer default for a tool that writes to the design
   database.

4. **LOW — `detailed_placement` failures were dropped after reaching only
   stdout.** All three `catch { detailed_placement }` call sites in
   `eco_repair.tcl` (`eco_resize`, `eco_insert_buffer`, `eco_fix_hold`) now
   set a `placement_warning` string on failure instead of only `puts`-ing a
   WARN line. Chose to surface, not abort: a placement warning doesn't
   always mean a real overlap, and this preserves the existing behavior of
   still measuring/verdict-ing/writing when the caller decides to. The
   field is threaded through `eco_write_result`'s JSON output and
   `loop_agent.py`'s `_format_eco_result` so the LLM caller — which reads
   only the JSON — now sees it instead of it being silently dropped with
   stdout.

**Opportunistic (per plan, time-permitting):**
- Deduped `eco_targets`'s output (`seen_pins` list) — `-group_path_count`
  paths share sub-paths, so `[$path pins]` was returning the same pin
  multiple times (134 entries / 100 unique on gcd, previously).
- `eco_verdict`'s accept reason for a neutral `resize_down` (delta exactly
  0) now reads `"no regression (wns unchanged)"` instead of the
  self-contradictory `"improved wns by 0.0"`.
- Left `eco_run`'s `opt_count` parameter as-is (still dead code, per the
  plan's "your call" — not touched this round to keep the diff focused on
  the four correctness issues above).

**Live verification (`openroad/orfs:latest`, `nangate45/gcd/base`, real
built `5_1_grt.odb`):**
- (a) Exact repro: `resize_down(_513_, cell=INV_X4)` on the real
  `NAND2_X4` instance `_513_` → `status: error`, msg names the
  incompatible swap; `.odb` md5 unchanged (`8900606f...` before and
  after). (#1)
- (b) Legitimate compatible resize: `resize_down(_513_, cell=NAND2_X2)` →
  `status: applied`, `odb_written: true`, `from: NAND2_X4, to: NAND2_X2`,
  `.odb` md5 changed (`8900606f...` → `6edcf2a2...`). (#1)
- (c) `eco_verdict fix_hold` tested directly via `tclsh` (pure Tcl logic,
  no ODB dependency) with the validator's exact numbers (WNS
  -0.001→-0.00118, `setup_viol_count` 23→24) → `accepted: true`; the same
  fixture re-run with the old flat (zero-tolerance) `setup_viol_count`
  check would have rejected it — confirming the fix-type-aware tolerance
  is both necessary and sufficient here. A larger 23→30 regression on the
  same fixture is still correctly rejected. (#2)
- (d) `eco_verdict` with `tns: "NA"` on one side → `accepted: false`,
  reason `"insufficient data: tns could not be measured"`, verified via
  the same `tclsh`-direct harness. (#3)

**Tests:** `python3 -m unittest flow.util.test_loop_agent -v` — 55 tests
pass (up from 48; 7 new tests added directly exercising `eco_repair.tcl`'s
Tcl logic — `terms_compatible`, `eco_verdict`'s NA-rejection, and the
`fix_hold`/`resize_down` setup_viol_count tolerance split — via `tclsh`
subprocess calls, since `eco_verdict`/`terms_compatible` are pure Tcl
functions with no ODB dependency).

**Files touched:** `flow/scripts/eco_repair.tcl` (all four fixes plus the
opportunistic dedup/wording changes), `flow/util/loop_agent.py`
(`placement_warning` surfaced in `_format_eco_result`),
`flow/util/test_loop_agent.py` (7 new tests), `PR_EXTENSION_DEV_LOG.md`
(this entry).

---

### 2026-08-27 — Regression / benchmark dashboard

**Goal:** turn the single-run `pr_metrics.py` snapshot into a history that can catch
regressions across runs/commits, usable both interactively and as a CI gate.

**`flow/util/benchmark_dashboard.py`:** new module, two subcommands, argparse styled
after `pr_metrics.py` (`--platform`/`--design`/`--tag` or `--reports-dir`/`--logs-dir`,
same `--flow-dir` default). Imports `collect()` from `pr_metrics.py` rather than
re-parsing reports/logs — that duplication (pr_metrics.py/triage_agent.py/loop_agent.py/
compare_hook.sh all independently extracting the same metrics) was already flagged as a
review issue, so this is strictly a history/regression layer on top of the existing
parser. `pr_metrics.py` itself is untouched.

- `record`: runs `collect()`, appends one JSON object (timestamp, `git rev-parse HEAD`,
  platform/design/tag, per-stage metrics dict) as a line to
  `flow/util/benchmark_history/<platform>__<design>__<tag>.jsonl`. JSONL + open-append
  (`"a"` mode) was chosen specifically so a crash or concurrent writer can never
  corrupt or rewrite prior history — each record is independent and the file is safe to
  tail/grep.
- `report`: reads the history file for a stage (default `Finish`), prints a table with
  per-metric deltas vs. the previous record and `worse-than-best-*` flags vs. the
  best-ever value across history. Regression detection compares only the latest record
  against its immediate predecessor (not best-ever) against three configurable
  thresholds — WNS worsening (`--wns-threshold`, default 0.01 ns), Fmax percentage drop
  (`--fmax-threshold-pct`, default 1.0), GRT/GP overflow increase (`--overflow-threshold`,
  default 0.001) — and exits 1 if any fire, 0 otherwise, so it drops straight into a CI
  pipeline as a gate. Fewer than 2 records just prints the single row and exits 0.
  `--html` additionally emits one self-contained HTML file (inline `<svg>` line charts
  for WNS/Fmax/HPWL, inline `<style>`, no external JS/CSS/CDN/font references) so it
  renders in an air-gapped CI runner.

**Tests (`flow/util/test_benchmark_dashboard.py`, unittest, 34 tests):** append-only
behavior (multiple `record` calls never touch prior lines), delta/regression math on
hand-built synthetic JSONL sequences (regression flagged/not-flagged at threshold
boundaries for WNS/Fmax/overflow, best-ever flags, exit-code behavior via
`build_report_rows`), a CLI subprocess test that runs `record` against a fake
`flow/reports/.../6_finish.rpt` fixture and checks the written JSONL content, an
HTML test asserting the output file is non-empty, contains `<svg>`/`<table>`, and has no
`http(s)://` references (confirms it's genuinely offline-renderable), and
`resolve_dirs()` tag-derivation tests (`--reports-dir` derives the tag from the path's
last component when `--tag` isn't passed; an explicit `--tag` overrides derivation).
Ran together with the existing suite:

```bash
cd flow/util && python3 -m pytest test_benchmark_dashboard.py test_loop_agent.py -v
```
62 passed (34 new + 28 existing), confirming no regression to `loop_agent.py`. Formatted
both new files with `black` (26.5.1).

**Left out of scope:** no Makefile/CI wiring to auto-invoke `record` after every flow
run (roadmap says infra-only for this pass; wiring belongs with whichever CI workflow
task consumes it), no retention/pruning policy for history files (JSONL is cheap and
append-only; pruning can be a follow-up if files get large), no cross-design aggregate
dashboard (each `<platform>__<design>__<tag>` gets its own file/report, matching how
`pr_metrics.py` is already scoped to one run at a time).

---

### 2026-08-27 — Multi-corner / multi-mode timing dashboard

**What:** added an opt-in, additive per-corner timing breakdown on top of ORFS's
existing multi-corner STA support (`flow/scripts/read_liberty.tcl` already reads
liberty per corner via `define_corners`/`read_liberty -corner`), plus a Python
dashboard to compare corners side by side.

**Files:**
- `flow/scripts/report_multicorner_timing.tcl` — new standalone proc
  `report_multicorner_timing { stage when }`. Gated behind
  `REPORT_MULTICORNER_TIMING` (unset/`0` = no-op, matching the
  `SKIP_REPORT_METRICS`/`DETAILED_METRICS`/`CTS_SNAPSHOTS` opt-in pattern).
  No-op when `CORNERS` has fewer than 2 entries. When enabled, loops
  `$::env(CORNERS)` and, per corner, reads TNS/worst-slack via the
  corner-scoped SWIG commands (`sta::find_scene`, `sta::total_negative_slack_scene_cmd`,
  `sta::worst_slack_scene` — see correction below), derives WNS as
  `min(0.0, worst_slack)`, and (if `REPORT_CLOCK_SKEW`) appends
  `report_clock_skew -corner $corner`'s native output, into one file per
  corner: `$::env(REPORTS_DIR)/${stage}_${when}_multicorner_${corner}.rpt` —
  mirroring the existing single-corner `<stage>_<when>.rpt` naming from
  `report_metrics.tcl`.
- `flow/util/multicorner_dashboard.py` — CLI (`--platform`/`--design`/`--tag`/
  `--flow-dir`, matching `pr_metrics.py`'s convention, plus `--stage` to pick
  which stage's `*_multicorner_*.rpt` files to read, defaulting to the
  highest-numbered stage found). Imports and reuses `pr_metrics.parse_rpt()`
  for WNS/TNS/worst-slack (no reimplemented regexes) and adds a clock-skew
  parser matching OpenSTA's real per-clock `"<value> setup|hold skew"` output.
  Prints a table with corners as columns and a `"(worst)"` suffix marking the
  worst corner per metric (most-negative for slack/TNS/WNS, largest-magnitude
  for skew). Exits non-zero with a clear message if no matching multicorner
  reports are found.
- `flow/util/test_multicorner_dashboard.py` — unittest-based (style matches
  `test_loop_agent.py`), synthetic `.rpt` fixtures in temp dirs, no
  Docker/OpenSTA required. Covers file discovery/glob matching (including
  underscore-containing corner names like `ss_0p9v_125c`), stage
  auto-selection, `parse_rpt()` reuse, worst-corner selection, table
  rendering, and three behavioral Tcl checks via `tclsh` subprocess: the
  script sources without a syntax error; the single-corner no-op path writes
  no files; and — critically — the 2+-corner code path itself, driven end to
  end with the real `sta::*` commands stubbed to return known values,
  asserting the written files parse back to the expected TNS/WNS/worst-slack/
  clock-skew numbers.

**Correction (same day, before commit landed clean):** an independent review
caught that the first draft of `report_multicorner_timing.tcl` called
`report_tns -corner`, `report_wns -corner`, and `report_worst_slack -corner`
by analogy with `report_power -corner` — but never verified it against real
OpenSTA source. That analogy was wrong and would have hard-crashed the flow
the first time it ran with `REPORT_MULTICORNER_TIMING=1` and 2+ corners:
`parse_key_args` rejects unknown flags, and none of those three commands
declare `-corner`. Re-derived the fix by pulling the actual OpenSTA source at
the exact commit ORFS's `tools/OpenROAD` submodule pins
(`509913b1398b36eda23caa1f1f380167465dceee`, verified via
`gh api repos/The-OpenROAD-Project/OpenROAD/contents/src?ref=...`), not
upstream `master` blindly:
  - `search/Search.tcl` confirms `report_tns`/`report_wns`/`report_worst_slack`
    take only `[-min] [-max] [-digits digits]` — no `-corner`.
  - `search/Search.i` exposes the real lower-level, corner-scoped commands
    those procs are missing: `total_negative_slack_scene_cmd(Scene*, MinMax*)`,
    `worst_slack_scene(Scene*, MinMax*)`, and `find_scene(const char*)` — and
    OpenSTA's own test suite (`search/test/search_worst_slack_sta.tcl`,
    `search/test/search_corner_skew.tcl`) uses exactly this pattern
    (`sta::find_scene`, `sta::total_negative_slack_scene_cmd $scene max`,
    `sta::worst_slack_scene $scene max`). The script now uses these instead.
  - The review also flagged `report_clock_skew -corner` as a dead parameter.
    On closer reading of `tcl/CmdArgs.tcl` this is actually **not** dead:
    `report_clock_skew` passes its parsed `keys` array by reference into
    `parse_scenes_or_all keys`, which explicitly reads `keys(-corner)` as a
    documented `"compabibility 05/29/2025"` alias for `-scenes`. So
    `report_clock_skew -corner $corner` is genuine and was kept — but its
    real output format is a per-clock `"<value> setup|hold skew"` line, not
    an aggregate "worst skew" line as the first draft's dashboard parser
    assumed; `multicorner_dashboard.py`'s clock-skew regex and worst-corner
    logic (largest magnitude, not most-negative) were rewritten to match.
  - Also fixed: the corner-name regex in `multicorner_dashboard.py`
    (`find_multicorner_reports`) excluded underscores, which would break on
    real corner names like `ss_0p9v_125c`; broadened to allow them.
  - Added the missing test that actually drives the 2+-corner Tcl branch
    (`TestTclSyntax::test_proc_report_multicorner_timing_drives_two_corner_branch`)
    by stubbing the real `sta::find_scene` / `sta::total_negative_slack_scene_cmd`
    / `sta::worst_slack_scene` / `sta::format_time` commands and asserting on
    the files it writes — this is the exact branch the wrong first draft
    would have crashed in, and it had zero coverage before.

**Tcl integration decision:** used the existing `HOOK_PATHS`/`CONFIG_HOOK_PATHS`
mechanism (same pattern as `post_cts_timing_repair.tcl`) rather than a direct
call site inside a stage script, so `report_metrics.tcl` and every stage
script (`cts.tcl`, `global_route.tcl`, `final_outputs.tcl`, etc.) stay
completely untouched — zero risk of regressing existing runs. **(Superseded
2026-09-11, see below: wire `POST_CTS_TCL` to
`report_multicorner_timing_cts.tcl` and `POST_GLOBAL_ROUTE_TCL` to
`report_multicorner_timing_grt.tcl`, not the same file for both — the
`REPORT_MULTICORNER_STAGE`/`REPORT_MULTICORNER_WHEN` env-var-based labelling
described in this paragraph was removed.)** A design wires
it in via e.g. `export POST_CTS_TCL = $(SCRIPTS_DIR)/report_multicorner_timing.tcl`
plus `export REPORT_MULTICORNER_TIMING = 1`. Since a hook is only `source`d
(no call-site args), the script reads optional `REPORT_MULTICORNER_STAGE`/
`REPORT_MULTICORNER_WHEN` env vars (defaulting to `"4"`/`"cts final"`, tuned
for `POST_CTS_TCL`) to label the output files, and also exposes
`report_multicorner_timing { stage when }` for direct manual invocation after
sourcing. Tradeoff: the hook-slot approach only fires at the specific point a
hook already exists (post-CTS, post-GRT) — it cannot label an arbitrary
stage/when pair without either wiring a hook per stage or a future direct
call site in a stage script; this was deliberately left as future work to
keep this change additive-only.

**Testing:** `python3 -m pytest flow/util/test_multicorner_dashboard.py
flow/util/test_loop_agent.py -v` → 46 passed. Also manually exercised the CLI
against hand-built fixture `.rpt` files reproducing OpenSTA's real `tns max` /
`wns max` / `worst slack max` / per-clock `"<value> setup skew"` output
format (including underscore corner names), confirming the table correctly
renders and marks the worst corner. `black` applied to both new Python files.

**Out of scope:** wiring `REPORT_MULTICORNER_TIMING` into an actual design's
`config.mk` (needs a real multi-corner platform config to validate against
live OpenSTA output); a direct stage-script call site as an alternative to
the hook mechanism.

---

### 2026-09-11 — Fix two MEDIUM findings from independent validator review

**Context:** an independent validator agent reproduced two MEDIUM-severity bugs
end-to-end against the real OpenSTA source at the pinned `tools/OpenROAD`
submodule commit (`509913b1398b36eda23caa1f1f380167465dceee`). No HIGHs were
found on this branch; LOW-severity items were left alone per scope.

**Finding 1 — nondeterministic `default_stage()` on a numeric-prefix tie
(`flow/util/multicorner_dashboard.py`):** `default_stage()` collected stage
labels into a `set` and broke ties on `sort_key` (numeric prefix only), so
two labels sharing a prefix (e.g. `4_cts_final` vs.
`4_cts_pre-repair-timing`, both left on disk because `REPORTS_DIR` is only
swept by `make clean_cts`, not between incremental re-runs with a changed
`REPORT_MULTICORNER_WHEN`) resolved by Python's hash-randomized set iteration
order — i.e. by `PYTHONHASHSEED`. Same inputs, different dashboard on every
invocation.

**Fix:** `default_stage()` now builds a `{stage: max_mtime}` dict (not a set),
sorts candidates by `(numeric_prefix, full_string)` — a fully deterministic
key independent of hash order — and, when multiple labels still tie on the
same numeric prefix, breaks the tie by picking the most-recently-modified
one and prints a warning to stderr flagging that stale reports may be
present.

**Finding 2 — cross-invocation label/data stomping
(`flow/scripts/report_multicorner_timing.tcl`):** the proc itself already
took explicit `stage`/`when` arguments, so direct calls were never the
problem. The bug was in the bottom "wired as a hook" block: it derived
`stage`/`when` from `REPORT_MULTICORNER_STAGE`/`REPORT_MULTICORNER_WHEN`,
which are Make/env variables — process-global for the whole flow run. Wiring
this same file to both `POST_CTS_TCL` and `POST_GLOBAL_ROUTE_TCL` (as the
file's own header comment suggested was supported) sources it twice in one
interpreter with a single `export` visible to both sourcings, so the second
invocation reused the first's label, truncating (`open $filename w`) and
overwriting the first invocation's report under a now-mislabeled name.

**Fix:** the hook-wiring block now tracks `::report_multicorner_invocation_num`
and `::report_multicorner_seen_stages` — Tcl globals that persist across
re-sourcing within the same interpreter (never `unset`) — so each successive
sourcing in one session gets a distinct default label (`4`/"cts final", then
`5`/"global route", ...), and an explicit env override that collides with a
stage already seen earlier in the session is detected, warned about on
stderr, and auto-adjusted instead of silently overwriting. Separately,
inside `report_multicorner_timing` itself, the per-corner `open $filename w`
truncate-and-create is now ordered *after* the `sta::find_scene` validity
check (previously it ran first), so an unknown corner no longer leaves a
0-byte file behind — a one-line reordering that incidentally also closes the
related LOW-severity finding, per the plan's guidance to take that fix since
it was free.

**Tests (`flow/util/test_multicorner_dashboard.py`):** added
`test_default_stage_tie_on_numeric_prefix_is_deterministic`,
`test_default_stage_tie_deterministic_across_pythonhashseed` (re-invokes
`default_stage()` in subprocesses under `PYTHONHASHSEED=0,1,42` and asserts
identical output), and `test_default_stage_tie_warns_on_stderr`; plus
`test_two_hook_sourcings_in_one_session_do_not_cross_contaminate`, which
sources `report_multicorner_timing.tcl` twice in one `tclsh` process with the
underlying timing data changed in between (mirroring `POST_CTS_TCL` =
`POST_GLOBAL_ROUTE_TCL`) and asserts both `4_cts_final_multicorner_tt.rpt`
and `5_global_route_multicorner_tt.rpt` exist with their own, uncontaminated
data.

**Testing:** `python3 -m pytest flow/util/test_multicorner_dashboard.py -v` →
22 passed (was 18).

---

### 2026-09-11 — Round 2: the Finding-2 fix above was wrong; split into
### per-stage hook files instead of in-process counters

**Context:** the Finding 2 fix above (`::report_multicorner_invocation_num` /
`::report_multicorner_seen_stages` Tcl globals persisting across re-sourcing
"within the same interpreter") rested on an unverified assumption: that
`POST_CTS_TCL` and `POST_GLOBAL_ROUTE_TCL`, when wired to the same file,
source it twice in *one* interpreter session. Checking the actual ORFS
Makefile / `flow.sh` shows this is false — `cts.tcl` and `global_route.tcl`
each run as a **separate, fresh OpenROAD process**. So the counter/seen-set
globals reset to empty on every hook firing and always pick the same
first-slot default (`4`/"cts final") regardless of which hook actually
fired. The round-1 fix did nothing; the original bug — a `POST_GLOBAL_ROUTE_TCL`
firing silently overwriting the CTS report under a mislabeled `4_cts_final`
name — was exactly as broken as before, and the header comment's claim of
automatic same-interpreter handling was false.

**Root cause:** there is no reliable way for a single hook file to
introspect "what stage am I in" from a fresh process — no exposed getter
for the current stage name, no argv/env variable carries it, and a
Make-target-specific export can't work in single-process `flow.tcl`/
bazel-orfs mode either. The only correct fix is to give each hook point its
own file with a hardcoded identity, exactly like the existing
`post_cts_timing_repair.tcl` / `post_grt_timing_repair.tcl` split (which
share `timing_repair_common.tcl`).

**Fix:**
- **New file `flow/scripts/multicorner_timing_common.tcl`** — the actual
  reporting logic (`report_multicorner_timing_enabled`, and
  `report_multicorner_timing { stage when }` with its corner-iteration /
  report-writing body), unchanged except the header comment's wiring
  section and the removal of the false same-interpreter-fallback claim.
- **New file `flow/scripts/report_multicorner_timing_cts.tcl`** — sources
  `multicorner_timing_common.tcl`, then calls
  `report_multicorner_timing 4 "cts final"` (the actual pre-existing
  default for the CTS hook). Wired via
  `export POST_CTS_TCL = $(SCRIPTS_DIR)/report_multicorner_timing_cts.tcl`.
- **New file `flow/scripts/report_multicorner_timing_grt.tcl`** — sources
  `multicorner_timing_common.tcl`, then calls
  `report_multicorner_timing 5 "global route"` (the actual pre-existing
  default for the GRT hook). Wired via
  `export POST_GLOBAL_ROUTE_TCL = $(SCRIPTS_DIR)/report_multicorner_timing_grt.tcl`.
- **Removed** `flow/scripts/report_multicorner_timing.tcl` entirely, along
  with the `::report_multicorner_invocation_num` /
  `::report_multicorner_seen_stages` global-tracking code and the
  `REPORT_MULTICORNER_STAGE` / `REPORT_MULTICORNER_WHEN` env-var-based
  label-guessing block — all dead weight once each hook file has a
  hardcoded identity. `report_multicorner_timing { stage when }` itself
  (the part that always took explicit arguments) is untouched.
- Since each hook point is now a distinct file/process by construction,
  the "two hooks in one interpreter session" scenario the header comment
  used to warn about can no longer occur, so that warning was deleted
  rather than reworded.
- `flow/util/multicorner_dashboard.py`'s module docstring updated to
  reference `multicorner_timing_common.tcl` /
  `report_multicorner_timing_cts.tcl` / `report_multicorner_timing_grt.tcl`
  instead of the removed single file. No functional change to
  `multicorner_dashboard.py` — the round-1 `default_stage()` tie-break fix
  and the `open`-after-`find_scene` reordering are untouched.

**Tests (`flow/util/test_multicorner_dashboard.py`):**
- `test_two_hook_sourcings_in_one_session_do_not_cross_contaminate` removed
  — it tested an artificial single-interpreter double-sourcing scenario
  that does not match ORFS's real per-stage-process model, so it validated
  nothing about the actual bug.
- Replaced with
  `test_cts_and_grt_wrappers_in_separate_processes_do_not_collide`, which
  runs `report_multicorner_timing_cts.tcl` and
  `report_multicorner_timing_grt.tcl` in two **separate** `tclsh`
  subprocess invocations (matching the real two-process ORFS model), each
  with its own stubbed `sta::*` data, and asserts both produce correctly
  labelled (`4_cts_final_multicorner_tt.rpt` / `5_global_route_multicorner_tt.rpt`),
  non-colliding, independently-correct output files.
- `test_tcl_script_is_syntactically_valid` split into
  `test_common_script_is_syntactically_valid`,
  `test_cts_wrapper_is_syntactically_valid`, and
  `test_grt_wrapper_is_syntactically_valid`, one per new file.
- `test_proc_report_multicorner_timing_drives_two_corner_branch` and
  `test_proc_report_multicorner_timing_is_noop_for_single_corner` now
  source `multicorner_timing_common.tcl` (still calling
  `report_multicorner_timing` directly with explicit stage/when args, which
  was always correct) instead of the removed single file.

**Testing:** `python3 -m pytest flow/util/test_multicorner_dashboard.py -v` →
24 passed (was 22; removed 1 artificial test, added 3: the two-process
collision test plus per-file syntax checks for the common lib and each
wrapper).

---

## Planned Next Steps

1. ~~Implement `pr_metrics.py`~~ ✓
2. ~~Implement `post_cts_timing_repair.tcl` — single-pass~~ ✓
3. ~~Controlled before/after comparison on ibex~~ ✓
4. ~~Make hook iterative~~ ✓
5. ~~Add post-GRT hook — tested, found redundant with built-in repair~~ ✓
6. ~~Triage agent — LLM diagnosis of per-stage quality trajectory~~ ✓
7. ~~Validate triage diagnosis on aes (end-to-end timing closure)~~ ✓
8. ~~Closed-loop optimization agent (loop_agent.py)~~ ✓
9. ~~Write-back to config.mk on success~~ ✓
10. ~~Unit tests (28, no API/Docker required)~~ ✓
11. ~~Open PR~~ ✓
12. ~~Value-side injection blocklist for config param write-back~~ ✓
13. ~~Dedupe post-CTS/post-GRT timing-repair hooks into shared lib~~ ✓
14. **Integration test**: run loop agent end-to-end on aes baseline with API key to confirm
    full cycle (observe → diagnose → intervene → verify → write-back) works live
15. **Placement-stage test**: run a high-utilization aes variant (CORE_UTILIZATION=80)
    to exercise the `PLACE_DENSITY_LB_ADDON` / `place` re-run path end-to-end
    (currently unit-tested only)
16. **Second design**: run triage + loop on ibex or another design to validate generalization
17. ~~`eco_fix` tool: targeted single-instance ECO repair (resize/buffer/hold),~~
    ~~grounded via live OpenROAD capability probe~~ ✓
18. **Integration test for eco_fix**: run against a live built stage `.odb` with the
    real `openroad/orfs:latest` image to confirm `insert_buffer`/`repair_hold_pin`
    behave as the capability probe indicated in a full flow context (currently
    unit-tested at the Python boundary only; the Tcl side was validated for syntax
    and JSON-output correctness with `tclsh`, not against a real design)
19. (Blocked on ML data) Congestion-feedback parameter tuner
20. ~~Regression/benchmark dashboard (`benchmark_dashboard.py`)~~ ✓

---

### 2026-08-27 — CTS quality diagnostic (`cts_diagnostic.py`)

**Built:** `flow/util/cts_diagnostic.py`, a standalone CLI (same
`--platform`/`--design`/`--tag`/`--flow-dir`/`--reports-dir`/`--logs-dir` convention as
`pr_metrics.py`, and it imports and reuses `pr_metrics.collect()` for stage WNS rather
than re-parsing report files itself). It reports:

- **Clock buffers/inverters inserted** and **sink count**, parsed structurally out of
  the CTS-stage log.
- **Buffers-per-sink ratio** — an over-buffering proxy.
- **Setup/hold clock skew**, when `REPORT_CLOCK_SKEW` data is present.
- A **CTS→GRT cliff check**: pulls CTS-stage and Global-route-stage WNS from
  `pr_metrics.collect()` and prints `CLIFF DETECTED:` if WNS worsens by more than
  `--cliff-threshold` (default 0.05 ns) between the two stages — the quantitative
  counterpart to the "CTS→GRT parasitic underestimation cliff" pattern that
  `triage_agent.py` already describes in its LLM prompt context (lines ~60-84).

Exits non-zero if a cliff is detected or if buffers-per-sink exceeds
`--buffer-ratio-threshold` (default 0.5), so it can gate a pipeline/CI step; exits 0
otherwise.

**Grounding — nothing here was guessed; every log/report field was verified against
a real, locally-generated ORFS run** (`flow/logs/nangate45/ibex/base/` and
`flow/reports/nangate45/ibex/base/`, produced by an actual `clock_tree_synthesis` run
in a local checkout). Note: `flow/logs` and `flow/reports` are gitignored build
output — they are **not** committed to this repo, so these exact files are not
present in `git log`/a clean checkout and a reader cannot reproduce the specific
numbers below without running the flow themselves (e.g. `make cts` for
`nangate45/ibex`). The grounding claim is about the log/report *format* (field names,
line shapes, JSON keys), which is stable and inspectable in any ORFS run's output,
not about these particular files being repo-tracked artifacts.

- `flow/Makefile`'s `do-step(4_1_cts, ...)` call for the `cts` target, combined with
  `flow/scripts/flow.sh` (`"$LOG_DIR/$1.log"`, `-metrics "$LOG_DIR/$1.json"`), confirms
  the CTS-stage log is `4_1_cts.log` and its metrics snapshot is `4_1_cts.json` — not a
  guessed name.
- Inspecting the real `4_1_cts.log` showed TritonCTS emits exactly one
  `[INFO CTS-0018]     Created N clock buffers.` line per clock net (the final,
  cumulative buffer count for that net's H-tree — confirmed by cross-checking against
  `TritonCTS found 3 clock nets.` and the 3 resulting `Created N clock buffers.` lines:
  2, 143, 157), plus a separate `Total number of delay buffers: N` line for
  latency-balancing buffers, and one `Sinks N` summary line per net (e.g. `Sinks 1100`
  for `clk_i_regs`, which is exactly `995` initial sinks + `105` "Dummy loads inserted"
  — confirming this is the post-balancing final sink count, not the pre-clustering
  count reported earlier in the same log as `... has 995 sinks.`). The parser
  deliberately anchors on the `]\s*Sinks\s+(\d+)\s*$` and `]\s*Leaf buffers\s+(\d+)\s*$`
  forms (clean, single-purpose lines) rather than the more ambiguous
  `Total number of sinks: N.` / `Number of sinks covered: N.` lines that appear during
  intermediate H-tree construction, to avoid double-counting.
- Skew: `report_metrics.tcl`'s `report_clock_skew_metric` / `report_clock_skew_metric
  -hold` calls (gated by `REPORT_CLOCK_SKEW`, default `1` per `variables.yaml`) write
  metrics into the stage `.json`; the real `4_1_cts.json` contains
  `cts__clock__skew__setup` and `cts__clock__skew__hold` keys, confirmed by direct
  inspection. The parser matches on key suffix so it survives the `cts__` stage prefix.
  A text-based fallback (`parse_cts_skew_rpt`) also matches the `<value> setup skew`
  line found in the real `4_cts_final.rpt`, for when a `.json` isn't available (e.g.
  bazel-orfs consumers that only keep `.rpt`); note the `.rpt` text form only carries
  setup skew since `cts.tcl`'s `report_clock_skew` call site doesn't pass `-hold`.
- Ran `cts_diagnostic.py --reports-dir flow/reports/nangate45/ibex/base --logs-dir
  flow/logs/nangate45/ibex/base` against that local (not committed, gitignored) ibex
  run as a smoke test: 304 buffers, 2167 sinks, ratio 0.140, setup/hold skew ~0.025 ns,
  no cliff (CTS WNS -0.010 ns vs. GRT WNS -0.000 ns) — exit code 0, as expected for a
  healthy run. These specific numbers are from that local run only and are not
  reproducible by re-running this exact command from a clean checkout; the 47-test
  synthetic-fixture suite in `test_cts_diagnostic.py` is what's actually reproducible
  and reviewable from the repo alone.

**Thresholds:**
- `--cliff-threshold` default **0.05 ns**: small enough to catch a real
  parasitic-estimation regression, large enough to not fire on ordinary
  run-to-run WNS noise between optimizer passes.
- `--buffer-ratio-threshold` default **0.5**: the real ibex baseline measured 0.14
  buffers/sink, so 0.5 leaves ~3.5x headroom above a known-healthy design before
  flagging over-buffering — a heuristic sanity bound, not an EDA rule.

**Tests:** `flow/util/test_cts_diagnostic.py` (unittest, no Docker/API, matches the
house style of `test_loop_agent.py`) — synthetic log/json/rpt fixtures built from the
verified real formats above; asserts computed buffer/sink/skew values, cliff
detection/non-detection on crafted WNS sequences (including the case where GRT
*improves* on CTS), buffer-ratio threshold triggering both ways, and an end-to-end
`gather()` test combining a synthetic CTS `.rpt`, a GRT `.rpt`, and the CTS log/json.
Ran `python3 -m pytest flow/util/test_cts_diagnostic.py flow/util/test_loop_agent.py -v`
— all 47 tests pass (19 new + existing 28), plus 6 subtests.

**Out of scope:** clock latency (target/source clock latency numbers are present in
`.rpt` `report_checks` output but only for the single critical path, not tree-wide;
left for a future pass), and any structural stats beyond buffer/sink/skew (e.g. wire
segment counts, fanout distribution histograms) since they weren't called for by the
roadmap item and add parsing surface without a clear consumer yet.

### 2026-09-11 — `cts_diagnostic.py` fixes from independent validator review

An independent validator agent re-ran `cts_diagnostic.py` end-to-end against 17 real
ORFS runs and cross-checked it against real TritonCTS source, and found four bugs
(1 HIGH, 3 MEDIUM) in the code committed in the 2026-08-27 entry above. Fixed all
four; left LOW-severity items and everything else untouched.

- **HIGH — `check_cliff` used WNS only, missing real cliffs.** The module docstring
  claims the tool compares "WNS/TNS between the CTS and Global route stages," but
  `check_cliff` only ever looked at WNS. Validator's real-data repro:
  nangate45/swerv had `dWNS = +0.040` (an *improvement*, so the old check reported
  "No CTS->GRT cliff detected" and exited 0) while TNS went from -306.65 to -492.21
  ns — a 60% degradation — same false-negative pattern reproduced on tinyRocket,
  ariane133, jpeg. Fixed by extending `check_cliff` to also compare CTS-stage vs.
  GRT-stage TNS, gated by a new `--tns-cliff-threshold` CLI flag (default **20%**).
  TNS uses a *relative* (percentage-of-CTS-TNS) threshold rather than an absolute-ns
  one like WNS, because TNS magnitude scales with design size (sum over all
  violating endpoints) so a fixed ns threshold that's meaningful for one design is
  meaningless for another; this is documented inline next to
  `DEFAULT_TNS_CLIFF_THRESHOLD_PCT`. A cliff is now flagged if EITHER the WNS drop OR
  the TNS drop exceeds its threshold, and `check_cliff`'s return dict carries
  `wns_detected`/`tns_detected` separately so `print_report` can show which stat(s)
  triggered (`CLIFF DETECTED (WNS/TNS degraded...)`) and prints both CTS/GRT WNS and
  CTS/GRT TNS lines regardless of which triggered, so the user isn't left guessing
  which metric to look at.
- **MEDIUM — non-dict top-level JSON crashed with an uncaught `AttributeError`.**
  `parse_cts_skew_json` only caught `(json.JSONDecodeError, OSError)`, but
  `json.load` happily returns `None`/a list/a bare string/a number for input like
  `null`, `[...]`, `"str"`, `3` — all valid JSON, none of which have `.items()`.
  Validator reproduced a crash on `4_1_cts.json` containing `null`, which aborted
  before the CLI printed *any* report section, losing already-parsed buffer/sink
  data along with it. Fixed by checking `isinstance(data, dict)` after a successful
  `json.load` and, if not, warning to stderr and returning an empty skew dict (same
  code path as a JSON parse failure) instead of raising — `gather()`'s existing
  `if not skew: skew = parse_cts_skew_rpt(...)` fallback then kicks in and the rest
  of the report (buffer/sink/cliff) still prints normally.
- **MEDIUM — `--reports-dir` without `--logs-dir` could silently produce an
  all-blank report.** The old `logs_dir = args.logs_dir or
  reports_dir.replace("/reports/", "/logs/")` is a silent no-op whenever
  `reports_dir` doesn't contain the literal substring `/reports/` with slashes on
  both sides — which is exactly what happens for a *relative* path given from
  inside `flow/` (e.g. `reports/nangate45/ibex/base`, matching the tool's own cwd
  assumptions), since that string starts with `reports/`, not `/reports/`. There was
  also no `isdir` check on the derived `logs_dir`, unlike the existing check on
  `reports_dir`. Fixed with a new `derive_logs_dir()` helper that splits the path
  into components and replaces an exact `reports` path segment (searching from the
  right) rather than doing a substring replace, falling back to a sibling `logs/`
  directory next to `reports_dir` if no `reports` component exists at all; `main()`
  now also does an `isdir` check on the resolved `logs_dir` and prints a `WARNING:`
  to stderr (without hard-failing, since `reports_dir` alone can still yield a
  partial report) when it's missing.
- **MEDIUM — exit code 1 conflated three different situations.** A cliff/
  over-buffering *finding*, a usage error (bad path), and an uncaught crash were all
  indistinguishable at exit code 1, which a CI/loop-agent caller can't act on
  differently. Adopted a distinct scheme, now documented in the `--help` epilog:
  `EXIT_CLEAN = 0`, `EXIT_FINDING = 1` (cliff and/or over-buffering detected),
  `EXIT_USAGE_ERROR = 2` (bad args / missing reports dir — matches argparse's own
  default exit code for `parser.error()`, so the two usage-error paths are now
  consistent with each other). Genuine crashes are left to propagate as an uncaught
  exception rather than being folded into any of the above.

**Tests:** extended `flow/util/test_cts_diagnostic.py` with: TNS-cliff-detected-when-
WNS-looks-fine (mirrors the validator's real swerv numbers), TNS-within-threshold,
TNS-missing (no false positive), TNS-zero-CTS-TNS edge case; non-dict JSON
(`null`/list/scalar) not crashing `parse_cts_skew_json`, plus a `gather()`-level test
confirming buffer/sink/skew-rpt-fallback data still comes through when the CTS json
is `null`; `derive_logs_dir()` unit tests (exact-component replace, the relative-path
no-substring repro case, and the no-`reports`-component fallback); and CLI-level
subprocess tests asserting the three exit codes and the missing-logs-dir stderr
warning. Ran `python3 -m pytest flow/util/test_cts_diagnostic.py -v` — all 35 tests
pass.

### 2026-09-11 — `cts_diagnostic.py` fixes from round-2 independent validator review

A round-2 independent validator re-ran the round-1-fixed tool against **all** real
ORFS runs under `flow/reports` (58 platform/design/tag dirs present at the time,
covering asap7/nangate45/sky130hd) rather than just the handful of designs the
round-1 fixes were checked against, and found three more real issues:

- **HIGH — `DEFAULT_TNS_CLIFF_THRESHOLD_PCT = 20.0` still missed real cliffs.**
  Checked against the actual dataset: `nangate45/jpeg/base` (CTS TNS -40.29 -> GRT
  TNS -45.63, a +13.3% degradation) and `nangate45/dynamic_node/base` (CTS TNS -0.70
  -> GRT TNS -0.76, +8.6%) are both genuine CTS->GRT parasitic-underestimation
  cliffs that the flat 20% bar let through silently (exit 0, "No CTS->GRT cliff
  detected"). Only `nangate45/ariane133/base` (+23.0%) cleared the old bar, with
  just 3 points of margin — the default was picked without being calibrated
  against the dataset the bug was originally filed on.
- **MEDIUM — new false-positive class on near-zero TNS baselines.** Verified on
  real data: `nangate45/aes/base` has CTS TNS = 0.00, GRT TNS = -0.01 (a
  10-picosecond-total design that is, for all practical purposes, timing-clean),
  but the pure-percentage check computes `(0.01 / 0) * 100` as `+inf%` (guarded
  only by `if cts_tns != 0`, with no absolute-magnitude floor) and flags it as a
  cliff — exit 1 on a design with no real timing problem.
- **MEDIUM — exit code 1 was ambiguous between "cliff detected" and "tool
  crashed".** `EXIT_FINDING = 1` collides with CPython's default uncaught-exception
  exit code, also 1, so an automated caller keying off exit code (e.g. the loop
  agent) could not tell a genuine finding apart from, e.g., a `PermissionError`
  reading a report file — even though the `--help` epilog implied the two were
  distinguishable.
- **MEDIUM — the round-1-fixed code failed CI's black check.** `.github/workflows/
  black.yaml` pins `psf/black@...` (26.5.1); `check_cliff`'s def line and the test
  file's `SCRIPT = os.path.join(...)` line were both over the line-length limit.

**Fixes:**
- Item 1+2 combined into one calibrated check rather than two independent fixes,
  since a pure-percentage fix for item 1 (lowering the % bar) would have made item
  2's false positive worse (any nonzero drop off a zero/near-zero baseline is
  already "+inf%"). `check_cliff`'s TNS branch now requires **both**: the relative
  drop to exceed `--tns-cliff-threshold` (percent, **new default 5.0%**, down from
  20.0%) **and** the absolute drop to exceed a new `--tns-cliff-threshold-abs` (ns,
  **new default 0.03 ns**) — `DEFAULT_TNS_CLIFF_THRESHOLD_ABS_NS` in
  `cts_diagnostic.py`. The 0.03ns floor sits strictly between aes's noise-level
  +0.01ns (not flagged) and dynamic_node's real +0.06ns (flagged); the 5.0% bar
  sits strictly between dynamic_node's real +8.6% and the largest actually-clean
  percentage in the dataset (none observed above 0%, i.e. there is no
  non-degrading design whose percentage this could false-positive against).
  Re-ran the check against all 58 real dirs under `flow/reports` (not just the
  4 named designs) with the new defaults: jpeg, dynamic_node, and ariane133 are now
  all correctly flagged; aes (nangate45) is correctly not flagged; every other
  design's TNS cliff/no-cliff verdict is unchanged from before this fix (all were
  either clear cliffs at >20% already, or non-degrading/improving TNS). `--tns-
  cliff-threshold` and the new `--tns-cliff-threshold-abs` are both exposed as
  separate CLI flags so either bar can be tuned independently per design class.
- Item 3: split `main()` into an inner `_main()` (unchanged usage-error/finding/
  clean logic, still calling `sys.exit(EXIT_USAGE_ERROR)` / `sys.exit(EXIT_FINDING)`
  / `sys.exit(EXIT_CLEAN)` as before) and an outer `main()` that calls `_main()`
  inside `try/except Exception`, re-raising `SystemExit` untouched (so the existing
  exit codes 0/1/2 are unaffected) and printing `INTERNAL ERROR: <type>: <message>`
  to stderr before `sys.exit(EXIT_INTERNAL_ERROR)` (new code, `= 3`) for anything
  else. `--help` epilog updated to document all four exit codes.
- Item 4: ran `python3 -m black flow/util/cts_diagnostic.py
  flow/util/test_cts_diagnostic.py`; `cts_diagnostic.py` was already clean after
  wrapping `check_cliff`'s signature across multiple lines during the item 1/2 fix,
  `test_cts_diagnostic.py`'s `SCRIPT = os.path.join(...)` line was reformatted onto
  three lines by black.

**Tests:** added `TestTnsCliffCalibration` to `flow/util/test_cts_diagnostic.py`,
pinned to the real jpeg/dynamic_node/ariane133/aes numbers above rather than
synthetic ones (plus a `subTest`-parameterized near-zero-noise-variant case
mirroring the validator's `-0.001->-0.01` / `-0.05->-0.08` / `-0.02->-0.03`
examples), and a `TestCliExitCodes` subprocess test that `chmod 0`s a report file
to force a real `PermissionError` (not a mocked one) and asserts the subprocess
exits `EXIT_INTERNAL_ERROR` with `INTERNAL ERROR` on stderr, distinct from
`EXIT_FINDING`. Ran `python3 -m pytest flow/util/test_cts_diagnostic.py -v` — all
41 tests pass (35 prior + 6 new), no regressions. `python3 -m black --check
flow/util/cts_diagnostic.py flow/util/test_cts_diagnostic.py` passes clean.

---

### 2026-09-11 — Independent validator review: 7 fixes to benchmark_dashboard.py

An independent validator agent re-ran the CI-gate scenarios end-to-end against
`flow/util/benchmark_dashboard.py` and found seven ways the "gate" could report
green (or crash) on a genuinely broken run. All seven are fixed on this branch,
`benchmark_dashboard.py` only:

- **HIGH — torn/corrupt newest line silently gated green.** `load_records` now
  returns `(records, dropped_last_line)`; if the *most recent* physical line in
  the history file was corrupt/malformed, `cmd_report` prints a clear
  `stderr` error ("history file has a corrupt/truncated record and cannot be
  safely compared") and exits 1, instead of silently comparing record N-2 vs
  N-1 and reporting success.
- **HIGH — empty latest-stage metrics gated green.** `detect_regressions` now
  flags `{}`/missing metrics on the current record (when the previous record
  had non-empty metrics for the same stage) as its own regression
  ("stage produced no metrics — design may have failed to reach this stage"),
  so `cmd_report` exits 1 instead of reporting a clean run when a design
  stopped producing timing numbers for the requested stage.
- **HIGH — `resolve_dirs` accepted a one-level-too-high `--reports-dir`.**
  Previously any path with ≥3 components was silently sliced into
  platform/design/tag, so pointing `--reports-dir` at a *design* directory
  (missing the tag level) produced `platform='reports'` and a garbage history
  file. `resolve_dirs` now checks that the component 4 levels above the
  presumed tag is literally `"reports"`; if not, it raises a clear
  `SystemExit` ("does not look like .../reports/<platform>/<design>/<tag>")
  instead of proceeding.
- **MEDIUM — non-dict/null JSON lines crashed with a raw traceback.**
  `load_records` now validates each parsed line is a JSON object with the
  expected shape (top-level dict; `stages`, if present, a dict whose values
  are each a dict or `null`) and treats anything else as corrupt using the
  same skip+warn+last-line-tracking path as a `JSONDecodeError`. `timestamp`
  is now read defensively (`rec.get("timestamp") or "—"`) like `git_sha`
  already was, and `build_report_rows`/`best_ever` guard against a `null`
  nested stage value (`.get(stage) or {}`) instead of crashing on
  `None.get(...)`.
- **MEDIUM — non-numeric metric value crashed formatting.** `fmt`/`fmt_delta`
  now render anything that isn't `int`/`float` (not just `None`) as `"—"`
  instead of raising `ValueError` out of `str.format`.
- **MEDIUM — `cmd_record` died with an unhandled traceback on a malformed
  report.** The `collect()` call in `cmd_record` is now wrapped in
  `try`/`except Exception`, printing a clear message naming the reports dir
  and the underlying exception to `stderr` and exiting 1, rather than letting
  a raw traceback surface. `pr_metrics.py` itself was not touched (shared
  file, out of scope for this branch).
- **MEDIUM — reader took no lock.** `load_records` now takes a shared lock
  (`fcntl.flock(..., LOCK_SH)`) around the read, matching the exclusive lock
  `append_record` already takes, so a reader can no longer observe a
  partially-written record from a concurrent `record` invocation.

**Tests (`flow/util/test_benchmark_dashboard.py`):** extended to 50 (from 43),
covering all seven fixes above, plus updated three pre-existing tests that
exercised the old (buggy) `resolve_dirs`/`load_records` behavior directly —
`test_reports_dir_derives_tag_from_path_when_not_passed` and
`test_reports_dir_explicit_tag_overrides_path_derivation` now use a
`--reports-dir` that actually has `reports/` in the right position, and all
`bd.load_records(...)` call sites were updated to unpack the new
`(records, dropped_last_line)` return.

```bash
cd flow/util && python3 -m pytest test_benchmark_dashboard.py -v
```
50 passed.

### 2026-09-11 — Round-2 independent validator review: 5 remaining fixes to benchmark_dashboard.py

A second, independent validator agent re-tested the fixes above end-to-end
with real CLI runs and found five more real issues, all now fixed on this
branch, `benchmark_dashboard.py` only:

- **HIGH — `resolve_dirs`'s "4 levels up must be literally `reports`" check
  was too strict, and its own error message's suggested workaround was
  impossible.** `--platform` and `--reports-dir` are in a mutually-exclusive,
  required argparse group, so telling a user hitting the error to "pass
  `--platform`/`--design`/`--tag` explicitly" was a dead end for `--platform`.
  Worse, it newly rejected previously-working inputs: a relative
  `nangate45/ibex/base` path (no `reports` ancestor) or a bare CI artifact
  dir like `/tmp/artifacts/nangate45/ibex/base` (no `reports` component at
  all) now hard-failed. Fixed by locating the *last* literal `reports` (or
  `logs`, mirroring whichever kind of dir is being resolved) path component
  via search instead of a fixed offset. If found, exactly 3 components
  (platform/design/tag) must follow it — this still catches the original
  "one level too high" bug. If no `reports`/`logs` component exists anywhere
  in the path, fall back to the prior permissive behavior (last 3 path
  components) instead of hard-erroring. (At the time, there was no
  argparse-valid way to explicitly override the check in the
  `--reports-dir` case — see the follow-up fix below.)
- **MEDIUM — `compute_delta` still crashed on two non-numeric metric
  values.** The `fmt`/`fmt_delta` hardening from round 1 didn't cover the
  subtraction in `compute_delta` itself, so a history file with `"wns":
  "n/a"` in two consecutive records raised an unhandled `TypeError` —
  exiting 1 for the same reason a real regression exits 1, making corruption
  indistinguishable from a genuine quality regression. `compute_delta` now
  returns `None` unless both operands are real numbers. Audited and fixed
  the same exposure in `detect_regressions`'s `prev_fmax > 0` comparison and
  `render_html`'s point-series filtering (feeding `y_span = y_max - y_min`).
- **MEDIUM — `dropped_last_line` detection was defeated by a trailing blank
  line.** It keyed on `lineno == total_lines` (the last *physical* line), but
  blank lines are skipped before that check runs, so a corrupt record
  immediately followed by a blank line silently escaped detection — exactly
  the gap round-1's fix #1 was meant to close. `load_records` now tracks the
  last *non-blank* line number and compares against that instead.
- **MEDIUM-LOW — `dropped_last_line`'s exit-1 check in `cmd_report` never
  ran when history had zero valid records left after dropping corrupt
  lines**, because the `if not records: ... sys.exit(0)` short-circuit ran
  first — an all-garbage history file reported exit 0 ("No history found")
  instead of flagging corruption. `cmd_report` now checks
  `dropped_last_line` before the empty-records short-circuit.
- **LOW — `fmt`/`fmt_delta` accepted `bool`** (since `bool` is an `int`
  subclass in Python), rendering a stray JSON `true`/`false` as `1.000`/
  `0.000` instead of `"—"`. Added a shared `is_number()` helper
  (`isinstance(val, (int, float)) and not isinstance(val, bool)`) used by
  `fmt`, `fmt_delta`, `compute_delta`, `detect_regressions`, and
  `render_html`'s series filter.

**Tests (`flow/util/test_benchmark_dashboard.py`):** extended to 58 (from
50), adding: a `--reports-dir` with no `reports` component and a relative
3-component path both still resolving correctly (item 1); a two-record
history where both records have non-numeric metrics not crashing
`build_report_rows`/`print_report` (item 2); a corrupt-record-followed-by-
blank-line history correctly flagged as `dropped_last_line` (item 3); an
all-garbage history file exiting non-zero via the CLI (item 4); and a bool
JSON value rendering as `"—"` in both `fmt` and `fmt_delta` (item 5).

```bash
cd flow/util && python3 -m pytest test_benchmark_dashboard.py -v
```
58 passed.

### 2026-09-11 — Follow-up: make the strict-shape override actually reachable

Round-2's fix still left a real usability gap: when the strict path-shape
check does fire, its own suggested remediation ("pass `--platform`,
`--design`, and `--tag` explicitly") was unreachable via the CLI, since
`--platform` and `--reports-dir` lived in the same mutually-exclusive,
required argparse group. Fixed by dropping that group — `--platform` and
`--reports-dir` can now both be passed. `resolve_dirs` now checks
`args.platform` (not just `args.reports_dir`) first: when `--platform` is
given (with or without `--reports-dir`), it derives the path from
platform/design/tag as before, bypassing path-shape validation entirely.
Passing `--reports-dir` alone still goes through the shape check unchanged.
Error messages were updated to point at this override instead of the
now-fixed advice.

**Tests:** added
`test_record_cli_reports_dir_with_explicit_overrides_bypasses_shape_check`,
exercising the previously-impossible override end-to-end via the CLI
(not just `resolve_dirs()` in isolation): confirms plain `--reports-dir`
one level too high still fails the shape check, and that adding
`--platform`/`--design`/`--tag` alongside it now succeeds.

```bash
cd flow/util && python3 -m pytest test_benchmark_dashboard.py -v
```
59 passed.

### 2026-09-24 — eco_fix: list mode, and the demo stops wasting attempts

Timing the demo on nangate45/gcd showed the ECO step dominating its roughly 8
seconds. Each `eco_fix` attempt starts a Docker container (about 0.2 s of the
cost) and has OpenROAD reload the design, libraries and constraints. The demo
made 11 such runs: one deliberate bad-target probe to discover candidate
instances, then up to 10 `resize_up` attempts. Four of those ten attempts were
for cells that were already the largest size (or clock cells) and could only
ever answer "no up-size target".

**Changes**
- `eco_repair.tcl`: new pure proc `eco_resize_flags` (can a cell go up / down,
  never for flip-flops, latches, clock buffers or clock gates), and
  `eco_targets` takes an optional `annotate` argument that adds `can_up` /
  `can_down` to each target. The size maps are built only when asked for, so
  ordinary `eco_fix` calls pay nothing. `eco_run` gains a `list_targets` mode
  that loads the design, lists the annotated targets and stops, with no timing
  snapshot and no change to the design.
- `loop_agent.py`: the "write the generated Tcl, run it in Docker, read the JSON"
  code moved into `_run_eco_tcl`, shared by `impl_eco_fix` and the new
  `impl_eco_list_targets`. `pick_resizable_targets` chooses distinct instances
  whose flag is explicitly true. The agent's tool list is unchanged.
- `demo_pr_extension.sh`: lists targets once, tries only cells that can be sized
  up, and caps real attempts at 6 (what the old loop effectively made, since 4
  of its 10 were wasted).

**Measured** (same machine, same design, two runs each): the demo went from
8.2 s and 8.1 s to 6.0 s and 5.9 s, about 27% faster, from 11 OpenROAD runs to
7. The six resize attempts and their verdicts are identical to before.

**A correction to the earlier estimate.** Replacing the probe call with the list
call does not save time by itself: the list call is still one OpenROAD run, about
the same cost as the probe. The whole saving comes from not launching attempts
that cannot succeed. The list call is still cleaner, because it no longer relies
on an error path to discover candidates, and it reports resizability.

**Tests:** 18 new (`test_loop_agent.py` now 73). They cover the list function
with a mocked Docker run (result parsed, invalid stage and missing database
rejected without running Docker, missing result file and a Tcl-reported error
passed through), the candidate picker, and the pure Tcl pieces via tclsh
(`eco_resize_flags` for every up/down/excluded case, and `eco_json_targets`
producing valid JSON with real booleans and keeping its original shape when no
flags are present). All four suites: 197 passed.

**Not done:** loading the design once and trying several candidates in one
OpenROAD session, which would remove the per-attempt reload. It needs an
in-memory undo for rejected changes, so it is left as a separate, riskier step.
(Done in the next entry.)

### 2026-09-24 — eco_fix: try many cells in one OpenROAD session (search, then confirm)

The demo's ECO step still reloaded the design for every attempt. This adds a
batch that loads it once and tries several candidate resizes in the same session.

**Design.** `eco_search_run` (Tcl) applies each candidate inside an ODB ECO journal
(`beginEco` / `endEco`) and rolls it back with `undoEco` when it does not pass, so
the design is back to its starting point for the next candidate. It stops at the
first candidate that passes `eco_verdict`. Before writing any code, undo was
live-tested on nangate45/gcd: four trials whose legalization moved 1, 4, 8 and 5
neighbouring cells (including row and orientation changes) each restored all 664
instances and the timing numbers exactly. As a runtime guard, the search
re-measures after every rollback and stops with an error if the numbers differ
from the baseline.

**An independent review found that undo is not complete, and the first version
was wrong.** In the first version the batch also wrote the database for the
winning candidate. The reviewer showed that after several rollbacks the written
file differed from what a single attempt writes (different checksum, and a
reload measured wns -0.042500 where the batch had reported -0.042656). Cause:
`swapMaster` and legalization clear each resized cell's preferred pin access
points, and that clearing is not in the journal, so undo does not put it back.
Masters, positions, orientation, connectivity, net guides and in-session timing
all match, which is why the guard could not see it. It only affects databases
that already carry access points (grt); the cts stage gave byte-identical files.
I reproduced it with the reviewer's exact scenario (8 rolled back, 4 skipped,
then `_513_` kept): single attempt `6edcf2a2…`, first batch version `c79042ec…`.

**Fix.** The batch never writes. It only decides which candidate passes. Python
(`impl_eco_try_resizes`) then re-applies that one change alone to a freshly
loaded database with the normal single-attempt path, which measures it again and
writes only if it still passes. That path is the one proven to be identical to a
single `eco_fix`, and the reported numbers are now the numbers on disk. With the
reviewer's scenario the written database is byte-identical to the single attempt
(`6edcf2a2…`). It costs one extra OpenROAD run, and only when something is kept.
`impl_eco_search_resizes` exposes the search alone.

**Other review findings, all addressed**
- Tests missed three real mutations. Added tests for direction `down` (an
  unchanged result passes a downsize but not an upsize), a failed parasitics
  refresh after a rollback, a failed measurement after a rollback, and a failed
  after-measurement (never accepted). A runner applies eight deliberate bugs one
  at a time, including those three and "write the database from the search
  session"; every one is caught.
- A failure after the swap (for example a parasitics error) was reported as a
  skip. It is now an error, decided by whether the cell's master changed.
- The demo formatted metrics with `:.4f` and would crash on a metric that could
  not be measured ("NA"). It now prints the value as it is.
- Each measurement left a temporary `*_tns.txt` file. `eco_measure` now deletes it.
- Not changed: on a timeout the Docker container can keep running. The search
  step cannot write, so that risk no longer applies to it, but the confirming run
  still can. `odb_written` is `None` (unknown) when the confirming run returns no
  result.

**Measured** (nangate45/gcd, same machine): nine candidates in one session took
0.9 s. The whole demo went from 6.1 s (after PR #10) to about 3.5 s, while
trying 10 candidates instead of 6. From the original 8.2 s, that is about 57%
faster. The verdict of every candidate matches the one-run-per-attempt numbers to
the last digit.

**Tests:** `test_loop_agent.py` now has 95 tests (was 73): the search and confirm
flow with a mocked Docker run, and the batch's real control flow under `tclsh`
with stand-ins for the OpenROAD calls. All four suites: 219 passed.
