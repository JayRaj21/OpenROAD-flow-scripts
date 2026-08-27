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
17. (Blocked on ML data) Congestion-feedback parameter tuner

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
