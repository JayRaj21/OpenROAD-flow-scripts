# P&R Quality Diagnosis and Repair Tools

This guide covers a set of command-line tools that help you understand why a
place-and-route run has poor timing, track quality across runs, and try small,
checked repairs. They live in `flow/util/` and `flow/scripts/` and work on the
files a normal flow run already produces. Most of them only read those files and
change nothing.

| Tool | What it answers | Needs |
|---|---|---|
| `pr_metrics.py` | How do timing, wirelength and congestion change from stage to stage? | A finished run |
| `cts_diagnostic.py` | Is the clock tree healthy, and does timing fall apart between CTS and global route? | A finished run |
| `benchmark_dashboard.py` | Did this run get worse than the last one? | A finished run, run repeatedly |
| `multicorner_dashboard.py` | Which timing corner is the worst? | A multi-corner build with hooks on |
| `eco_fix` | Does upsizing or downsizing this one cell help? | Docker and a finished run |
| `triage_agent.py` | Why is timing bad, and what should I change? | An Anthropic API key |
| `loop_agent.py` | Can an LLM fix timing on its own? | An API key and Docker |

There is also a demo script (described near the end) that runs most of them in order.

## Terms used in the results

- **WNS** (worst negative slack): the timing slack of the single worst path, in
  nanoseconds. Zero or positive means every path meets timing. More negative is
  worse.
- **TNS** (total negative slack): the sum of the negative slack over all failing
  endpoints. It grows with the number of failing paths, so it shows how *widespread*
  a problem is, where WNS shows how *bad* the worst one is.
- **Hold slack**: the same idea for hold timing. Positive is good.
- **Fmax**: the highest clock frequency the design could run at, given its
  current worst slack.
- **Stage**: one step of the flow. The tools here use these names, in flow order:
  Global place, Resizer, Detail place, CTS, Global route, Finish.
- **CTS**: clock tree synthesis, the step that builds the clock distribution network.

## Before you start

You need a finished flow run for a design, because the tools read its reports and
logs. To build the small `gcd` example on the `nangate45` platform:

```shell
cd flow
util/docker_shell make DESIGN_CONFIG=designs/nangate45/gcd/config.mk
```

That writes reports to `flow/reports/nangate45/gcd/base/`, logs to
`flow/logs/nangate45/gcd/base/`, and design databases to
`flow/results/nangate45/gcd/base/`. Every tool below takes the same three options
to find them: `--platform`, `--design`, and `--tag` (the variant name, `base` by
default). Run everything from the `flow/` directory.

The examples in this guide use real output from the `nangate45/gcd` design.

A note on units: the tools label timing values as nanoseconds. That is correct for
`nangate45`, but a platform whose timing library uses another unit (for example
picoseconds) will print numbers in that unit under the same label. This has not been
tested on such a platform.

## pr_metrics.py: stage-by-stage metrics

**What it does.** Reads the reports and logs from a finished run and prints one
table showing how timing, wirelength and routing overflow change through the flow.
The other tools build on it.

**Run it.**

```shell
python3 util/pr_metrics.py --platform nangate45 --design gcd
```

**Inputs.** The `.rpt` files in the run's reports folder (`3_global_place.rpt`,
`3_resizer.rpt`, `3_detailed_place.rpt`, `4_cts_final.rpt`, `5_global_route.rpt`,
`6_finish.rpt`) plus two logs: `3_3_place_gp.log` and `5_1_grt.log`. You can point at
folders directly with `--reports-dir` and `--logs-dir` instead.

**Output.** A table printed to the screen. It does not write any files.

```text
Stage              WNS (ns)   TNS (ns)  Worst slack  Fmax (MHz)    HPWL (um)  GRT overflow
Global place         +0.000     +0.000       +0.010      2244.3    5,409,062        0.0000
Resizer              +0.000     +0.000       +0.010      2244.3            —             —
Detail place         +0.000     +0.000       +0.010      2227.5            —             —
CTS                  -0.030     -0.300       -0.030      2041.1            —             —
Global route         -0.040     -0.420       -0.040      1989.4            —             —
Finish                    —          —            —           —            —             —
```

**Reading it.**

- Read down the WNS and TNS columns. The stage where they first turn negative is
  where the problem starts. Here timing is clean through placement, goes negative at
  CTS, and gets worse at global route.
- **HPWL** is the total estimated wire length after global placement. **GRT overflow**
  is how much routing demand exceeded capacity, and it is shown for the Global place
  and Global route rows only. Overflow above zero is a congestion warning.
- A dash means that stage has no report. In the example, `Finish` is empty because
  that build stopped after global route. Dashes in `HPWL` and `GRT overflow` are
  normal: only the two stages named above have those numbers.
- A `Total power` line appears at the bottom when a report includes it.

## cts_diagnostic.py: clock tree health

**What it does.** Two checks. First, it counts the buffers and sinks in the clock
tree and reads the clock skew. Second, it looks for the "CTS-to-global-route cliff":
during CTS the tools estimate wire delay from cell placement, which is optimistic,
and after global route the real routed wires are known. A design can look fine at
CTS and then lose timing at global route.

**Run it.**

```shell
python3 util/cts_diagnostic.py --platform nangate45 --design gcd
```

**Inputs.** The CTS log (`4_1_cts.log`), the CTS metrics file (`4_1_cts.json`, or
`4_cts_final.rpt` if that file is missing), and the CTS and Global route timing from
`pr_metrics`. If the logs folder cannot be found it prints a warning and reports only
the timing part.

**Output.** A report printed to the screen, and an exit code you can use in scripts.

```text
Clock buffers/inverters inserted: 5
Clock sinks:                      38
Buffers per sink:                 0.132
Setup skew (ns):                  0.00104932
----------------------------------------------------------------------
CTS WNS: -0.030 ns   GRT WNS: -0.040 ns   drop: +0.010 ns
CTS TNS: -0.300 ns   GRT TNS: -0.420 ns   drop: +0.120 ns (+40.0%)
CLIFF DETECTED (TNS degraded by more than threshold) between CTS and Global route ...
```

| Exit code | Meaning |
|---|---|
| 0 | Clean |
| 1 | A finding: a cliff, over-buffering, or both |
| 2 | Bad input, for example a missing reports folder |
| 3 | Internal error (a bug) |

**Options.**

| Option | Default | What it controls |
|---|---|---|
| `--cliff-threshold` | 0.05 | WNS must get worse than this many ns between CTS and global route to count as a cliff |
| `--tns-cliff-threshold` | 5 | TNS must get worse by more than this percent to count |
| `--tns-cliff-threshold-abs` | 0.03 | and by more than this many ns (this stops tiny changes on a design that is already clean from looking huge as a percentage) |
| `--buffer-ratio-threshold` | 0.5 | Buffers per sink above this is flagged as over-buffered |

A cliff is reported when WNS crosses its threshold, or when TNS crosses both of its
thresholds.

**Reading it.**

- **Buffers per sink** is a rough measure of how heavy the clock tree is. A value
  above the threshold means the clock tree may be wasting area and power.
- **Skew** is the difference in clock arrival time between endpoints. It is printed as
  a magnitude. Small is better.
- **A cliff** means timing you measured at CTS is not to be trusted, because routed
  wires are slower than the estimate. The tool suggests enabling the post-CTS repair
  hook (`POST_CTS_TCL=post_cts_timing_repair.tcl`) or leaving extra setup margin during
  repair (`SETUP_SLACK_MARGIN`).
- The example above is a real cliff: WNS only moved 0.010 ns, but TNS got 40 percent
  worse. That is why TNS is checked as well as WNS.

**Limits.** The buffer count includes delay buffers inserted for latency balancing,
and the sink count is taken after TritonCTS adds dummy load cells, so the ratio is
approximate. A cliff means timing got worse between the two stages. It does not say
what caused it.

## benchmark_dashboard.py: regression history

**What it does.** Keeps a history of a design's metrics across runs, so you can tell
whether a code or setting change made things worse. It has two commands. `record`
saves the current run's metrics. `report` compares the latest record with the one
before it and flags a regression.

**Run it.**

```shell
python3 util/benchmark_dashboard.py record --platform nangate45 --design gcd
python3 util/benchmark_dashboard.py report --platform nangate45 --design gcd \
    --stage "Global route" --html trend.html
```

Run `record` after each flow run. `report` needs at least two records to compare.

**Inputs.** The same reports and logs as `pr_metrics.py` (it uses that tool to read
them). `report` also reads the history file.

**Output.**

- **The history file**, one JSON line per `record`, at
  `flow/util/benchmark_history/<platform>__<design>__<tag>.jsonl`. Each line holds a
  timestamp, the git commit, and the metrics for every stage. The file is always
  written next to the script, even if you pass `--flow-dir`, and git does not ignore
  it, so it appears as untracked until you commit it or delete it.
- **A table** on the screen showing one row per record.
- **An HTML chart** (with `--html <path>`): a single file with line charts of WNS, Fmax and HPWL and a
  results table. It needs no internet connection.

```text
Timestamp             SHA         WNS (ns)     dWNS  Fmax(MHz)    dFmax  Flags
2026-09-20T21:23:38   d42faa9c      -0.040        —     1989.4        —
2026-09-20T21:23:38   d42faa9c      -0.040   +0.000     1989.4     +0.0
No regressions detected against previous record.
```

**Options.**

| Option | Default | What it controls |
|---|---|---|
| `--stage` | `Finish` | Which stage's numbers to report on. Use `"Global route"` if your build did not run the whole flow, because `Finish` is empty then |
| `--last N` | all | Show only the last N records. This does not change what counts as a regression |
| `--wns-threshold` | 0.01 | A WNS drop of more than this many ns is a regression |
| `--fmax-threshold-pct` | 1.0 | An Fmax drop of more than this percent is a regression |
| `--overflow-threshold` | 0.001 | A routing-overflow increase of more than this is a regression |
| `--html <path>` | none | Also write the HTML chart to this path |

**Reading it.**

- A regression is only judged against the **previous** record. A row can also carry a
  `worse-than-best-...` flag, which means it is worse than the best value ever
  recorded for that metric. That flag is informational and does not affect the exit
  code.
- **Exit code 0:** no regression found, or no history yet. **Exit code 1:** a
  regression was found, the stage stopped producing metrics, or the last line of the
  history file is corrupt. That makes it usable as a CI gate.
- Overflow is only tracked for the Global place and Global route stages.

**Limits.** A record whose stage has no metrics counts as a regression if the
previous record had them, since that usually means a run stopped early. TNS is saved
in the history but is not shown in the table and does not trigger a regression.

## multicorner_dashboard.py: timing per corner

**What it does.** Some designs are timed at several "corners" (for example, a slow
and a fast one). The normal reports only show the combined worst case. This tool
shows the timing of each corner side by side and marks the worst one.

It needs the flow to write a report per corner. That is done by two hook scripts you
switch on for the build.

**Set it up.** Add these lines to the design's `config.mk` for a design that has two
or more corners listed in `CORNERS`:

```make
export REPORT_MULTICORNER_TIMING = 1
export POST_CTS_TCL = $(SCRIPTS_DIR)/report_multicorner_timing_cts.tcl
export POST_GLOBAL_ROUTE_TCL = $(SCRIPTS_DIR)/report_multicorner_timing_grt.tcl
```

Then rebuild the design. In the repository only `ihp-sg13g2/i2c-gpio-expander` has
more than one corner (`CORNERS = slow fast`). The hook files are separate because
CTS and global route run as separate OpenROAD processes, so each needs to label its
own output.

**Run it.**

```shell
python3 util/multicorner_dashboard.py --platform ihp-sg13g2 --design i2c-gpio-expander
python3 util/multicorner_dashboard.py --platform ihp-sg13g2 --design i2c-gpio-expander \
    --stage 5_global_route
```

**Inputs.** Files named `<stage>_multicorner_<corner>.rpt` in the reports folder,
for example `4_cts_final_multicorner_slow.rpt` and `5_global_route_multicorner_fast.rpt`.
Use `--stage` to pick one stage. Without it, the tool uses the latest stage that has
files and prints a warning if two labels are tied.

**Output.** A table with one column per corner and rows for WNS, TNS, worst slack and
clock skew. The worst corner in each row is marked `(worst)`. Nothing is written to
disk.

**Reading it.** For WNS, TNS and worst slack, the lowest (most negative) value is the
worst. For skew, the largest magnitude is the worst. A design can pass in one corner
and fail in another, and this is the view that shows it. Clock skew rows appear only
when clock-skew reporting is on, which it is by default.

**Limits.**

- `POST_CTS_TCL` and `POST_GLOBAL_ROUTE_TCL` each hold one script. You cannot use
  them for these hooks and for the timing repair hooks (`post_cts_timing_repair.tcl`,
  which `loop_agent.py` sets) in the same build. The second setting replaces the first.
- Reports from an earlier run stay in the reports folder. If you drop a corner and
  rebuild, the old corner's file is still there and the table will still show it.
  Delete the old files first.
- The tool was checked against the OpenSTA commands the flow pins, and it is covered
  by unit tests. It has not been run on a real multi-corner build.
- `REPORT_MULTICORNER_TIMING` is not listed in `FlowVariables.md`. It is described here.

## eco_fix: one targeted repair

**What it does.** "ECO" means a small engineering change to a finished design. This
tool tries one such change on a single cell and measures the effect before and after,
in one OpenROAD session. It keeps the change only if it helps and nothing else gets
worse. It takes about a second on a small design, compared with minutes for re-running
a stage.

There are three fix types:

| Fix type | What it does | Aimed at |
|---|---|---|
| `resize_up` | Swaps a cell for the next stronger drive strength | Setup timing (WNS) |
| `resize_down` | Swaps a cell for the next weaker drive strength | Saving area and power without hurting timing |
| `fix_hold` | Repairs a hold violation at one pin by adding delay | Hold timing |

**Run it.** `eco_fix` is a Python function, and the autonomous agent calls it as a
tool. To try it yourself:

```shell
python3 -c "
import itertools
from util.loop_agent import impl_eco_fix
print(impl_eco_fix('resize_up', 'output42', 'grt', '', 'nangate45', 'gcd', 'base', '.', [], itertools.count(1)))
"
```

The arguments are, in order: fix type, target, stage, optional cell name, platform,
design, tag, the `flow/` folder (`.` when you run from there), a list that collects a
change log, and a counter that numbers the runs.

**Inputs.**

- **Fix type:** `resize_up`, `resize_down` or `fix_hold`.
- **Target:** for the resizes, an instance name such as `_646_`. For `fix_hold`, a pin
  name such as `_412_/D`. Names may only contain letters, digits and `_ . / [ ] $ : \ -`.
- **Stage:** `cts` edits `4_cts.odb`, `grt` edits `5_1_grt.odb`. The database for that
  stage must already exist.
- **Cell** (optional): a specific library cell to swap to. It must have the same
  signal pins as the current cell. Without it, the tool picks the next size up or down.

**Not sure what to target?** Call it with a made-up instance name. The result always
lists the instances on the worst setup paths, so you can pick real ones.

**Output.** A result block printed to the screen. Files are also written, with
`<n>` being the run number:

- `flow/objects/<platform>/<design>/<tag>/eco/eco<n>.tcl`: the generated script
- `flow/objects/<platform>/<design>/<tag>/eco/eco<n>.json`: the full result
- `flow/logs/<platform>/<design>/<tag>/eco<n>.log`: the OpenROAD log

Each run replaces the file numbered the same as an earlier run.

```text
status: rejected  (verdict: REJECTED — no improvement in wns)
fix: resize output42 BUF_X1 -> BUF_X2
metric                   before        after      delta
wns                -0.04265615759127782 -0.04245662275253383 0.0001995348387439852
tns                       -0.42        -0.42        0.0
worst_hold_slack   0.06095005228595483 0.06095005228595483        0.0
setup_viol_count             13           13
hold_viol_count               0            0
odb_written: False
```

**When a change is kept.** The rule is the same shape for each fix type: the change
must improve the target metric by enough, and must not make anything else worse by
more than a small allowance.

| Fix type | Must improve | Setup violations may rise by |
|---|---|---|
| `resize_up` | WNS by at least 0.001 ns | 0 |
| `resize_down` | Nothing. WNS must not drop by more than 0.001 ns, and TNS not by more than 0.05 ns | 0 |
| `fix_hold` | Hold slack by at least 0.001 ns. WNS may drop by up to 0.01 ns | 3 |

For all of them, TNS must not drop by more than 0.05 ns, hold slack must not drop by
more than 0.001 ns, and the count of hold violations must not increase. If WNS, TNS or
hold slack cannot be measured, the change is rejected.

**Reading it.**

- **`status: rejected` and `odb_written: False`:** the change was measured and
  not kept. The design database was not touched. This is the normal result on a
  design that is already near closure.
- **`status: applied` and `odb_written: True`:** the change was kept and written into
  the `.odb` file **in place**. The reports (`.rpt`) do not update. To see the new
  numbers in `pr_metrics.py`, re-run the later stages. A later re-run of that stage or
  an earlier one also throws the change away.
- **`status: error`:** the fix could not be tried, and the message says why. Common
  ones: `instance not found`, `cell excluded from resize` (flip-flops, latches, clock
  buffers and clock gates are never resized), `no up-size target` (already the largest size), and `incompatible
  swap` (the requested cell has different pins).

**Limits.**

- It needs Docker and the `openroad/orfs` image, because it runs OpenROAD inside the
  container. A run times out after 15 minutes.
- Buffer insertion was planned as a fourth fix type but is switched off. Testing
  showed that OpenROAD's `insert_buffer` command crashes the whole OpenROAD process on
  every input tried in the shipped image.
- A kept `resize_down` has been verified end to end on a real design. A kept
  `fix_hold` has not, because the example design has no hold violations to fix.
- It does not pick a target for you. The caller decides which cell to try.

## triage_agent.py: LLM diagnosis

**What it does.** Reads the stage-by-stage table from `pr_metrics.py`, sends it to
Claude, and prints a diagnosis: the likely root cause, the evidence, and specific ORFS
settings to try. It changes no files.

**Run it.**

```shell
pip install anthropic
python3 util/triage_agent.py --platform nangate45 --design gcd
```

It needs an Anthropic API key in the environment variable `ANTHROPIC_API_KEY`. A
call costs money. It uses the `claude-opus-5` model.

**Inputs.** The same reports and logs as `pr_metrics.py`. **What is sent:** the
stage table (numbers only), the run label such as `nangate45/gcd/base`, and the tool's
built-in instructions. No design files are sent.

**Output.** Printed to the screen only: the table, then the answer under four
headings (Root cause, Evidence, Recommended actions, Expected outcome), then the token
count.

**Reading it.** Treat it as an experienced colleague's suggestion, not a fact. The
recommendations name real ORFS settings (`SETUP_SLACK_MARGIN`, `TNS_END_PERCENT`,
`PLACE_DENSITY_LB_ADDON`, `OPT_POST_GRT_WNS`, `POST_CTS_TCL`), so you can check them
against what `cts_diagnostic.py` found. For a design like the `gcd` example, both
should point at the CTS-to-global-route cliff.

**Limits.** It sees only the summary numbers, so it cannot notice anything that is
not in them. Its unit tests do not call the live API, so treat the first real run as
a trial.

## loop_agent.py: autonomous fixing

**What it does.** An LLM reads the metrics and then works on its own: it changes
allowed ORFS settings, re-runs flow stages, checks whether timing improved, and repeats.
It can also call `eco_fix` for a single-cell repair. It stops when timing closes
(WNS at or above 0 and TNS at or above -0.05 ns at Finish) or when it runs out of
budget.

**Run it.**

```shell
python3 util/loop_agent.py --platform nangate45 --design gcd
```

It needs `ANTHROPIC_API_KEY`, `pip install anthropic`, and Docker with the
`openroad/orfs` image. It makes several paid API calls.

**What it can and cannot do.**

- It can only change these settings: `SETUP_SLACK_MARGIN`, `TNS_END_PERCENT`,
  `OPT_POST_GRT_WNS`, `PLACE_DENSITY_LB_ADDON`, `POST_CTS_TCL` and
  `POST_GLOBAL_ROUTE_TCL`. Values containing shell or Make syntax are refused.
- It runs at most 20 model turns and is told to use at most 3 improvement iterations.
- Each stage run has a 30-minute limit.

**Output.**

- Progress printed to the screen as it works.
- A change log at `flow/logs/<platform>/<design>/<tag>/loop_agent_changes.json`.
- **Changes to your design's `config.mk`.** If the agent reports success, the settings
  it used are written into `flow/designs/<platform>/<design>/config.mk`. This edits a
  file that git tracks, so run `git diff` afterwards to see what changed. ECO changes
  are not written back; they only live in the `.odb`.

**Things to know before running it.**

- To force a stage to re-run, it **deletes** that stage's output files first (for
  example the `.odb` files for that stage). It rebuilds them, but do not run it on a
  build you cannot afford to redo.
- The model sees the metrics table and the tail of the make output. It does not see
  your design source.
- Its unit tests do not call the live API or run Docker, so treat a first real run as
  a trial and try it on the small `gcd` design first.

## The demo script

`util/demo_pr_extension.sh` runs the unit tests and then `pr_metrics`,
`cts_diagnostic`, `benchmark_dashboard` and `eco_fix` in order on one design, with a
short explanation before each step.

```shell
util/demo_pr_extension.sh --pause      # wait for Enter between steps
util/demo_pr_extension.sh --build      # build the design first if it is missing
util/demo_pr_extension.sh --llm        # also run the triage agent (uses your API key)
```

Each run saves a report and the chart under `flow/util/demo_runs/<platform>-<design>-<tag>/<date-time>/`:
`report.txt` (a copy of everything printed) and `benchmark_dashboard.html`. Git
ignores that folder. The demo puts the design database and benchmark history back the
way it found them.

It leaves out the multi-corner dashboard (which needs the special build above) and
`loop_agent.py` (which edits `config.mk`).

## Common problems

| Message | What it means |
|---|---|
| `reports directory not found` | The run has not been built, or the platform, design or tag is misspelled |
| A `—` in a table | That stage has no report yet, or the number is not produced for that stage |
| `x-api-key header is required` | `ANTHROPIC_API_KEY` is not set in this terminal |
| `No multi-corner reports found` | `REPORT_MULTICORNER_TIMING` was not set, `CORNERS` has one entry, or the hooks are not in `config.mk` |
| `eco_fix` says `not found — run that stage first` | The stage's `.odb` does not exist yet. Build that stage first |
| `error: --reports-dir ... does not look like .../reports/<platform>/<design>/<tag>` | Pass `--platform`, `--design` and `--tag` together with `--reports-dir` |

## Tests

```shell
python3 -m pytest util/test_loop_agent.py util/test_cts_diagnostic.py \
    util/test_multicorner_dashboard.py util/test_benchmark_dashboard.py -q
```

These need no Docker and no API key.
