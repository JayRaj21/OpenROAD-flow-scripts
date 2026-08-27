# report_multicorner_timing.tcl
#
# Additive, opt-in per-corner timing breakdown.
#
# report_metrics.tcl already loops $::env(CORNERS) for report_power (see
# its "report_power" section), but report_tns / report_wns /
# report_worst_slack are called without any per-corner scoping, so only
# the merged worst-case view across corners is ever written to the stage
# .rpt file. This script fills that gap without touching
# report_metrics.tcl or any stage script.
#
# --- Mechanism, verified against the exact OpenSTA commit ORFS's
#     tools/OpenROAD submodule pins (509913b1398b36eda23caa1f1f380167465dceee,
#     github.com/The-OpenROAD-Project/OpenSTA), NOT assumed: ---
#
#   report_tns / report_wns / report_worst_slack do NOT take -corner.
#   (search/Search.tcl: `define_cmd_args "report_tns" {[-min] [-max]
#   [-digits digits]}`, same for report_wns/report_worst_slack; the SWIG
#   bindings they call -- total_negative_slack_cmd(min_max) and
#   worst_slack_cmd(min_max) in search/Search.i -- take only a MinMax,
#   no corner/scene.) Passing -corner to any of these three raises
#   "... is not a known keyword or flag." from parse_key_args -- it does
#   not silently ignore it. So per-corner TNS/WNS/worst-slack values are
#   read here via the lower-level, corner-scoped SWIG commands that
#   search/Search.i genuinely exposes and that OpenSTA's own test suite
#   uses this way (search/test/search_worst_slack_sta.tcl,
#   search/test/search_corner_skew.tcl):
#     sta::find_scene $corner                       -> Scene* for a
#                                                       CORNERS name
#                                                       (define_corners is
#                                                       a deprecated alias
#                                                       for define_scenes_cmd,
#                                                       so CORNERS entries
#                                                       are scene names)
#     sta::total_negative_slack_scene_cmd $scene max -> per-corner TNS
#     sta::worst_slack_scene $scene max               -> per-corner worst
#                                                        slack (max)
#   WNS is then derived exactly as OpenSTA's own report_wns proc derives
#   it from worst slack: wns = min(0.0, worst_slack).
#
#   report_clock_skew's -corner flag, by contrast, IS genuinely consumed
#   (contrary to how it might look from a shallow read of just its own
#   proc body): `parse_key_args` collects it into `keys(-corner)`, and
#   `report_clock_skew` passes that `keys` array by reference into
#   `parse_scenes_or_all keys` (tcl/CmdArgs.tcl), which explicitly reads
#   `keys(-corner)` as a "compabibility 05/29/2025" alias for `-scenes`
#   and resolves it via find_scenes. So `report_clock_skew -corner
#   $corner` really does scope the report to that one corner, the same
#   way report_power -corner does -- this was verified by reading
#   parse_scenes_or_all's body at the pinned commit, not assumed by
#   analogy with report_power.
#
#   Output format note: report_clock_skew does not print an aggregate
#   "worst skew" summary line -- it prints one "<value> setup skew" /
#   "<value> hold skew" line per clock (already the worst launch/capture
#   pair for that clock). flow/util/multicorner_dashboard.py parses all
#   such lines per corner and reports the largest-magnitude one.
#
# Opt-in: no-op unless REPORT_MULTICORNER_TIMING is set to a non-empty,
# non-"0" value -- matches the SKIP_REPORT_METRICS / DETAILED_METRICS /
# CTS_SNAPSHOTS opt-in flags already used in flow/scripts/.
#
# No-op when CORNERS has 0 or 1 entries -- nothing to break out per-corner.
#
# Wiring -- pick one:
#
#   1. HOOK_PATHS / CONFIG_HOOK_PATHS mechanism (see post_cts_timing_repair.tcl
#      for the pattern). Add to a design config.mk:
#        export REPORT_MULTICORNER_TIMING = 1
#        export POST_CTS_TCL = $(SCRIPTS_DIR)/report_multicorner_timing.tcl
#      Optionally set REPORT_MULTICORNER_STAGE / REPORT_MULTICORNER_WHEN to
#      label the output files for whichever hook point this is wired to
#      (defaults below are tuned for POST_CTS_TCL: stage "4", when
#      "cts final"). The same file can be wired to POST_GLOBAL_ROUTE_TCL
#      instead with REPORT_MULTICORNER_STAGE=5, REPORT_MULTICORNER_WHEN=
#      "global route".
#
#   2. Direct call from a Tcl console or another script, after sourcing:
#        source $::env(SCRIPTS_DIR)/report_multicorner_timing.tcl
#        report_multicorner_timing 6 "finish"
#      (this still requires REPORT_MULTICORNER_TIMING to be set to run).
#
# Output: one file per corner,
#   $::env(REPORTS_DIR)/${stage}_${when}_multicorner_${corner}.rpt
# mirroring the existing single-corner "<stage>_<when>.rpt" naming
# convention from report_metrics.tcl, so flow/util/multicorner_dashboard.py
# can glob and parse them per corner.

proc report_multicorner_timing_enabled {} {
  if { ![info exists ::env(REPORT_MULTICORNER_TIMING)] } {
    return false
  }
  if { $::env(REPORT_MULTICORNER_TIMING) eq "" || $::env(REPORT_MULTICORNER_TIMING) eq "0" } {
    return false
  }
  return true
}

proc report_multicorner_timing { stage when } {
  if { ![report_multicorner_timing_enabled] } {
    return
  }

  if { ![env_var_exists_and_non_empty CORNERS] } {
    return
  }

  if { [llength $::env(CORNERS)] < 2 } {
    return
  }

  set when_tag [string map {" " "_"} $when]

  foreach corner $::env(CORNERS) {
    set filename $::env(REPORTS_DIR)/${stage}_${when_tag}_multicorner_${corner}.rpt
    set fileId [open $filename w]
    close $fileId

    set scene [sta::find_scene $corner]
    if { $scene eq "NULL" } {
      puts "Warning: report_multicorner_timing: no scene found for corner '$corner', skipping"
      continue
    }

    set tns [sta::total_negative_slack_scene_cmd $scene max]
    set worst_slack [sta::worst_slack_scene $scene max]
    set wns $worst_slack
    if { $wns > 0.0 } {
      set wns 0.0
    }

    set fileId [open $filename a]
    puts $fileId "\n=========================================================================="
    puts $fileId "Corner: $corner"
    puts $fileId "$when report_tns (corner $corner)"
    puts $fileId "--------------------------------------------------------------------------"
    puts $fileId "tns max [sta::format_time $tns 4]"

    puts $fileId "\n=========================================================================="
    puts $fileId "$when report_wns (corner $corner)"
    puts $fileId "--------------------------------------------------------------------------"
    puts $fileId "wns max [sta::format_time $wns 4]"

    puts $fileId "\n=========================================================================="
    puts $fileId "$when report_worst_slack (corner $corner)"
    puts $fileId "--------------------------------------------------------------------------"
    puts $fileId "worst slack max [sta::format_time $worst_slack 4]"
    close $fileId

    if { [info exists ::env(REPORT_CLOCK_SKEW)] && $::env(REPORT_CLOCK_SKEW) } {
      set fileId [open $filename a]
      puts $fileId "\n=========================================================================="
      puts $fileId "$when report_clock_skew -corner $corner"
      puts $fileId "--------------------------------------------------------------------------"
      close $fileId
      report_clock_skew -corner $corner >> $filename
    }
  }
  unset corner
}

# When wired directly as a HOOK_PATHS entry (POST_CTS_TCL /
# POST_GLOBAL_ROUTE_TCL), this file is only `source`d -- there is no call
# site available to pass stage/when, so pull them from env with defaults
# tuned for the post-CTS hook point and invoke immediately.
if { [report_multicorner_timing_enabled] } {
  set report_multicorner_stage "4"
  set report_multicorner_when "cts final"
  if { [info exists ::env(REPORT_MULTICORNER_STAGE)] && $::env(REPORT_MULTICORNER_STAGE) ne "" } {
    set report_multicorner_stage $::env(REPORT_MULTICORNER_STAGE)
  }
  if { [info exists ::env(REPORT_MULTICORNER_WHEN)] && $::env(REPORT_MULTICORNER_WHEN) ne "" } {
    set report_multicorner_when $::env(REPORT_MULTICORNER_WHEN)
  }
  report_multicorner_timing $report_multicorner_stage $report_multicorner_when
  unset report_multicorner_stage report_multicorner_when
}
