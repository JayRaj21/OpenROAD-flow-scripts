# report_multicorner_timing.tcl
#
# Additive, opt-in per-corner timing breakdown.
#
# report_metrics.tcl already loops $::env(CORNERS) for report_power (see
# its "report_power" section), but report_tns / report_wns /
# report_worst_slack / report_clock_skew are called without -corner, so
# only the merged worst-case view across corners is ever written to the
# stage .rpt file. OpenSTA's report_tns/report_wns/report_worst_slack/
# report_clock_skew all accept the same "-corner <name>" flag report_power
# already uses here, so this script fills the per-corner gap without
# touching report_metrics.tcl or any stage script.
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

    set fileId [open $filename a]
    puts $fileId "\n=========================================================================="
    puts $fileId "Corner: $corner"
    puts $fileId "$when report_tns -corner $corner"
    puts $fileId "--------------------------------------------------------------------------"
    close $fileId
    report_tns -corner $corner >> $filename

    set fileId [open $filename a]
    puts $fileId "\n=========================================================================="
    puts $fileId "$when report_wns -corner $corner"
    puts $fileId "--------------------------------------------------------------------------"
    close $fileId
    report_wns -corner $corner >> $filename

    set fileId [open $filename a]
    puts $fileId "\n=========================================================================="
    puts $fileId "$when report_worst_slack -corner $corner"
    puts $fileId "--------------------------------------------------------------------------"
    close $fileId
    report_worst_slack -corner $corner >> $filename

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
