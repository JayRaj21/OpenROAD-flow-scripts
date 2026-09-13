# eco_repair.tcl
#
# On-demand, single-instance ECO repair for loop_agent.py's eco_fix tool.
# Loads an already-built stage database, applies ONE caller-specified fix
# (resize up/down or hold fix; buffer insert is implemented but disabled —
# see eco_insert_buffer), measures timing before and after in the same
# OpenROAD session, and writes the stage .odb back only if the targeted
# metric improved and nothing else regressed.
#
# Extends the ::trepair namespace defined in timing_repair_common.tcl and
# reuses its helpers (build_upsize_map, find_master, is_excluded).
#
# This file only defines procs — sourcing it does nothing. The caller
# invokes trepair::eco_run explicitly.

source [file join [file dirname [info script]] timing_repair_common.tcl]

namespace eval trepair {
# repair_hold_pin tuning — single-shot defaults, safe to tune later.
set ::eco_max_buffer_percent 50
set ::eco_max_passes 1

# -----------------------------------------------------------------------
# Load a stage odb and warm up STA, mirroring open.tcl's GUI_TIMING path.
# Returns the parasitics flag to reuse after an edit (-placement or
# -global_routing).
# -----------------------------------------------------------------------
proc eco_load { odb_file stage_tag } {
  if { ![file exists $odb_file] } {
    error "eco: odb not found: $odb_file"
  }
  set ::env(ODB_FILE) $odb_file
  set ::env(GUI_TIMING) 1
  source $::env(SCRIPTS_DIR)/open.tcl

  # Mirror open.tcl's own grt::have_routes guard: a 5_1_grt.odb can exist
  # without actual routes (e.g. an aborted/partial GRT run), in which case
  # -global_routing parasitics are not available and estimate_parasitics
  # calls made with that flag later would fail.
  if { $stage_tag eq "grt" } {
    set have_routes 0
    catch { set have_routes [grt::have_routes] }
    if { $have_routes } {
      return "-global_routing"
    }
    puts "WARN \[eco\] no global routing results available; falling back to -placement"
  }
  return "-placement"
}

# -----------------------------------------------------------------------
# Snapshot timing metrics. phase is "before" or "after"; id names the
# temp report file under OBJECTS_DIR/eco/.
# -----------------------------------------------------------------------
proc eco_measure { id phase } {
  set wns [sta::worst_slack -max]
  set worst_hold_slack [sta::worst_slack -min]
  set setup_viol_count [sta::endpoint_violation_count max]
  set hold_viol_count [sta::endpoint_violation_count min]

  set tns "NA"
  set f [file join $::env(OBJECTS_DIR) eco ${id}_${phase}_tns.txt]
  file mkdir [file dirname $f]
  set fileId [open $f w]
  close $fileId
  report_tns >> $f
  set fileId [open $f r]
  set contents [read $fileId]
  close $fileId
  if { [regexp {tns\s+max\s+([-\d.]+)} $contents -> tns_val] } {
    set tns $tns_val
  }

  return [dict create \
    wns $wns \
    tns $tns \
    worst_hold_slack $worst_hold_slack \
    setup_viol_count $setup_viol_count \
    hold_viol_count $hold_viol_count]
}

# -----------------------------------------------------------------------
# Informational: instances on the worst setup paths. Never applies
# anything.
# -----------------------------------------------------------------------
proc eco_targets { count } {
  set targets {}
  set seen_pins {}
  set db [::ord::get_db]
  set block [[$db getChip] getBlock]

  # Path has no prevPath method in this build (confirmed via live probe:
  # "Invalid method. Must be one of: ... pin edge tag pins start_path") — use
  # [$path pins] to get every pin along the path in one call instead of
  # manually walking backwards. -group_path_count (not the deprecated
  # -group_count) ensures multiple distinct violating endpoints are
  # returned, not just one path's worth of pins.
  set all_ends [find_timing_paths -path_delay max -sort_by_slack -group_path_count $count]
  set path_ends [lrange $all_ends 0 [expr { $count - 1 }]]

  foreach path_end $path_ends {
    set cur ""
    catch { set cur [$path_end path] }
    if { $cur eq "" || $cur eq "NULL" } { continue }

    set pins {}
    catch { set pins [$cur pins] }

    foreach pin $pins {
      set pin_name ""
      catch { set pin_name [get_full_name $pin] }
      if { $pin_name eq "" } { continue }
      if { [lsearch -exact $seen_pins $pin_name] >= 0 } { continue }
      lappend seen_pins $pin_name

      set slash [string last "/" $pin_name]
      if { $slash > 0 } {
        set inst_name [string range $pin_name 0 [expr { $slash - 1 }]]
        set odb_inst [$block findInst $inst_name]
        if { $odb_inst ne "NULL" && $odb_inst ne "" } {
          set cell_name [[$odb_inst getMaster] getName]
          lappend targets [dict create pin $pin_name inst $inst_name cell $cell_name]
        }
      }
    }
  }
  return $targets
}

# -----------------------------------------------------------------------
# Build a map: current_cell_name -> previous (weaker) drive_cell_name.
# Inverse of build_upsize_map.
# -----------------------------------------------------------------------
proc build_downsize_map { } {
  array set upsize [build_upsize_map]
  array set downsize {}
  foreach curr [array names upsize] {
    set downsize($upsize($curr)) $curr
  }
  return [array get downsize]
}

# -----------------------------------------------------------------------
# Compare two lists of terminal names for equality regardless of order.
# Pure list logic, factored out so it's unit-testable without an ODB
# database (see test_loop_agent.py's Tcl-level tests).
# -----------------------------------------------------------------------
proc terms_compatible { terms1 terms2 } {
  return [expr { [lsort $terms1] eq [lsort $terms2] }]
}

# -----------------------------------------------------------------------
# Signal (non power/ground) terminal names of a master, e.g. {A B ZN}.
# -----------------------------------------------------------------------
proc master_signal_terms { master } {
  set terms {}
  foreach mt [$master getMTerms] {
    set sig_type [$mt getSigType]
    if { $sig_type eq "POWER" || $sig_type eq "GROUND" } { continue }
    lappend terms [$mt getName]
  }
  return $terms
}

# -----------------------------------------------------------------------
# Resize one instance up or down. target_cell is an explicit master name,
# or {} to pick automatically from the up/down-size map.
#
# swapMaster silently no-ops (returns without error) when the new master's
# terminals don't match the current one's, e.g. an explicit `cell` that
# isn't actually a drive-strength/footprint-compatible variant. Guard
# against this two ways: (1) for the explicit-cell path, pre-check that the
# current and target masters have matching signal terminal names before
# attempting the swap, so we reject with a clear message before ever
# calling swapMaster; (2) for both the explicit-cell and auto-selected
# paths, re-read the instance's master right after swapMaster and confirm
# it actually changed to the intended cell -- a safety net in case the
# terminal-name heuristic misses an incompatibility swapMaster itself would
# still reject.
# -----------------------------------------------------------------------
proc eco_resize { inst_name direction target_cell parasitics_flag } {
  set db [::ord::get_db]
  set block [[$db getChip] getBlock]

  set inst [$block findInst $inst_name]
  if { $inst eq "NULL" || $inst eq "" } {
    return [dict create status error kind resize msg "instance not found: $inst_name"]
  }

  set curr_master [$inst getMaster]
  set curr_cell [$curr_master getName]
  if { [is_excluded $curr_cell] } {
    return [dict create status error kind resize msg "cell excluded from resize: $curr_cell"]
  }

  if { $target_cell ne "" } {
    set new_master [find_master $target_cell]
    if { $new_master eq "" } {
      return [dict create status error kind resize msg "master not found: $target_cell"]
    }
    if {
      ![terms_compatible [master_signal_terms $curr_master] \
        [master_signal_terms $new_master]]
    } {
      return [dict create status error kind resize msg \
        "incompatible swap: $target_cell has different signal terminals than \
$curr_cell on $inst_name; not a legal drive-strength/footprint-compatible variant"]
    }
    set new_cell $target_cell
  } else {
    if { $direction eq "up" } {
      array set sizemap [build_upsize_map]
    } else {
      array set sizemap [build_downsize_map]
    }
    if { ![info exists sizemap($curr_cell)] } {
      return [dict create status error kind resize \
        msg "no $direction-size target for: $curr_cell"]
    }
    set new_cell $sizemap($curr_cell)
    set new_master [find_master $new_cell]
    if { $new_master eq "" } {
      return [dict create status error kind resize msg "master not found: $new_cell"]
    }
  }

  $inst swapMaster $new_master

  set actual_cell [[$inst getMaster] getName]
  if { $actual_cell ne $new_cell } {
    return [dict create status error kind resize msg \
      "swapMaster silently failed: $inst_name is still $actual_cell after attempting \
swap to $new_cell (likely incompatible terminals/footprint)"]
  }

  set placement_warning ""
  set result [catch { detailed_placement } msg]
  if { $result != 0 } {
    puts "WARN \[eco\] detailed_placement failed: $msg"
    set placement_warning "detailed_placement failed after resize: $msg"
  }
  set result [catch { estimate_parasitics $parasitics_flag } msg]
  if { $result != 0 } {
    return [dict create status error kind resize msg "estimate_parasitics failed: $msg"]
  }

  return [dict create \
    status ok \
    msg "" \
    kind resize \
    inst $inst_name \
    from $curr_cell \
    to $new_cell \
    placement_warning $placement_warning]
}

# -----------------------------------------------------------------------
# Insert a buffer on a net via the public insert_buffer command.
#
# DISABLED as of 2026-09-12: live-tested against openroad/orfs:latest and
# confirmed to segfault the OpenROAD process itself (Signal 11 in
# rsz::Resizer::insertBufferAfterDriver) regardless of arguments tried
# (explicit -buffer_cell, -location, -load_pins). A segfault kills the
# whole `make run` subprocess, so no JSON result can ever be written for
# this path — eco_run's dispatch below returns a clean disabled-error
# without ever calling this proc. Kept defined for future use once this
# is fixed upstream or a different invocation form is found safe.
#
# Also note: MIN_BUF_CELL_AND_PORTS is a 3-token list (cell + input port +
# output port, e.g. "BUF_X1 A Z"), not a bare cell name — passing the whole
# string as -buffer_cell fails with STA-0116. Use [lindex ... 0] to extract
# just the cell name, and wrap the env read in a catch since the var may be
# undefined on some platforms. Both issues are fixed below in case this
# proc is re-enabled later.
# -----------------------------------------------------------------------
proc eco_insert_buffer { net_name buf_cell parasitics_flag } {
  set cell $buf_cell
  if { $cell eq "" } {
    set env_ok [catch { set cell [lindex $::env(MIN_BUF_CELL_AND_PORTS) 0] } env_msg]
    if { $env_ok != 0 } {
      return [dict create status error kind insert_buffer \
        msg "MIN_BUF_CELL_AND_PORTS not available: $env_msg"]
    }
  }

  set result [catch { insert_buffer -net $net_name -buffer_cell $cell } msg]
  if { $result != 0 } {
    return [dict create status error kind insert_buffer msg "insert_buffer failed: $msg"]
  }

  set placement_warning ""
  set result [catch { detailed_placement } msg]
  if { $result != 0 } {
    puts "WARN \[eco\] detailed_placement failed: $msg"
    set placement_warning "detailed_placement failed after insert_buffer: $msg"
  }
  set result [catch { estimate_parasitics $parasitics_flag } msg]
  if { $result != 0 } {
    return [dict create status error kind insert_buffer \
      msg "estimate_parasitics failed: $msg"]
  }

  return [dict create \
    status ok \
    msg "" \
    kind insert_buffer \
    inst "" \
    from $net_name \
    to $cell \
    placement_warning $placement_warning]
}

# -----------------------------------------------------------------------
# Repair hold at a single endpoint pin via rsz::repair_hold_pin.
# -----------------------------------------------------------------------
proc eco_fix_hold { end_pin parasitics_flag } {
  set pin_obj [get_pin $end_pin]
  if { $pin_obj eq "" || $pin_obj eq "NULL" } {
    return [dict create status error kind fix_hold msg "pin not found: $end_pin"]
  }

  set result [catch {
    rsz::repair_hold_pin $pin_obj $::env(SETUP_SLACK_MARGIN) $::env(HOLD_SLACK_MARGIN) \
      0 $::eco_max_buffer_percent $::eco_max_passes
  } msg]
  if { $result != 0 } {
    return [dict create status error kind fix_hold msg "repair_hold_pin failed: $msg"]
  }

  set placement_warning ""
  set result [catch { detailed_placement } msg]
  if { $result != 0 } {
    puts "WARN \[eco\] detailed_placement failed: $msg"
    set placement_warning "detailed_placement failed after fix_hold: $msg"
  }
  set result [catch { estimate_parasitics $parasitics_flag } msg]
  if { $result != 0 } {
    return [dict create status error kind fix_hold msg "estimate_parasitics failed: $msg"]
  }

  return [dict create \
    status ok \
    msg "" \
    kind fix_hold \
    inst "" \
    from $end_pin \
    to $end_pin \
    placement_warning $placement_warning]
}

# -----------------------------------------------------------------------
# Write the result dict out by hand as JSON. Instance/net/pin names may
# contain backslashes (e.g. yosys/ODB escaped names like
# ctrl.state.out\[0\]$_DFF_P_), and msg carries raw OpenSTA/OpenROAD error
# text which may contain arbitrary characters — both must be escaped.
# Order matters: backslash must be escaped BEFORE the quote, otherwise the
# backslash inserted by the quote-escaping step would itself be re-escaped.
# -----------------------------------------------------------------------
proc eco_json_str { s } {
  set s [string map { \\ \\\\ } $s]
  set s [string map { \" \\\" } $s]
  set s [string map [list "\n" {\n} "\r" {\r} "\t" {\t}] $s]
  return "\"$s\""
}

proc eco_json_num { v } {
  if { $v eq "NA" } {
    return {"NA"}
  }
  return $v
}

proc eco_json_metrics { m } {
  set tmpl {{"wns":%s,"tns":%s,"worst_hold_slack":%s,"setup_viol_count":%s,"hold_viol_count":%s}}
  return [format $tmpl \
    [eco_json_num [dict get $m wns]] [eco_json_num [dict get $m tns]] \
    [eco_json_num [dict get $m worst_hold_slack]] [eco_json_num [dict get $m setup_viol_count]] \
    [eco_json_num [dict get $m hold_viol_count]]]
}

proc eco_json_targets { targets } {
  set items {}
  foreach t $targets {
    lappend items [format {{"pin":%s,"inst":%s,"cell":%s}} \
      [eco_json_str [dict get $t pin]] [eco_json_str [dict get $t inst]] \
      [eco_json_str [dict get $t cell]]]
  }
  return "\[[join $items ,]\]"
}

# -----------------------------------------------------------------------
# Verdict: accept only if the targeted metric improved and nothing else
# regressed past tolerance.
#
# fix_hold gets its own, looser WNS-regression tolerance (tol_wns_hold):
# repairing a hold violation structurally trades a small, bounded amount of
# setup slack for hold closure (the hold buffer adds a few ps to tens of ps
# of extra delay on the data path, which can eat into an unrelated setup
# path's margin). The generic tol_wns (~1ps) treats that expected collateral
# cost as a regression and rejects nearly every real hold fix. tol_wns_hold
# is not unbounded, though — a runaway hold fix that blows past it is still
# caught and rejected.
#
# If wns/tns/worst_hold_slack could not be measured ("NA", e.g. report_tns
# parsing failed) there is no safe way to verdict: silently treating that as
# a zero delta both understates a real regression and can wrongly accept an
# ECO on unmeasured data. Refuse to accept in that case rather than guess.
# -----------------------------------------------------------------------
proc eco_verdict { fix_type before after tol_wns tol_tns tol_hold tol_wns_hold } {
  set target_metric [expr { $fix_type eq "fix_hold" ? "worst_hold_slack" : "wns" }]

  foreach m {wns tns worst_hold_slack} {
    if { [dict get $before $m] eq "NA" || [dict get $after $m] eq "NA" } {
      return [dict create accepted false target_metric $target_metric \
        reason "insufficient data: $m could not be measured"]
    }
  }

  set d_wns [expr { [dict get $after wns] - [dict get $before wns] }]
  set d_tns [expr { [dict get $after tns] - [dict get $before tns] }]
  set d_hold [expr { [dict get $after worst_hold_slack] - [dict get $before worst_hold_slack] }]
  set hold_viol_before [dict get $before hold_viol_count]
  set hold_viol_after [dict get $after hold_viol_count]
  set setup_viol_before [dict get $before setup_viol_count]
  set setup_viol_after [dict get $after setup_viol_count]

  set wns_tol [expr { $fix_type eq "fix_hold" ? $tol_wns_hold : $tol_wns }]

  # fix_hold structurally trades a small amount of setup slack for hold
  # closure (see tol_wns_hold above); that expected collateral WNS cost
  # (up to tol_wns_hold, e.g. tens of ps) is often just enough to flip one
  # or two already-near-zero endpoints from passing to violating, without
  # representing a real, independent setup regression. A zero-tolerance
  # setup_viol_count check would reject almost every real hold fix that
  # tol_wns_hold was specifically added to allow through. Give fix_hold a
  # small, bounded allowance here too -- 3 endpoints is enough to absorb
  # that expected collateral flip-over without letting a genuine setup
  # regression through unnoticed. resize_up/resize_down/default keep zero
  # tolerance: they have no structural reason to trade away setup
  # violations, and live-testing confirmed zero-tolerance correctly still
  # rejects real WNS-only regressions there.
  set setup_viol_tol [expr { $fix_type eq "fix_hold" ? 3 : 0 }]

  switch -- $fix_type {
    resize_up - insert_buffer {
      set improved [expr { $d_wns >= $tol_wns }]
    }
    resize_down {
      set improved [expr { $d_wns >= -$tol_wns && $d_tns >= -$tol_tns }]
    }
    fix_hold {
      set improved [expr { $d_hold >= $tol_hold }]
    }
    default {
      set improved 0
    }
  }

  set deltas [dict create wns $d_wns tns $d_tns worst_hold_slack $d_hold]

  set no_regression 1
  set reason ""
  if { $d_wns < -$wns_tol } {
    set no_regression 0
    set reason "wns regressed by $d_wns"
  } elseif { $d_tns < -$tol_tns } {
    set no_regression 0
    set reason "tns regressed by $d_tns"
  } elseif { $d_hold < -$tol_hold } {
    set no_regression 0
    set reason "worst_hold_slack regressed by $d_hold"
  } elseif { $hold_viol_after > $hold_viol_before } {
    set no_regression 0
    set reason "hold_viol_count increased from $hold_viol_before to $hold_viol_after"
  } elseif { $setup_viol_after > $setup_viol_before + $setup_viol_tol } {
    set no_regression 0
    set reason "setup_viol_count increased from $setup_viol_before to $setup_viol_after"
  }

  set accepted_bool [expr { $improved && $no_regression }]
  set accepted [expr { $accepted_bool ? "true" : "false" }]
  if { $accepted_bool } {
    if { $fix_type eq "resize_down" && [dict get $deltas $target_metric] == 0 } {
      set reason "no regression ($target_metric unchanged)"
    } else {
      set reason "improved $target_metric by [dict get $deltas $target_metric] and no regression"
    }
  } elseif { $reason eq "" } {
    set reason "no improvement in $target_metric"
  }

  return [dict create accepted $accepted target_metric $target_metric reason $reason]
}

# -----------------------------------------------------------------------
# Write the final result dict out as JSON. Factored out of eco_run so
# every early-return path (a thrown error from a fatal-if-uncaught call)
# writes the same clean JSON as the normal completion path.
# -----------------------------------------------------------------------
proc eco_write_result {
  json_out id status msg fix before after delta verdict targets
  odb_written
} {
  set line1 [format {"id":%s,"status":%s,"msg":%s} \
    [eco_json_str $id] [eco_json_str $status] [eco_json_str $msg]]
  set placement_warning ""
  if { [dict exists $fix placement_warning] } {
    set placement_warning [dict get $fix placement_warning]
  }
  set line2 [format {"fix":{"kind":%s,"inst":%s,"from":%s,"to":%s},"placement_warning":%s} \
    [eco_json_str [dict get $fix kind]] [eco_json_str [dict get $fix inst]] \
    [eco_json_str [dict get $fix from]] [eco_json_str [dict get $fix to]] \
    [eco_json_str $placement_warning]]
  set line3 [format {"before":%s,"after":%s} \
    [eco_json_metrics $before] [eco_json_metrics $after]]
  set line4 [format {"delta":{"wns":%s,"tns":%s,"worst_hold_slack":%s}} \
    [dict get $delta wns] [dict get $delta tns] [dict get $delta worst_hold_slack]]
  set line5 [format {"verdict":{"accepted":%s,"target_metric":%s,"reason":%s}} \
    [dict get $verdict accepted] [eco_json_str [dict get $verdict target_metric]] \
    [eco_json_str [dict get $verdict reason]]]
  set line6 [format {"targets":%s,"odb_written":%s} [eco_json_targets $targets] $odb_written]

  set body [join [list $line1 $line2 $line3 $line4 $line5 $line6] ,]
  set fileId [open $json_out w]
  puts $fileId "\{$body\}"
  close $fileId
}

# -----------------------------------------------------------------------
# Orchestrator. Called once by the generated per-eco script.
# -----------------------------------------------------------------------
proc eco_run {
  json_out id odb_file stage fix_type target opt_cell opt_count
  tol_wns tol_tns tol_hold tol_wns_hold
} {
  set status "error"
  set msg ""
  set fix [dict create kind "" inst "" from "" to ""]
  set before [dict create wns NA tns NA worst_hold_slack NA setup_viol_count NA hold_viol_count NA]
  set after $before
  set delta [dict create wns 0 tns 0 worst_hold_slack 0]
  set verdict [dict create accepted false target_metric "" reason ""]
  set targets {}
  set odb_written false

  set load_ok [catch { set pflag [eco_load $odb_file $stage] } load_msg]
  if { $load_ok != 0 } {
    set msg $load_msg
    eco_write_result $json_out $id $status $msg $fix $before $after $delta $verdict \
      $targets $odb_written
    return
  }

  set measure_ok [catch { set before [eco_measure $id before] } measure_msg]
  if { $measure_ok != 0 } {
    set msg "eco_measure (before) failed: $measure_msg"
    set fix [dict create kind unknown inst "" from "" to ""]
    set before [dict create wns NA tns NA worst_hold_slack NA setup_viol_count NA \
      hold_viol_count NA]
    set after $before
    eco_write_result $json_out $id $status $msg $fix $before $after $delta $verdict \
      $targets $odb_written
    return
  }
  set after $before

  set targets_ok [catch { set targets [eco_targets 5] } targets_msg]
  if { $targets_ok != 0 } {
    puts "WARN \[eco\] eco_targets failed: $targets_msg"
    set targets {}
  }

  switch -- $fix_type {
    resize_up {
      set fix [eco_resize $target up $opt_cell $pflag]
    }
    resize_down {
      set fix [eco_resize $target down $opt_cell $pflag]
    }
    insert_buffer {
      # DISABLED: insert_buffer segfaults the OpenROAD process itself on
      # every tested input in this build (verified 2026-09-12 — see
      # eco_insert_buffer's docstring). A segfault cannot be caught by
      # Tcl and kills the whole `make run` subprocess before any JSON can
      # be written, so eco_insert_buffer must never be called here.
      set fix [dict create status error kind insert_buffer inst "" from "" to "" \
        msg "insert_buffer is disabled in this OpenROAD build: it segfaults on \
all tested inputs (verified 2026-09-12); not safe to call."]
    }
    fix_hold {
      set fix [eco_fix_hold $target $pflag]
    }
    default {
      set fix [dict create status error kind unknown inst "" from "" to "" \
        msg "unknown fix_type: $fix_type"]
    }
  }

  if { ![dict exists $fix inst] } {
    set fix [dict merge {kind "" inst "" from "" to "" placement_warning ""} $fix]
  }
  if { ![dict exists $fix placement_warning] } {
    dict set fix placement_warning ""
  }

  if { [dict get $fix status] eq "ok" } {
    set after_ok [catch { set after [eco_measure $id after] } after_msg]
    if { $after_ok != 0 } {
      set after $before
      set status "error"
      set msg "eco_measure (after) failed: $after_msg"
    } else {
      set verdict [eco_verdict $fix_type $before $after $tol_wns $tol_tns $tol_hold $tol_wns_hold]
      set d_wns [expr { [dict get $after wns] - [dict get $before wns] }]
      set d_hold [expr {
        [dict get $after worst_hold_slack] - [dict get $before worst_hold_slack]
      }]
      if { [dict get $before tns] eq "NA" || [dict get $after tns] eq "NA" } {
        set d_tns 0.0
      } else {
        set d_tns [expr { [dict get $after tns] - [dict get $before tns] }]
      }
      set delta [dict create wns $d_wns tns $d_tns worst_hold_slack $d_hold]

      if { [dict get $verdict accepted] } {
        set write_ok [catch { write_db $odb_file } write_msg]
        if { $write_ok != 0 } {
          set status "error"
          set msg "write_db failed: $write_msg"
        } else {
          set odb_written true
          set status "applied"
        }
      } else {
        set status "rejected"
      }
    }
  } else {
    set after $before
    set status "error"
    set msg [dict get $fix msg]
  }

  eco_write_result $json_out $id $status $msg $fix $before $after $delta $verdict \
    $targets $odb_written
}
} ;# namespace trepair
