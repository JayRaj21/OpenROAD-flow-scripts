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
  file delete -force $f
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
# Can a cell be resized up / down? Returns {can_up can_down} as 0/1 values.
# upsize_map / downsize_map are array-get lists (cell -> next cell), as
# returned by build_upsize_map / build_downsize_map. A cell that is excluded
# from resizing (flip-flops, latches, clock buffers, clock gates) can never be
# resized. Pure list logic, so it is unit-testable without an ODB database.
# -----------------------------------------------------------------------
proc eco_resize_flags { cell upsize_map downsize_map } {
  if { [is_excluded $cell] } {
    return [list 0 0]
  }
  array set up $upsize_map
  array set down $downsize_map
  return [list [info exists up($cell)] [info exists down($cell)]]
}

# -----------------------------------------------------------------------
# Informational: instances on the worst setup paths. Never applies
# anything.
#
# With annotate=1, each target also carries can_up / can_down flags saying
# whether eco_resize could size that instance's cell up or down, so a caller
# can skip cells that would only come back with "no up-size target". The
# size maps are built once per call and only when asked for, so the ordinary
# eco_run path pays nothing for this.
# -----------------------------------------------------------------------
proc eco_targets { count { annotate 0 } } {
  set targets {}
  set seen_pins {}
  set db [::ord::get_db]
  set block [[$db getChip] getBlock]

  set upsize_map {}
  set downsize_map {}
  if { $annotate } {
    set upsize_map [build_upsize_map]
    set downsize_map [build_downsize_map]
  }

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
          set target [dict create pin $pin_name inst $inst_name cell $cell_name]
          if { $annotate } {
            lassign [eco_resize_flags $cell_name $upsize_map $downsize_map] can_up can_down
            dict set target can_up $can_up
            dict set target can_down $can_down
          }
          lappend targets $target
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
    set extra ""
    if { [dict exists $t can_up] && [dict exists $t can_down] } {
      set up [expr { [dict get $t can_up] ? "true" : "false" }]
      set down [expr { [dict get $t can_down] ? "true" : "false" }]
      set extra ",\"can_up\":$up,\"can_down\":$down"
    }
    lappend items [format {{"pin":%s,"inst":%s,"cell":%s%s}} \
      [eco_json_str [dict get $t pin]] [eco_json_str [dict get $t inst]] \
      [eco_json_str [dict get $t cell]] $extra]
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

  # list_targets: report the instances on the worst setup paths, each marked
  # with whether it can be sized up / down, and stop. Nothing is changed and
  # no timing snapshot is taken, so this is one cheap load-and-report run.
  if { $fix_type eq "list_targets" } {
    set fix [dict create kind list_targets inst "" from "" to "" placement_warning ""]
    set list_ok [catch { set targets [eco_targets 5 1] } list_msg]
    if { $list_ok != 0 } {
      set targets {}
      set msg "eco_targets failed: $list_msg"
    } else {
      set status "listed"
    }
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

# =======================================================================
# Batch SEARCH: try several candidate resizes in ONE OpenROAD session, and
# report which one would pass. It never writes the database.
#
# eco_run loads the design once per attempt. eco_search_run loads it once for
# a whole list of candidates, applying each inside an ODB ECO journal and
# rolling it back with undoEco when it does not pass.
#
# Why it does not write: undoEco restores cell masters, positions,
# orientations, connectivity and timing exactly (live-tested on
# nangate45/gcd), but NOT every piece of database state. The resizer clears a
# resized cell's preferred pin access points, and that clearing is not in the
# journal, so after rollbacks those cells differ from a pristine load. An
# independent review found that a database written after several rollbacks
# differed from the one a single attempt writes, and reloaded with slightly
# different timing than the batch had measured. The guard below cannot see
# this, because in-session timing still matches the baseline.
#
# So the search only decides WHICH candidate passes. The caller re-applies
# that one change alone to a freshly loaded database (a normal eco_run), which
# is the path whose written result is byte-identical to a single attempt.
#
# Because a wrong undo would silently poison every later attempt in the
# search, it still re-measures after every rollback and STOPS if the numbers
# differ from the baseline. The search stops at the first candidate that
# passes eco_verdict.
# =======================================================================

# -----------------------------------------------------------------------
# JSON for the list of attempts. Each attempt is a dict with inst, outcome
# (accepted | rejected | skipped | error), reason, from, to, placement_warning,
# and optionally after (metrics dict) and delta (dict of wns/tns/hold).
# -----------------------------------------------------------------------
proc eco_json_attempts { attempts } {
  set items {}
  foreach a $attempts {
    set after "null"
    if { [dict exists $a after] } {
      set after [eco_json_metrics [dict get $a after]]
    }
    set delta "null"
    if { [dict exists $a delta] } {
      set d [dict get $a delta]
      set delta [format {{"wns":%s,"tns":%s,"worst_hold_slack":%s}} \
        [dict get $d wns] [dict get $d tns] [dict get $d worst_hold_slack]]
    }
    set head [format {"inst":%s,"outcome":%s,"reason":%s,"from":%s,"to":%s} \
      [eco_json_str [dict get $a inst]] [eco_json_str [dict get $a outcome]] \
      [eco_json_str [dict get $a reason]] [eco_json_str [dict get $a from]] \
      [eco_json_str [dict get $a to]]]
    set tail [format {"placement_warning":%s,"after":%s,"delta":%s} \
      [eco_json_str [dict get $a placement_warning]] $after $delta]
    lappend items "\{$head,$tail\}"
  }
  return "\[[join $items ,]\]"
}

proc eco_write_batch_result {
  json_out id status msg direction before attempts candidate
} {
  set body [join [list \
    [format {"id":%s,"status":%s,"msg":%s} \
      [eco_json_str $id] [eco_json_str $status] [eco_json_str $msg]] \
    [format {"direction":%s,"candidate":%s} \
      [eco_json_str $direction] [eco_json_str $candidate]] \
    [format {"before":%s} [eco_json_metrics $before]] \
    [format {"attempts":%s} [eco_json_attempts $attempts]]] ,]
  set fileId [open $json_out w]
  puts $fileId "\{$body\}"
  close $fileId
}

# -----------------------------------------------------------------------
# Difference between two metrics dicts as {wns tns worst_hold_slack}, with a
# tns delta of 0.0 when either side could not be measured (matching eco_run).
# -----------------------------------------------------------------------
proc eco_metric_delta { before after } {
  set d_wns [expr { [dict get $after wns] - [dict get $before wns] }]
  set d_hold [expr { [dict get $after worst_hold_slack] - [dict get $before worst_hold_slack] }]
  if { [dict get $before tns] eq "NA" || [dict get $after tns] eq "NA" } {
    set d_tns 0.0
  } else {
    set d_tns [expr { [dict get $after tns] - [dict get $before tns] }]
  }
  return [dict create wns $d_wns tns $d_tns worst_hold_slack $d_hold]
}

# -----------------------------------------------------------------------
# Orchestrator for the batch search. direction is "up" or "down"; candidates
# is a Tcl list of instance names. status is "found" (candidate names the
# instance whose change passed; nothing was written), "rejected" (none
# passed), or "error".
# -----------------------------------------------------------------------
proc eco_search_run {
  json_out id odb_file stage direction candidates
  tol_wns tol_tns tol_hold tol_wns_hold
} {
  set status "error"
  set msg ""
  set na [dict create wns NA tns NA worst_hold_slack NA setup_viol_count NA hold_viol_count NA]
  set before $na
  set attempts {}
  set candidate ""
  set fix_type [expr { $direction eq "up" ? "resize_up" : "resize_down" }]

  if { $direction ne "up" && $direction ne "down" } {
    eco_write_batch_result $json_out $id error "unknown direction: $direction" \
      $direction $before $attempts $candidate
    return
  }

  set load_ok [catch { set pflag [eco_load $odb_file $stage] } load_msg]
  if { $load_ok != 0 } {
    eco_write_batch_result $json_out $id error $load_msg $direction $before $attempts \
      $candidate
    return
  }

  set measure_ok [catch { set before [eco_measure ${id}_base before] } measure_msg]
  if { $measure_ok != 0 } {
    set before $na
    eco_write_batch_result $json_out $id error "eco_measure (baseline) failed: $measure_msg" \
      $direction $before $attempts $candidate
    return
  }

  set db [::ord::get_db]
  set block [[$db getChip] getBlock]
  set n 0
  set stopped_early 0

  foreach inst $candidates {
    incr n
    set attempt [dict create inst $inst outcome error reason "" from "" to "" \
      placement_warning ""]
    set accepted 0

    set odb_inst [$block findInst $inst]
    set has_inst [expr { $odb_inst ne "NULL" && $odb_inst ne "" }]
    set orig_cell ""
    if { $has_inst } {
      set orig_cell [[$odb_inst getMaster] getName]
    }

    odb::dbDatabase_beginEco $block
    set apply_ok [catch { set fix [eco_resize $inst $direction {} $pflag] } apply_msg]
    if { $apply_ok != 0 } {
      dict set attempt reason "exception while applying: $apply_msg"
    } elseif { [dict get $fix status] ne "ok" } {
      # The resize did not complete. If the cell's master is unchanged nothing
      # was applied (excluded cell, no size target, ...): that is a skip. If
      # the master changed, a change was applied and then a later step failed:
      # report it as an error, not a skip. Either way it is rolled back below.
      set now_cell $orig_cell
      if { $has_inst } {
        catch { set now_cell [[$odb_inst getMaster] getName] }
      }
      if { $now_cell eq $orig_cell } {
        dict set attempt outcome skipped
        dict set attempt reason [dict get $fix msg]
      } else {
        dict set attempt reason "[dict get $fix msg] (a change was applied, then rolled back)"
      }
    } else {
      dict set attempt from [dict get $fix from]
      dict set attempt to [dict get $fix to]
      if { [dict exists $fix placement_warning] } {
        dict set attempt placement_warning [dict get $fix placement_warning]
      }
      set after_ok [catch { set after [eco_measure ${id}_a$n after] } after_msg]
      if { $after_ok != 0 } {
        dict set attempt reason "eco_measure (after) failed: $after_msg"
      } else {
        set verdict [eco_verdict $fix_type $before $after $tol_wns $tol_tns $tol_hold \
          $tol_wns_hold]
        dict set attempt after $after
        dict set attempt delta [eco_metric_delta $before $after]
        dict set attempt reason [dict get $verdict reason]
        if { [dict get $verdict accepted] } {
          set accepted 1
        } else {
          dict set attempt outcome rejected
        }
      }
    }

    odb::dbDatabase_endEco $block

    if { $accepted } {
      # This candidate passed. Nothing is written: the caller re-applies it
      # alone to a freshly loaded database (see the header comment).
      dict set attempt outcome accepted
      set candidate $inst
      set status "found"
      lappend attempts $attempt
      break
    }

    # Not accepted: roll the change back, then prove the design is back to
    # the baseline before trying the next candidate.
    odb::dbDatabase_undoEco $block
    set refresh_ok [catch { estimate_parasitics $pflag } refresh_msg]
    set check_ok [catch { set check [eco_measure ${id}_u$n undo] } check_msg]
    lappend attempts $attempt
    if { $refresh_ok != 0 || $check_ok != 0 } {
      set why [expr { $refresh_ok != 0 ? $refresh_msg : $check_msg }]
      set msg "could not verify the rollback after $inst; search stopped. $why"
      set stopped_early 1
      break
    }
    if { $check ne $before } {
      set msg "rollback check failed after $inst: timing after undo does not match the\
 baseline; search stopped. baseline=$before now=$check"
      set stopped_early 1
      break
    }
  }

  if { $status ne "found" && !$stopped_early && $msg eq "" } {
    set status "rejected"
  }
  eco_write_batch_result $json_out $id $status $msg $direction $before $attempts $candidate
}
} ;# namespace trepair
