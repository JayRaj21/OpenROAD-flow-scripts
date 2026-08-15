# post_grt_timing_repair.tcl
#
# POST_GLOBAL_ROUTE hook: identify instances on setup-critical paths and
# upsize them to the next drive strength available in the loaded libraries.
#
# Complements post_cts_timing_repair.tcl.  At the post-CTS stage, parasitics
# are estimated from placement; violations that only appear under real wire
# geometry (like aes) are not yet visible.  By the time global routing has run,
# actual route topology is known, so this hook catches those late-appearing
# violations before detail routing locks in the geometry.
#
# Algorithm is identical to the post-CTS hook.  The only difference is that
# estimate_parasitics uses -global_routing (GRT topology) instead of
# -placement (idealised RC estimates), which is accurate at this stage.
#
# Usage — add to a design config or Makefile:
#   export POST_GLOBAL_ROUTE_TCL = $(SCRIPTS_DIR)/post_grt_timing_repair.tcl
#
# Or source manually inside an OpenROAD session after global_route has run:
#   source flow/scripts/post_grt_timing_repair.tcl

namespace eval pgtr {

# -----------------------------------------------------------------------
# Build a map: current_cell_name -> next_drive_cell_name
# Discovered dynamically from whatever libraries are loaded.
# -----------------------------------------------------------------------
proc build_upsize_map {} {
    array set by_base {}
    set db [::ord::get_db]

    foreach lib [$db getLibs] {
        foreach master [$lib getMasters] {
            set name [$master getName]
            if {[regexp {^(.+_X)(\d+)$} $name -> base drive]} {
                lappend by_base($base) [list [expr {int($drive)}] $name]
            }
        }
    }

    array set upsize {}
    foreach base [array names by_base] {
        set sorted [lsort -integer -index 0 $by_base($base)]
        for {set i 0} {$i < [llength $sorted] - 1} {incr i} {
            set curr [lindex [lindex $sorted $i]           1]
            set next [lindex [lindex $sorted [expr {$i+1}]] 1]
            set upsize($curr) $next
        }
    }

    return [array get upsize]
}

# -----------------------------------------------------------------------
# Search all loaded libs for a master by name.
# -----------------------------------------------------------------------
proc find_master {name} {
    set db [::ord::get_db]
    foreach lib [$db getLibs] {
        set m [$lib findMaster $name]
        if {$m ne "NULL" && $m ne ""} { return $m }
    }
    return ""
}

# -----------------------------------------------------------------------
# Cell types excluded from upsizing.
# -----------------------------------------------------------------------
proc is_excluded {cell_name} {
    foreach prefix {DFF SDFF DFFR DFFS DFFRS SDFFR SDFFS SDFFRS DLL DLH CLKBUF CLKGATE CLKGATETST} {
        if {[string match "${prefix}*" $cell_name]} { return 1 }
    }
    return 0
}

# -----------------------------------------------------------------------
# Collect upsize candidates from the N worst setup paths.
# seen_arr: array (by name) of already-swapped instances — skipped.
# -----------------------------------------------------------------------
proc collect_candidates {upsize_arr path_count seen_arr} {
    upvar $upsize_arr upsize
    upvar $seen_arr   seen

    set candidates {}
    set db    [::ord::get_db]
    set block [[$db getChip] getBlock]

    set all_ends  [find_timing_paths -path_delay max -sort_by_slack]
    set path_ends [lrange $all_ends 0 [expr {$path_count - 1}]]

    foreach path_end $path_ends {
        set cur ""
        catch { set cur [$path_end path] }
        for {set depth 0} {$depth < 200} {incr depth} {
            if {$cur eq "" || $cur eq "NULL"} { break }

            set pin_name ""
            catch { set pin_name [get_full_name [$cur pin]] }

            if {$pin_name ne ""} {
                set slash [string last "/" $pin_name]
                if {$slash > 0} {
                    set inst_name [string range $pin_name 0 [expr {$slash - 1}]]
                    if {![info exists seen($inst_name)]} {
                        set odb_inst [$block findInst $inst_name]
                        if {$odb_inst ne "NULL" && $odb_inst ne ""} {
                            set cell_name [[$odb_inst getMaster] getName]
                            if {![is_excluded $cell_name] && [info exists upsize($cell_name)]} {
                                set seen($inst_name) 1
                                lappend candidates [list $inst_name $cell_name $upsize($cell_name)]
                            }
                        }
                    }
                }
            }

            set prev ""
            catch { set prev [$cur prevPath] }
            if {$prev eq "" || $prev eq "NULL"} { break }
            set cur $prev
        }
    }
    return $candidates
}

# -----------------------------------------------------------------------
# Apply a list of {inst_name curr next} swaps via ODB.
# Returns {swap_count skip_count}.
# -----------------------------------------------------------------------
proc apply_swaps {candidates} {
    set db    [::ord::get_db]
    set block [[$db getChip] getBlock]
    set swap_count 0
    set skip_count 0

    foreach candidate $candidates {
        lassign $candidate inst_name curr next

        set inst [$block findInst $inst_name]
        if {$inst eq "NULL" || $inst eq ""} {
            puts "WARN \[pgtr\]   instance not found: $inst_name — skipping."
            incr skip_count
            continue
        }

        set new_master [find_master $next]
        if {$new_master eq ""} {
            puts "WARN \[pgtr\]   master not found: $next — skipping."
            incr skip_count
            continue
        }

        $inst swapMaster $new_master
        puts "INFO \[pgtr\]   swapped  $inst_name  $curr -> $next"
        incr swap_count
    }
    return [list $swap_count $skip_count]
}

# -----------------------------------------------------------------------
# Main procedure — iterative upsizing using GRT parasitics.
#
# path_count  : paths to inspect per iteration
# max_swaps   : hard cap on total swaps across all iterations
# max_iters   : iteration limit
# -----------------------------------------------------------------------
proc run {{path_count 10} {max_swaps 30} {max_iters 5}} {
    set wns_before [sta::worst_slack -max]

    if {$wns_before >= 0} {
        puts "INFO \[pgtr\] WNS [format %+.3f $wns_before] ns — no setup violations, skipping."
        return
    }
    puts "INFO \[pgtr\] WNS [format %+.3f $wns_before] ns — starting iterative cell upsizing (post-GRT)."

    array set upsize [build_upsize_map]
    puts "INFO \[pgtr\] Upsize map: [array size upsize] candidate transitions loaded."

    array set seen {}
    set total_swaps 0
    set wns_current $wns_before

    for {set iter 1} {$iter <= $max_iters} {incr iter} {
        set remaining [expr {$max_swaps - $total_swaps}]
        if {$remaining <= 0} {
            puts "INFO \[pgtr\] Iter $iter: swap cap ($max_swaps) reached — stopping."
            break
        }

        puts "INFO \[pgtr\] --- Iteration $iter (WNS [format %+.3f $wns_current] ns) ---"

        set candidates [collect_candidates upsize $path_count seen]

        if {[llength $candidates] == 0} {
            puts "INFO \[pgtr\] Iter $iter: no new swappable instances on critical paths — stopping."
            break
        }

        if {[llength $candidates] > $remaining} {
            set candidates [lrange $candidates 0 [expr {$remaining - 1}]]
        }

        lassign [apply_swaps $candidates] swap_count skip_count
        incr total_swaps $swap_count
        puts "INFO \[pgtr\] Iter $iter: $swap_count cell(s) upsized, $skip_count skipped."

        if {$swap_count == 0} {
            puts "INFO \[pgtr\] Iter $iter: nothing applied — stopping."
            break
        }

        # Re-legalise (widths changed) then re-estimate with GRT topology
        set result [catch { detailed_placement } msg]
        if {$result != 0} {
            puts "WARN \[pgtr\] detailed_placement failed: $msg"
        }
        estimate_parasitics -global_routing

        set wns_new [sta::worst_slack -max]
        set delta   [format %+.3f [expr {$wns_new - $wns_current}]]
        puts "INFO \[pgtr\] Iter $iter: WNS [format %+.3f $wns_current] -> [format %+.3f $wns_new] ns  (delta $delta ns)"

        set wns_current $wns_new

        if {$wns_current >= 0} {
            puts "INFO \[pgtr\] Timing closed after iteration $iter."
            break
        }
    }

    set total_delta [format %+.3f [expr {$wns_current - $wns_before}]]
    puts "INFO \[pgtr\] Done: $total_swaps total swap(s), WNS [format %+.3f $wns_before] -> [format %+.3f $wns_current] ns  (total $total_delta ns)"
}

} ;# namespace pgtr

# Run automatically when sourced as a POST_GLOBAL_ROUTE hook
pgtr::run
