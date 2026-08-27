# timing_repair_common.tcl
#
# Shared implementation for the post-CTS and post-global-route timing repair
# hooks (post_cts_timing_repair.tcl, post_grt_timing_repair.tcl). Identifies
# instances on setup-critical paths and upsizes them to the next drive
# strength available in the loaded libraries.
#
# Only combinational cells following the TYPE_X<N> naming convention are
# swapped. Flip-flops, clock cells, and cells already at maximum drive are
# left untouched.
#
# Each caller sources this file and then invokes:
#   trepair::run <namespace-tag> <log-prefix> <parasitics-flag> ?path_count? ?max_swaps? ?max_iters?
#
# where <parasitics-flag> is the flag passed to estimate_parasitics
# (e.g. -placement or -global_routing).

namespace eval trepair {

# -----------------------------------------------------------------------
# Build a map: current_cell_name -> next_drive_cell_name
# Discovered dynamically from whatever libraries are loaded, so this
# works for any PDK that follows the _X<N> convention.
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
        # Sort by drive strength numerically, build consecutive pairs
        set sorted [lsort -integer -index 0 $by_base($base)]
        for {set i 0} {$i < [llength $sorted] - 1} {incr i} {
            set curr [lindex [lindex $sorted $i]       1]
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
# Flip-flops: changing drive alters hold/setup arcs non-trivially.
# Clock cells: CTS balanced the tree for a specific drive; don't disturb.
# -----------------------------------------------------------------------
proc is_excluded {cell_name} {
    foreach prefix {DFF SDFF DFFR DFFS DFFRS SDFFR SDFFS SDFFRS DLL DLH CLKBUF CLKGATE CLKGATETST} {
        if {[string match "${prefix}*" $cell_name]} { return 1 }
    }
    return 0
}

# -----------------------------------------------------------------------
# Collect upsize candidates from the N worst setup paths.
# Returns a list of {inst_name curr_cell next_cell}, deduped.
# Already-seen instances (array passed by name) are skipped.
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
proc apply_swaps {candidates log_prefix} {
    set db    [::ord::get_db]
    set block [[$db getChip] getBlock]
    set swap_count 0
    set skip_count 0

    foreach candidate $candidates {
        lassign $candidate inst_name curr next

        set inst [$block findInst $inst_name]
        if {$inst eq "NULL" || $inst eq ""} {
            puts "WARN \[$log_prefix\]   instance not found: $inst_name — skipping."
            incr skip_count
            continue
        }

        set new_master [find_master $next]
        if {$new_master eq ""} {
            puts "WARN \[$log_prefix\]   master not found: $next — skipping."
            incr skip_count
            continue
        }

        $inst swapMaster $new_master
        puts "INFO \[$log_prefix\]   swapped  $inst_name  $curr -> $next"
        incr swap_count
    }
    return [list $swap_count $skip_count]
}

# -----------------------------------------------------------------------
# Main procedure — iterative upsizing.
#
# Each iteration:
#   1. Find the N worst setup paths and collect upsize candidates.
#   2. Apply swaps (skipping instances already swapped in prior iterations).
#   3. Re-legalise placement and re-estimate parasitics.
#   4. Re-run STA; stop if timing closed or no improvement was made.
#
# log_prefix       : short tag used in "INFO [tag] ..." log lines
# parasitics_flag  : flag passed to estimate_parasitics (-placement or
#                     -global_routing)
# path_count       : paths to inspect per iteration
# max_swaps        : hard cap on total swaps across all iterations
# max_iters        : iteration limit (guards against non-converging loops)
# -----------------------------------------------------------------------
proc run {log_prefix parasitics_flag {path_count 10} {max_swaps 30} {max_iters 5}} {
    set wns_before [sta::worst_slack -max]

    if {$wns_before >= 0} {
        puts "INFO \[$log_prefix\] WNS [format %+.3f $wns_before] ns — no setup violations, skipping."
        return
    }
    puts "INFO \[$log_prefix\] WNS [format %+.3f $wns_before] ns — starting iterative cell upsizing."

    array set upsize [build_upsize_map]
    puts "INFO \[$log_prefix\] Upsize map: [array size upsize] candidate transitions loaded."

    # 'seen' tracks every instance swapped across all iterations so we never
    # upsize the same cell twice (it would already be at the next drive level).
    array set seen {}
    set total_swaps 0
    set wns_current $wns_before

    for {set iter 1} {$iter <= $max_iters} {incr iter} {
        set remaining [expr {$max_swaps - $total_swaps}]
        if {$remaining <= 0} {
            puts "INFO \[$log_prefix\] Iter $iter: swap cap ($max_swaps) reached — stopping."
            break
        }

        puts "INFO \[$log_prefix\] --- Iteration $iter (WNS [format %+.3f $wns_current] ns) ---"

        set candidates [collect_candidates upsize $path_count seen]

        if {[llength $candidates] == 0} {
            puts "INFO \[$log_prefix\] Iter $iter: no new candidates on critical paths — stopping."
            break
        }

        # Cap this iteration's swaps to what's left in the budget
        if {[llength $candidates] > $remaining} {
            set candidates [lrange $candidates 0 [expr {$remaining - 1}]]
        }

        lassign [apply_swaps $candidates $log_prefix] swap_count skip_count
        incr total_swaps $swap_count
        puts "INFO \[$log_prefix\] Iter $iter: $swap_count cell(s) upsized, $skip_count skipped."

        if {$swap_count == 0} {
            puts "INFO \[$log_prefix\] Iter $iter: nothing applied — stopping."
            break
        }

        # Re-legalise (widths changed) then update wire models
        set result [catch { detailed_placement } msg]
        if {$result != 0} {
            puts "WARN \[$log_prefix\] detailed_placement failed: $msg"
        }
        estimate_parasitics $parasitics_flag

        set wns_new [sta::worst_slack -max]
        set delta   [format %+.3f [expr {$wns_new - $wns_current}]]
        set wns_msg "WNS [format %+.3f $wns_current] -> [format %+.3f $wns_new] ns"
        puts "INFO \[$log_prefix\] Iter $iter: $wns_msg  (delta $delta ns)"

        set wns_current $wns_new

        if {$wns_current >= 0} {
            puts "INFO \[$log_prefix\] Timing closed after iteration $iter."
            break
        }
    }

    set total_delta [format %+.3f [expr {$wns_current - $wns_before}]]
    set done_msg "WNS [format %+.3f $wns_before] -> [format %+.3f $wns_current] ns"
    puts "INFO \[$log_prefix\] Done: $total_swaps swap(s), $done_msg  (total $total_delta ns)"
}

} ;# namespace trepair
