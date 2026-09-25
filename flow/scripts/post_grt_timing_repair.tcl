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
# Shared implementation lives in timing_repair_common.tcl (namespace
# ::trepair) — this file just supplies the post-GRT log prefix and
# parasitics mode (-global_routing instead of -placement).
#
# Usage — add to a design config or Makefile:
#   export POST_GLOBAL_ROUTE_TCL = $(SCRIPTS_DIR)/post_grt_timing_repair.tcl
#
# Or source manually inside an OpenROAD session after global_route has run:
#   source flow/scripts/post_grt_timing_repair.tcl

source [file join [file dirname [info script]] timing_repair_common.tcl]

# Run automatically when sourced as a POST_GLOBAL_ROUTE hook
trepair::run pgtr -global_routing
