# post_cts_timing_repair.tcl
#
# POST_CTS hook: identify instances on setup-critical paths and upsize them
# to the next drive strength available in the loaded libraries.
#
# After all swaps the placement is re-legalised (cell widths change) and
# parasitics are re-estimated so downstream timing reflects the new sizes.
#
# Shared implementation lives in timing_repair_common.tcl (namespace
# ::trepair) — this file just supplies the post-CTS log prefix and
# parasitics mode.
#
# Usage — add to a design config or Makefile:
#   export POST_CTS_TCL = $(SCRIPTS_DIR)/post_cts_timing_repair.tcl
#
# Or source manually inside an OpenROAD session:
#   source flow/scripts/post_cts_timing_repair.tcl

source [file join [file dirname [info script]] timing_repair_common.tcl]

# Run automatically when sourced as a POST_CTS hook
trepair::run pctr -placement
