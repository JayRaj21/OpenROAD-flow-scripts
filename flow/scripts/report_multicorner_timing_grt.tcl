# report_multicorner_timing_grt.tcl
#
# POST_GLOBAL_ROUTE hook: write the opt-in per-corner timing breakdown
# (see multicorner_timing_common.tcl for the full mechanism) labelled as
# the post-global-route stage.
#
# Shared implementation lives in multicorner_timing_common.tcl -- this
# file just supplies the post-GRT stage/when label. Complements
# report_multicorner_timing_cts.tcl, which supplies the post-CTS label;
# cts.tcl and global_route.tcl each run in a separate OpenROAD process,
# so there is no shared interpreter state between the two hook points to
# disambiguate -- hence the split into two hardcoded-identity files,
# following the same convention as post_cts_timing_repair.tcl /
# post_grt_timing_repair.tcl and their shared timing_repair_common.tcl.
#
# Usage -- add to a design config.mk:
#   export REPORT_MULTICORNER_TIMING = 1
#   export POST_GLOBAL_ROUTE_TCL = $(SCRIPTS_DIR)/report_multicorner_timing_grt.tcl
#
# Or source manually inside an OpenROAD session after global_route has run:
#   source flow/scripts/report_multicorner_timing_grt.tcl

source [file join [file dirname [info script]] multicorner_timing_common.tcl]

report_multicorner_timing 5 "global route"
