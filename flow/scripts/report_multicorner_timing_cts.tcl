# report_multicorner_timing_cts.tcl
#
# POST_CTS hook: write the opt-in per-corner timing breakdown (see
# multicorner_timing_common.tcl for the full mechanism) labelled as the
# post-CTS stage.
#
# Shared implementation lives in multicorner_timing_common.tcl -- this
# file just supplies the post-CTS stage/when label. Complements
# report_multicorner_timing_grt.tcl, which supplies the post-GRT label;
# cts.tcl and global_route.tcl each run in a separate OpenROAD process,
# so there is no shared interpreter state between the two hook points to
# disambiguate -- hence the split into two hardcoded-identity files,
# following the same convention as post_cts_timing_repair.tcl /
# post_grt_timing_repair.tcl and their shared timing_repair_common.tcl.
#
# Usage -- add to a design config.mk:
#   export REPORT_MULTICORNER_TIMING = 1
#   export POST_CTS_TCL = $(SCRIPTS_DIR)/report_multicorner_timing_cts.tcl
#
# Or source manually inside an OpenROAD session after CTS has run:
#   source flow/scripts/report_multicorner_timing_cts.tcl

source [file join [file dirname [info script]] multicorner_timing_common.tcl]

report_multicorner_timing 4 "cts final"
