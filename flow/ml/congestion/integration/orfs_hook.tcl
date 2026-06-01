# ORFS hook: run ML congestion prediction after detailed placement.
#
# To enable, add to your design's config.mk:
#   export POST_DETAIL_PLACE_TCL = $(FLOW_HOME)/../ml/congestion/integration/orfs_hook.tcl
#
# Or source it manually from within an OpenROAD Tcl session:
#   source ml/congestion/integration/orfs_hook.tcl

set ml_root [file normalize [file join [file dirname [info script]] "../.."]]
set results_dir $::env(RESULTS_DIR)

puts "\[ML\] Running congestion prediction hook..."
puts "\[ML\] ML root: $ml_root"
puts "\[ML\] Results dir: $results_dir"

set env(ML_ROOT) $ml_root
set env(RESULTS_DIR) $results_dir
set env(GRID_SIZE) 64
set env(CONGESTION_THRESHOLD) 0.5

set script [file join $ml_root "congestion/integration/predict_congestion.py"]

if {[catch {
    exec openroad -python $script >@stdout 2>@stderr
} err]} {
    puts "\[ML\] WARNING: congestion prediction failed: $err"
    puts "\[ML\] Continuing with flow..."
}
