# ORFS hook: run ML congestion prediction after detailed placement.
#
# Enable by setting in your design's config.mk:
#   export POST_DETAIL_PLACE_TCL = $(FLOW_HOME)/ml/congestion/integration/orfs_hook.tcl
#
# Or set it on the make command line:
#   make ... POST_DETAIL_PLACE_TCL=/work/ml/congestion/integration/orfs_hook.tcl

set ml_root [file normalize [file join [file dirname [info script]] "../.."]]
set results_dir $::env(RESULTS_DIR)
set odb_path    [file join $results_dir "3_5_place_dp.odb"]
set script      [file join $ml_root "congestion/integration/predict_congestion.py"]

puts "\[ML\] Running congestion prediction..."

if {![file exists $odb_path]} {
    puts "\[ML\] WARNING: ODB not found at $odb_path — skipping"
    return
}

if {![file exists $script]} {
    puts "\[ML\] WARNING: predict_congestion.py not found — skipping"
    return
}

if {[catch {
    exec python3 $script \
        >@stdout 2>@stderr \
} err]} {
    puts "\[ML\] WARNING: congestion prediction failed: $err"
    puts "\[ML\] Continuing with flow..."
}
