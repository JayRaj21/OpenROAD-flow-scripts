# ML congestion prediction hook — runs after detailed placement for every design.
# Prints per-layer congestion estimates and warns if predicted max exceeds 50%.
# Requires: python3 with torch + numpy on the host, and a trained checkpoint at
#   flow/ml/congestion/model/checkpoints/best.pt
export POST_DETAIL_PLACE_TCL ?= $(FLOW_HOME)/ml/congestion/integration/orfs_hook.tcl

# Enable LEC (Logical Equivalence Check) only if kepler-formal is installed.
# kepler-formal is primarily an OpenROAD/ORFS developer tool, not an end-user
# tool. End-users would typically run LEC transactionally at project completion,
# not in every CI run where it wastes CI time.
export LEC_CHECK ?= $(if $(wildcard $(KEPLER_FORMAL_EXE)),1,0)
