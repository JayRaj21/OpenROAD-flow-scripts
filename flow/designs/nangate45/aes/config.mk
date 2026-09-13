export DESIGN_NICKNAME = aes
export DESIGN_NAME = aes_cipher_top
export PLATFORM    = nangate45

export VERILOG_FILES = $(sort $(wildcard $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/*.v))
export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/constraint.sdc

export FLOORPLAN_DEF = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/aes_ng45_fp.def

export PLACE_DENSITY_LB_ADDON = 0.20
export TNS_END_PERCENT        = 100
# workaround for high congestion in post-grt repair
export SKIP_INCREMENTAL_REPAIR = 1

# Triage-agent recommendations: pessimistic margin so CTS exposes violations
# that only appear at GRT under real wire RC; post-CTS hook sizes them while
# placement is still legalisable.
export SETUP_SLACK_MARGIN = 0.03
export POST_CTS_TCL = $(SCRIPTS_DIR)/post_cts_timing_repair.tcl

export SWAP_ARITH_OPERATORS = 1
export OPENROAD_HIERARCHICAL = 1
