export DESIGN_NAME = adder4
export PLATFORM    = nangate45

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NAME)/adder4.v
export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NAME)/constraint.sdc

export CORE_UTILIZATION  = 3
export PLACE_DENSITY_LB_ADDON = 0.10
export TNS_END_PERCENT   = 100
