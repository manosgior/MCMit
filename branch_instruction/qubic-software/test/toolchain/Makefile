#export PYTHONPATH := ..:$(PYTHONPATH)
export PYTHONPATH := .:$(COCOTB_BUILD_PATH)/..:$(PYTHONPATH);
#cd $(COCOTB_BUILD_PATH);

SIM ?= verilator
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES= $(shell python $(COCOTB_BUILD_PATH)/preinclude.py $(COCOTB_BUILD_PATH)/dsp.d -oneline --abspath)

EXTRA_ARGS += --trace --trace-structs --trace-fst
COMPILE_ARGS += -Wno-fatal

TOPLEVEL = dsp_sim_toplevel
MODULE = test_pulse_seq


all: fst
fst: sim
	mv dump.fst $(TOPLEVEL).fst
	printf "\a\a\a"

include $(shell cocotb-config --makefiles)/Makefile.sim

