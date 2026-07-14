export PYTHONPATH := ..:$(PYTHONPATH)

SIM ?= verilator
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES= $(shell python preinclude.py dsp.d -oneline)

EXTRA_ARGS += --trace --trace-structs --trace-fst
COMPILE_ARGS += -Wno-fatal

TOPLEVEL = dsp_sim_toplevel
MODULE = test_dsp

all: fst
fst: sim
	mv dump.fst $(TOPLEVEL).fst
	printf "\a\a\a"

include $(shell cocotb-config --makefiles)/Makefile.sim

