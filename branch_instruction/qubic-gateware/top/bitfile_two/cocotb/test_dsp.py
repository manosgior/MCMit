"""
This file contains basic tests for common sig gen types/channels. Includes:
    - qubit drive
    - readout drive
    - rdlo sweep
    - coupler (RF + DC) drive

Tests as included will simply plot the DAC (or ACC buffer) output of the simulated 
program; assertions are commented out since these depend on the delay being used
for the build being simulated. This test suite is intended as a starting point; it 
is expected that the user will add, remove, or modify tests for specific builds.
"""

import cocotb
from cocotb.triggers import Timer, RisingEdge
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import qubic.sim.cocotb_drivers as dsp
from qubic.rfsoc.hwconfig import DCOffsetCfg, RFSoCElementCfg, load_channel_configs, elemconfig_classfactory
import distproc.assembler as asm
import distproc.compiler as cm
import qubic.toolchain as tc
import qubic.sim.tools as st
import qubitconfig.qchip as qc
import distproc.hwconfig as hw
import json

COUPLER_DAC = 0

with open('../sim/gensrc/sim_memory_map.json') as f:
    memory_map = json.load(f)


@cocotb.test()
async def test_coupler_rf_dc(dut):
    channel_configs = load_channel_configs('../sim/gensrc/channel_config.json')
    qchip = qc.QChip('qubitcfg.json')
    prog = [{'name': 'pulse', 'freq':4428998888, 'amp': 0.5, 
             'twidth': 32.e-9, 'phase': 0,
             'env': 
                {'env_func': 'gaussian', 'paradict':{'sigmas': 3}}, 
             'dest': 'coupler_cdrv_0'},
            {'name': 'pulse', 'freq': None, 'phase': 0, 'twidth': 32.e-9,
             'amp': -0.5, 'env': None, 'dest': 'coupler_dc_0'},
            {'name': 'pulse', 'freq': None, 'phase': 0, 'twidth': 32.e-9,
             'amp': 0, 'env': None, 'dest': 'coupler_dc_0'}]
    compiled_prog = tc.run_compile_stage(prog, fpga_config=hw.FPGAConfig(), qchip=qchip, compiler_flags={'resolve_gates': False},
                                         proc_grouping=[('coupler_cdrv_{qubit}', 'coupler_dc_{qubit}')],
                                         qubit_grouping=None)
    binary = tc.run_assemble_stage(compiled_prog, channel_configs)
    
    cocotb.start_soon(dsp.generate_clock(dut))
    driver = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    await driver.flush_cmd_mem()
    await driver.load_asm_program(binary)

    await driver.run_program(500)

    plt.plot(driver.dac_out[COUPLER_DAC])
    plt.show()

