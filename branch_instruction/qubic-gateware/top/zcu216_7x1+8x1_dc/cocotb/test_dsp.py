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

RDRV_DAC = 0
QDRV_DAC = 1
COUPLER_DAC = 4

with open('../sim/gensrc/sim_memory_map.json') as f:
    memory_map = json.load(f)

@cocotb.test()
async def test_const_pulse_qdrv(dut):
    freq = 458.e6
    phase = 0
    tstart = 8
    pulse_length = 100
    amp = 0.9
    env_i = 0.2*np.ones(pulse_length)
    env_q = 0.1*np.ones(pulse_length)

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 16, 16)

    channel_configs = load_channel_configs('../sim/gensrc/channel_config.json')
    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['qubit_qdrv_0', 'qubit_rdrv_0', 'qubit_rdlo_0']}
    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.add_phase_reset()
    prog.add_pulse(freq, phase, amp, tstart, env_i + 1j*env_q, 'qubit_qdrv_0')
    prog.add_done_stb()
    prog_exe = prog.get_program_binaries()
    sim_prog = prog.get_sim_program()
    pulse_seq = [sim_prog[1]]

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)
    await dspunit.run_program(500)
    dacout_sim = st.generate_sim_dacout(pulse_seq, 16, interp_ratio=q0_chan_cfgs['qubit_qdrv_0'].elem_params['interp_ratio'])
    plt.plot(dspunit.dac_out[QDRV_DAC])
    plt.plot(dacout_sim)
    plt.show()
    # assert st.check_dacout_equal(dacout_sim, dspunit.dac_out[QDRV_DAC])

@cocotb.test()
async def test_const_pulse_rdrv(dut):
    freq = 347.e6
    phase = 0
    tstart = 8
    pulse_length = 100
    amp = 0.9
    env_i = 0.2*np.ones(pulse_length)
    env_q = 0.1*np.ones(pulse_length)

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 16, 16)

    channel_configs = load_channel_configs('../sim/gensrc/channel_config.json')
    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['qubit_qdrv_0', 'qubit_rdrv_0', 'qubit_rdlo_0']}
    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.add_phase_reset()
    prog.add_pulse(freq, phase, amp, tstart, env_i + 1j*env_q, 'qubit_rdrv_0')
    prog.add_done_stb()
    prog_exe = prog.get_program_binaries()
    sim_prog = prog.get_sim_program()
    pulse_seq = [sim_prog[1]]

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)
    await dspunit.run_program(500)
    dacout_sim = st.generate_sim_dacout(pulse_seq, 16, interp_ratio=q0_chan_cfgs['qubit_rdrv_0'].elem_params['interp_ratio'])
    plt.plot(dspunit.dac_out[RDRV_DAC])
    plt.plot(dacout_sim)
    plt.show()
    # assert st.check_dacout_equal(dacout_sim, dspunit.dac_out[RDRV_DAC])

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
                                         proc_grouping=[('qubit_qdrv_{qubit}', 'qubit_rdrv_{qubit}', 'qubit_rdlo_{qubit}'),
                                                        ('coupler_cdrv_{qubit}', 'coupler_dc_{qubit}')],
                                         qubit_grouping=None)
    binary = tc.run_assemble_stage(compiled_prog, channel_configs)
    
    cocotb.start_soon(dsp.generate_clock(dut))
    driver = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    await driver.flush_cmd_mem()
    await driver.load_asm_program(binary)

    await driver.run_program(500)

    plt.plot(driver.dac_out[COUPLER_DAC])
    plt.show()

@cocotb.test()
async def test_acc_sweep(dut):
    freq = 347.e6
    phase = np.pi/8
    tstart = 20
    pulse_length = 200
    amp = 0.9
    env_i = 0.2*np.ones(pulse_length)
    env_q = np.zeros(pulse_length)
    niters = 100

    channel_configs = load_channel_configs('../sim/gensrc/channel_config.json')
    qdrvelemcfg = RFSoCElementCfg(16, interp_ratio=4)
    rdrvelemcfg = RFSoCElementCfg(16, interp_ratio=16)
    rdloelemcfg = RFSoCElementCfg(4, interp_ratio=channel_configs['qubit_rdlo_0'].elem_params['interp_ratio'])

    adc_fullscale = 2**15-1

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    adc_signal = adc_fullscale*np.cos(2*np.pi*freq*(1.e-9*2*dsp.CLK_CYCLE/dspunit.adc_samples_per_clk)*np.arange(40000) + phase)
    adc_signal[-1] = 0

    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['qubit_qdrv_0', 'qubit_rdrv_0', 'qubit_rdlo_0']}
    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.add_phase_reset()
    prog.add_reg_write('n_iters', niters)
    prog.add_reg_write('i', 0)
    prog.add_reg_write('phase', 0, dtype='phase')
    prog.add_pulse(freq, 'phase', amp, tstart, env_i + 1j*env_q, 'qubit_rdlo_0', label='PULSE')
    prog.add_alu_cmd('inc_qclk', -(pulse_length + tstart), 'add')
    prog.add_alu_cmd('reg_alu', 1, 'add', 'i', 'i')
    prog.add_alu_cmd('reg_alu', 2*np.pi/niters, 'add', 'phase', 'phase')
    prog.add_alu_cmd('jump_cond', 'i', 'le', 'n_iters', jump_label='PULSE')
    prog.add_done_stb()
    prog_exe = prog.get_program_binaries()
    sim_prog = prog.get_sim_program()
    pulse_seq = [sim_prog[1]]

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)

    cocotb.start_soon(dspunit.generate_adc_signal(adc_signal, 0))
    await dspunit.run_program(10000)
    acc_buf = await dspunit.read_acc_buf(100)
    plt.plot(np.real(acc_buf), np.imag(acc_buf), '.')
    plt.show()
