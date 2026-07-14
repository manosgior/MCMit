import cocotb
from cocotb.triggers import Timer, RisingEdge
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import qubic.sim.cocotb_drivers as dsp
from qubic.rfsoc.hwconfig import DCOffsetCfg, RFSoCElementCfg, load_channel_configs, elemconfig_classfactory
import distproc.assembler as asm
import distproc.compiler as cm
import qubic.sim.tools as st
import qubitconfig.qchip as qc
import distproc.hwconfig as hw
import json

RDRV_IND = 14

with open('../sim/gensrc/sim_memory_map.json') as f:
    memory_map = json.load(f)

@cocotb.test()
async def test_const_pulse(dut):
    freq = 100.e6
    phase = 0
    tstart = 8
    pulse_length = 100
    amp = 0.9
    env_i = 0.2*np.ones(pulse_length)
    env_q = 0.1*np.ones(pulse_length)
    channel_configs = load_channel_configs('channel_config.json')

    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo']}
    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q0_chan_cfgs.items()}

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 16, 16)

    channel_configs = load_channel_configs('channel_config.json')
    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo']}
    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.add_phase_reset()
    prog.add_pulse(freq, phase, amp, tstart, env_i + 1j*env_q, 'Q0.qdrv')
    prog.add_done_stb()
    prog_exe = prog.get_program_binaries()
    sim_prog = prog.get_sim_program()
    pulse_seq = [sim_prog[1]]

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)
    await dspunit.run_program(500)
    dacout_sim = st.generate_sim_dacout(pulse_seq, 16, interp_ratio=q0_chan_cfgs['Q0.qdrv'].elem_params['interp_ratio'])
    plt.plot(dspunit.dac_out[0])
    plt.plot(dacout_sim)
    plt.show()
    assert st.check_dacout_equal(dacout_sim, dspunit.dac_out[0])
