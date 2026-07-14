import cocotb
from cocotb.triggers import Timer, RisingEdge
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import qubic.sim.cocotb_drivers as dsp
from qubic.rfsoc.hwconfig import DCOffsetCfg, RFSoCElementCfg, RFMixElementCfg, load_channel_configs, elemconfig_classfactory
import distproc.assembler as asm
import distproc.compiler as cm
import qubic.sim.tools as st
import qubitconfig.qchip as qc
import distproc.hwconfig as hw
import json

INTERP_RATIO = 4
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
    dacout_sim = st.generate_sim_dacout(pulse_seq, 16, interp_ratio=q0_chan_cfgs['Q0.qdrv'].elem_params['interp_ratio'], extra_delay=2)
    plt.plot(dspunit.dac_out[0])
    plt.plot(dacout_sim)
    plt.show()
    assert st.check_dacout_equal(dacout_sim, dspunit.dac_out[0])

@cocotb.test()
async def test_ramp_pulse(dut):
    """
    Using a freq that is harmonic of FPGA clock period, so
    cordic phase acc doesn't matter
    """
    freq = 2500.e6
    phase = 0
    tstart = 10
    pulse_length = 1000
    amp = 0.9
    env_i = np.arange(pulse_length)/pulse_length
    env_q = np.zeros(pulse_length)/pulse_length

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
    dacout_sim = st.generate_sim_dacout(pulse_seq, 16, interp_ratio=INTERP_RATIO, extra_delay=2)
    plt.plot(dspunit.dac_out[0])
    plt.plot(dacout_sim)
    plt.show()
    assert st.check_dacout_equal(dacout_sim, dspunit.dac_out[0])

@cocotb.test()
async def test_consecutive_id_ramp_pulse(dut):
    elemcfg = RFSoCElementCfg()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 16, 16)

    channel_configs = load_channel_configs('channel_config.json')
    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo']}
    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    channel_configs = load_channel_configs('channel_config.json')
    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo']}
    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.add_phase_reset()

    pulse_length = 92
    env_i = np.arange(pulse_length)**2/pulse_length**2
    env_q = 0.3*np.arange(pulse_length)/pulse_length
    prog.add_pulse(freq=0.85e9, phase=0, amp=0.9, start_time=50, env=env_i + 1j*env_q, dest='Q0.qdrv')

    pulse_length = 92
    env_i = np.arange(pulse_length)**2/pulse_length**2
    env_q = 0.3*np.arange(pulse_length)/pulse_length
    prog.add_pulse(freq=0.85e9, phase=0, amp=0.9, start_time=100, env=env_i + 1j*env_q, dest='Q0.qdrv')

    prog.add_done_stb()
    prog_exe = prog.get_program_binaries()
    sim_prog = prog.get_sim_program()
    pulse_seq = sim_prog[1:3]

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)

    await dspunit.run_program(500)
    dacout_sim = st.generate_sim_dacout(pulse_seq, 16, interp_ratio=INTERP_RATIO, extra_delay=2)
    #plt.plot(dspunit.dac_out[0])
    #plt.plot(dacout_sim)
    #plt.show()
    assert st.check_dacout_equal(dacout_sim, dspunit.dac_out[0])

@cocotb.test()
async def test_consecutive_ramp_pulse(dut):
    elemcfg = RFSoCElementCfg()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 16, 16)

    channel_configs = load_channel_configs('channel_config.json')
    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo']}
    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.add_phase_reset()

    pulse_length = 100
    env_i = np.arange(pulse_length)/pulse_length
    env_q = 0.5*np.arange(pulse_length)/pulse_length
    prog.add_pulse(freq=1.5e9, phase=0, amp=0.9, start_time=50, env=env_i + 1j*env_q, dest='Q0.qdrv')

    pulse_length = 92
    env_i = np.arange(pulse_length)**2/pulse_length**2
    env_q = 0.3*np.arange(pulse_length)/pulse_length
    prog.add_pulse(freq=0.85e9, phase=np.pi/4, amp=0.8, start_time=110, env=env_i + 1j*env_q, dest='Q0.qdrv')

    prog.add_done_stb()
    prog_exe = prog.get_program_binaries()
    sim_prog = prog.get_sim_program()
    pulse_seq = sim_prog[1:3]
    #pulse_seq = [sim_prog[1]]

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)

    await dspunit.run_program(500)
    dacout_sim = st.generate_sim_dacout(pulse_seq, 16, interp_ratio=INTERP_RATIO, extra_delay=2)
    #plt.plot(dspunit.dac_out[0])
    #plt.plot(dacout_sim)
    #plt.show()
    assert st.check_dacout_equal(dacout_sim, dspunit.dac_out[0])

@cocotb.test()
async def test_consecutive_immediate_ramp_pulse(dut):
    elemcfg = RFSoCElementCfg()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 16, 16)

    channel_configs = load_channel_configs('channel_config.json')
    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo']}
    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.add_phase_reset()

    pulse_length = 100
    env_i = np.arange(pulse_length)/pulse_length
    env_q = 0.5*np.arange(pulse_length)/pulse_length
    prog.add_pulse(freq=1.5e9, phase=0, amp=0.9, start_time=50, env=env_i + 1j*env_q, dest='Q0.qdrv')

    pulse_length = 92
    env_i = np.arange(pulse_length)**2/pulse_length**2
    env_q = 0.5*np.arange(pulse_length)/pulse_length
    prog.add_pulse(freq=0.27e9, phase=np.pi/4, amp=0.4, start_time=55, env=env_i + 1j*env_q, dest='Q0.qdrv')

    prog.add_done_stb()
    prog_exe = prog.get_program_binaries()
    sim_prog = prog.get_sim_program()
    pulse_seq = sim_prog[1:3]
    #pulse_seq = [sim_prog[1]]

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)

    await dspunit.run_program(500)
    dacout_sim = st.generate_sim_dacout(pulse_seq, 16, interp_ratio=INTERP_RATIO, extra_delay=2)
    # TODO: fix consecutive pulse issue
    plt.plot(dspunit.dac_out[0])
    plt.plot(dacout_sim)
    plt.show()
    assert st.check_dacout_equal(dacout_sim, dspunit.dac_out[0])

@cocotb.test()
async def test_qdrv_initphase_ramp(dut):
    freq = 347.e6
    phase = np.pi/8
    tstart = 10
    pulse_length = 1000
    amp = 0.9
    env_i = np.arange(pulse_length)/pulse_length
    env_q = np.zeros(pulse_length)/pulse_length
    elemcfg = RFSoCElementCfg()

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
    dacout_sim = st.generate_sim_dacout(pulse_seq, 16, interp_ratio=INTERP_RATIO, extra_delay=2)
    #plt.plot(dspunit.dac_out[0])
    #plt.plot(dacout_sim)
    #plt.show()
    assert st.check_dacout_equal(dacout_sim, dspunit.dac_out[0])

@cocotb.test()
async def test_rdrv_pulse(dut):
    freq = 347.e6
    phase = np.pi/8
    tstart = 10
    pulse_length = 1000
    amp = 0.9
    env_i = np.arange(pulse_length)/pulse_length
    env_q = np.zeros(pulse_length)/pulse_length
    elemcfg = RFSoCElementCfg(interp_ratio=16)

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 16, 16)

    channel_configs = load_channel_configs('channel_config.json')
    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo']}
    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.add_phase_reset()
    prog.add_pulse(freq, phase, amp, tstart, env_i + 1j*env_q, 'Q0.rdrv')
    prog.add_done_stb()
    prog_exe = prog.get_program_binaries()
    sim_prog = prog.get_sim_program()
    pulse_seq = [sim_prog[1]]

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)

    await dspunit.run_program(5000)
    dacout_sim = st.generate_sim_dacout(pulse_seq, 16, extra_delay=3, interp_ratio=16)
    plt.plot(dspunit.dac_out[14])
    plt.plot(dacout_sim)
    plt.show()
    assert st.check_dacout_equal(dacout_sim, dspunit.dac_out[14])


#@cocotb.test()
#async def test_adc_lb(dut):
#    freq = 347.e6
#    phase = np.pi/8
#    tstart = 10
#    pulse_length = 100
#    amp = 0.9
#    env_i = 0.2*np.ones(pulse_length)
#    env_q = np.zeros(pulse_length)
#
#    dacelemcfg = RFSoCElementCfg(16)
#    adcelemcfg = RFSoCElementCfg(4)
#
#    adc_fullscale = 2**15-1
#
#    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
#    adc_signal = adc_fullscale*np.cos(2*np.pi*freq*(1.e-9*2*dsp.CLK_CYCLE/dspunit.adc_samples_per_clk)*np.arange(pulse_length) + phase)
#    adc_signal[-1] = 0
#
#    channel_configs = load_channel_configs('channel_config.json')
#    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo']}
#    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q0_chan_cfgs.items()}
#    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
#    prog.add_phase_reset()
#    prog.add_pulse(freq, phase, amp, tstart, env_i + 1j*env_q, 'Q0.rdlo')
#    prog.add_done_stb()
#    prog_exe = prog.get_program_binaries()
#    sim_prog = prog.get_sim_program()
#    pulse_seq = [sim_prog[1]]
#
#    cocotb.start_soon(dsp.generate_clock(dut))
#    await dspunit.load_asm_program(prog_exe)
#
#    cocotb.start_soon(dspunit.generate_adc_signal(adc_signal, 0))
#    await dspunit.run_program(1000)
#
#    acq_buf = await dspunit.read_acq_buf(pulse_length+tstart+1000, 0)
#    
#    plt.plot(acq_buf)
#    plt.plot(adc_signal)
#    plt.show()

@cocotb.test()
async def test_accbuf(dut):
    freq = 347.e6
    phase = np.pi/8
    tstart = 10
    pulse_length = 100
    amp = 0.9
    env_i = 0.2*np.ones(pulse_length)
    env_q = np.zeros(pulse_length)

    qdrvelemcfg = RFSoCElementCfg(16, interp_ratio=4)
    rdrvelemcfg = RFSoCElementCfg(16, interp_ratio=16)
    rdloelemcfg = RFSoCElementCfg(4, interp_ratio=4)

    adc_fullscale = 2**15-1

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    adc_signal = adc_fullscale*np.cos(2*np.pi*freq*(1.e-9*2*dsp.CLK_CYCLE/dspunit.adc_samples_per_clk)*np.arange(pulse_length) + phase)
    adc_signal[-1] = 0

    channel_configs = load_channel_configs('channel_config.json')
    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo']}
    elem_configs = {k: elemconfig_classfactory(v.elem_type)(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.add_phase_reset()
    #prog.add_pulse(freq, phase, amp, tstart, env_i + 1j*env_q, 'Q0.rdrv')
    prog.add_pulse(freq, phase, amp, tstart, env_i + 1j*env_q, 'Q0.rdlo')
    prog.add_pulse(freq, phase, amp, tstart+300, env_i + 1j*env_q, 'Q0.rdlo', save_result=False)
    prog.add_done_stb()
    prog_exe = prog.get_program_binaries()
    sim_prog = prog.get_sim_program()
    pulse_seq = [sim_prog[1]]

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)

    cocotb.start_soon(dspunit.generate_adc_signal(adc_signal, 0))
    await dspunit.run_program(2500, nshots=5)
    acc_buf0 = await dspunit.read_acc_buf(10)

    print(f'acc buf 0: {acc_buf0}')

    channel_configs = load_channel_configs('channel_config.json')
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.add_phase_reset()
    #prog.add_pulse(freq, phase, amp, tstart, env_i + 1j*env_q, 1)
    prog.add_pulse(freq, phase + np.pi/2, amp, tstart, env_i + 1j*env_q, 'Q0.rdlo')
    prog.add_pulse(freq, phase + np.pi/2, amp, tstart+300, env_i + 1j*env_q, 'Q0.rdlo', save_result=True)
    prog.add_done_stb()
    prog_exe = prog.get_program_binaries()
    sim_prog = prog.get_sim_program()
    pulse_seq = [sim_prog[1]]

    #cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)

    cocotb.start_soon(dspunit.generate_adc_signal(adc_signal, 0))
    await dspunit.run_program(500)
    acc_buf1 = await dspunit.read_acc_buf(10)
    print(f'acc buf 1 {acc_buf1}')
    
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

    qdrvelemcfg = RFSoCElementCfg(16, interp_ratio=4)
    rdrvelemcfg = RFSoCElementCfg(16, interp_ratio=16)
    rdloelemcfg = RFSoCElementCfg(4, interp_ratio=4)

    adc_fullscale = 2**15-1

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    adc_signal = adc_fullscale*np.cos(2*np.pi*freq*(1.e-9*2*dsp.CLK_CYCLE/dspunit.adc_samples_per_clk)*np.arange(40000) + phase)
    adc_signal[-1] = 0

    channel_configs = load_channel_configs('channel_config.json')
    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo']}
    elem_configs = {k: elemconfig_classfactory(v.elem_type)(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.add_phase_reset()
    prog.add_reg_write('n_iters', niters)
    prog.add_reg_write('i', 0)
    prog.add_reg_write('phase', 0, dtype='phase')
    prog.add_pulse(freq, 'phase', amp, tstart, env_i + 1j*env_q, 'Q0.rdlo', label='PULSE', save_result=True)
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

#@cocotb.test()
#async def test_vna(dut):
#    fr = 6.555e9
#    f0 = fr-3e6
#    f1 = fr+3e6
#    nfreq = 512
#    freqs = np.linspace(f0, f1, nfreq)
#    rdlo_delay = 100
#    rdrv_amp = 0.25
#    pulse_length = 2000 #length in samples
#    rdrv_plength = pulse_length #env size, divide by 16 b/c interpolation
#    rdlo_plength = rdrv_plength
#    
#    vna_prog = [{'op': 'declare_freq', 'freq': f, 'dest': 'Q0.rdrv'} for f in freqs]
#    vna_prog.extend([{'op':'declare_freq', 'freq': f, 'dest': 2} for f in freqs])
#    vna_prog.append({'op': 'phase_reset'})
#    vna_prog.append({'op': 'reg_write', 'name': 'freq', 'value': 0})
#    vna_prog.append({'op': 'reg_write', 'name': 'i', 'value': 0})
#    vna_prog.append({'op': 'reg_write', 'name': 'niters', 'value': nfreq})
#    
#    vna_prog.append({'op': 'pulse', 'freq': 'freq', 'env': np.ones(rdrv_plength), 'amp': rdrv_amp, 
#                     'phase': 0, 'start_time': 20, 'dest': 'Q0.rdrv', 'label': 'PULSE_START'})
#    vna_prog.append({'op': 'pulse', 'freq': 'freq', 'env': np.ones(rdlo_plength), 'amp': 1, 
#                     'phase': 0, 'start_time': 20 + rdlo_delay, 'dest': 2})
#    
#    vna_prog.append({'op': 'reg_alu', 'in0': 1, 'in1_reg': 'freq', 'out_reg': 'freq', 'alu_op': 'add'})
#    vna_prog.append({'op': 'reg_alu', 'in0': 1, 'in1_reg': 'i', 'out_reg': 'i', 'alu_op': 'add'})
#    vna_prog.append({'op': 'inc_qclk', 'in0': -(20 + rdlo_delay + pulse_length + 100)})
#    vna_prog.append({'op': 'jump_cond', 'in0': 'i', 'in1_reg': 'niters', 'jump_label': 'PULSE_START', 'alu_op': 'le'})
#    vna_prog.append({'op': 'done_stb'})
#
#    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
#    # adc_signal = adc_fullscale*np.cos(2*np.pi*freq*(1.e-9*dsp.CLK_CYCLE/dspunit.adc_samples_per_clk)*np.arange(40000) + phase)
#    # adc_signal[-1] = 0
#
#    qdrvelemcfg = RFSoCElementCfg(16)
#    rdrvelemcfg = RFSoCElementCfg(16, interp_ratio=16)
#    rdloelemcfg = RFSoCElementCfg(4, interp_ratio=4)
#
#    prog = asm.SingleCoreAssembler([qdrvelemcfg, rdrvelemcfg, rdloelemcfg])
#    prog.from_list(vna_prog)
#
#    prog_exe = prog.get_program_binaries()
#    sim_prog = prog.get_sim_program()
#
#    cocotb.start_soon(dsp.generate_clock(dut))
#    await dspunit.load_env_buffer(env_buffers[0], 0, 0)
#    await dspunit.load_freq_buffer(freq_buffers[0], 0, 0)
#    await dspunit.load_env_buffer(env_buffers[1], 1, 0)
#    await dspunit.load_freq_buffer(freq_buffers[1], 1, 0)
#    await dspunit.load_env_buffer(env_buffers[2], 2, 0)
#    await dspunit.load_freq_buffer(freq_buffers[2], 2, 0)
#
#    # cocotb.start_soon(dspunit.generate_adc_signal(adc_signal, 0))
#    await dspunit.run_program(50000)
#    acc_buf = await dspunit.read_acc_buf(100)
#    plt.plot(np.real(acc_buf), np.imag(acc_buf), '.')
#    plt.show()

@cocotb.test()
async def test_compile_chain_linear(dut):
    qchip = qc.QChip('qubitcfg.json')
    fpga_config = {'alu_instr_clks': 2,
                   'fpga_clk_period': 2.e-9,
                   'jump_cond_clks': 3,
                   'jump_fproc_clks': 4,
                   'pulse_regwrite_clks': 1}
    program = [{'name': 'X90', 'qubit': ['Q0']},
               {'name': 'X90', 'qubit': ['Q1']},
               {'name': 'read', 'qubit': ['Q0']}]
    fpga_config = hw.FPGAConfig(**fpga_config)
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    #compiled_prog = cm.CompiledProgram(compiler.asm_progs, fpga_config)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()
    #ipdb.set_trace()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    # adc_signal = adc_fullscale*np.cos(2*np.pi*freq*(1.e-9*dsp.CLK_CYCLE/dspunit.adc_samples_per_clk)*np.arange(40000) + phase)
    # adc_signal[-1] = 0

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(1000)
    sim_prog = globalasm.assemblers[0].get_sim_program()
    #dacout_sim = st.generate_sim_dacout(sim_prog[1], 16)
    plt.plot(dspunit.dac_out[0])
    #plt.plot(dacout_sim)
    plt.show()

@cocotb.test()
async def test_compile_chain_pulse(dut):
    qchip = qc.QChip('qubitcfg.json')
    fpga_config = {'alu_instr_clks': 2,
                   'fpga_clk_period': 2.e-9,
                   'jump_cond_clks': 3,
                   'jump_fproc_clks': 4,
                   'pulse_regwrite_clks': 1}
    program = [{'name': 'pulse', 'phase': np.pi/2, 'freq': 'Q0.freq', 'env': np.ones(24*16//2),
                'twidth': 24.e-9, 'amp': 0.5, 'dest': 'Q0.qdrv'},
               {'name': 'read', 'qubit': ['Q0']}]
    fpga_config = hw.FPGAConfig(**fpga_config)
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    #compiled_prog = cm.CompiledProgram(compiler.asm_progs, fpga_config)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    # adc_signal = adc_fullscale*np.cos(2*np.pi*freq*(1.e-9*dsp.CLK_CYCLE/dspunit.adc_samples_per_clk)*np.arange(40000) + phase)
    # adc_signal[-1] = 0

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(1000)
    sim_prog = globalasm.assemblers[0].get_sim_program()
    #dacout_sim = st.generate_sim_dacout(sim_prog[1], 16)
    plt.plot(dspunit.dac_out[0])
    #plt.plot(dacout_sim)
    plt.show()

@cocotb.test()
async def test_fproc_read(dut):
    dac_samples_per_clk = 16
    adc_samples_per_clk = 4
    rdrv_interp_ratio = 16
    rdlo_interp_ratio = 4
    progdict = [{'op' : 'declare_reg', 'name' : 'fproc_meas0'},
            {'op' : 'phase_reset'},
            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 5, 'dest': 'Q0.rdrv'}, 
            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 25, 'dest': 'Q0.rdlo'}, 
            {'op' : 'idle', 'end_time': 25 + 200 + 64},
            {'op' : 'alu_fproc', 'in0': 0, 'alu_op': 'id1', 'out_reg': 'fproc_meas0'},
            {'op' : 'done_stb'}]

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    channel_configs = load_channel_configs('channel_config.json')
    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo']}
    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.from_list(progdict)
    prog_exe = prog.get_program_binaries()

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)
    await dspunit.run_program(500)

@cocotb.test()
async def test_fproc_jump(dut):
    progdict = [{'op' : 'declare_reg', 'name' : 'fproc_meas0'},
            {'op' : 'phase_reset'},
            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 5, 'dest': 'Q0.rdrv'}, 
            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 25, 'dest': 'Q0.rdlo', 'save_result': False}, 
            {'op' : 'idle', 'end_time': 25 + 200 + 64},
            {'op' : 'jump_fproc', 'in0': 1, 'alu_op': 'eq', 'jump_label': 'done'},
            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 300, 'dest': 'Q0.qdrv'}, 
            {'op' : 'done_stb', 'label': 'done'}]

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    channel_configs = load_channel_configs('channel_config.json')
    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo']}
    elem_configs = {k: elemconfig_classfactory(v.elem_type)(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.from_list(progdict)
    prog_exe = prog.get_program_binaries()

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)
    await dspunit.run_program(500)
    assert np.all(dspunit.dac_out[0] == 0)

@cocotb.test()
async def test_fproc_notjump(dut):
    progdict = [{'op' : 'declare_reg', 'name' : 'fproc_meas0'},
            {'op' : 'phase_reset'},
            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 5, 'dest': 'Q0.rdrv'}, 
            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 25, 'dest': 'Q0.rdlo', 'save_result': False}, 
            {'op' : 'idle', 'end_time': 25 + 200 + 64},
            {'op' : 'jump_fproc', 'in0': 0, 'alu_op': 'eq', 'jump_label': 'done'},
            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 302, 'dest': 'Q0.qdrv'}, 
            {'op' : 'done_stb', 'label': 'done'}]

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    channel_configs = load_channel_configs('channel_config.json')
    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo']}
    elem_configs = {k: elemconfig_classfactory(v.elem_type)(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.from_list(progdict)
    prog_exe = prog.get_program_binaries()

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)
    await dspunit.run_program(500)
    assert not np.all(dspunit.dac_out[0] == 0)

@cocotb.test()
async def test_fproc_jump_ph_adj(dut):
    progdict = [{'op' : 'declare_reg', 'name' : 'fproc_meas0'},
            {'op' : 'phase_reset'},
            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 5, 'dest': 'Q0.rdrv'}, 
            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : np.pi, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 25, 'dest': 'Q0.rdlo'}, 
            {'op' : 'idle', 'end_time': 25 + 200 + 64},
            {'op' : 'jump_fproc', 'in0': 0, 'alu_op': 'eq', 'jump_label': 'done'},
            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 302, 'dest': 'Q0.qdrv'}, 
            {'op' : 'done_stb', 'label': 'done'}]

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    qdrvelemcfg = RFSoCElementCfg(16, interp_ratio=4)
    rdrvelemcfg = RFSoCElementCfg(16, interp_ratio=16)
    rdloelemcfg = RFMixElementCfg(4, interp_ratio=4)
    channel_configs = load_channel_configs('channel_config.json')
    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo']}
    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.from_list(progdict)
    prog_exe = prog.get_program_binaries()

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)
    await dspunit.run_program(500)
    assert np.all(dspunit.dac_out[0] == 0)


@cocotb.test()
async def test_fproc_jump_diffcore(dut):
    """
    Same as above but change rdlo phase to 0
    """
    freq = 347.e6
    progdict1 = [{'op' : 'phase_reset'},
            {'op' : 'pulse', 'freq' : freq, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 5, 'dest': 'Q1.rdrv'}, 
            {'op' : 'pulse', 'freq' : freq, 'phase' : np.pi, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 25, 'dest': 'Q1.rdlo', 'save_result': False}, 
            {'op' : 'done_stb', 'label': 'done'}]

    progdict2 = [{'op' : 'declare_reg', 'name' : 'fproc_meas0'},
            {'op' : 'phase_reset'},
            {'op' : 'idle', 'end_time': 25 + 200 + 64},
            {'op' : 'jump_fproc', 'in0': 0, 'alu_op': 'eq', 'jump_label': 'done', 'func_id': 1},
            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 302, 'dest': 'Q2.qdrv'}, 
            {'op' : 'done_stb', 'label': 'done'}]


    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    qdrvelemcfg = RFSoCElementCfg(16, interp_ratio=4)
    rdrvelemcfg = RFSoCElementCfg(16, interp_ratio=16)
    rdloelemcfg = RFSoCElementCfg(4, interp_ratio=4)

    channel_configs = load_channel_configs('channel_config.json')
    q1_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q1.qdrv', 'Q1.rdrv', 'Q1.rdlo']}
    elem_configs = {k: elemconfig_classfactory(v.elem_type)(**v.elem_params) for k, v in q1_chan_cfgs.items()}
    prog1 = asm.SingleCoreAssembler(q1_chan_cfgs, elem_configs)

    q2_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q2.qdrv', 'Q2.rdrv', 'Q2.rdlo']}
    elem_configs = {k: elemconfig_classfactory(v.elem_type)(**v.elem_params) for k, v in q2_chan_cfgs.items()}
    prog2 = asm.SingleCoreAssembler(q2_chan_cfgs, elem_configs)

    prog1.from_list(progdict1)
    prog2.from_list(progdict2)
    prog_exe = prog1.get_program_binaries()
    prog_exe += prog2.get_program_binaries()

    adc_signal = (2**15-1)*np.cos(2*np.pi*freq*(1.e-9*dsp.CLK_CYCLE/dspunit.adc_samples_per_clk)*np.arange(500))
    adc_signal[-1] = 0

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)

    cocotb.start_soon(dspunit.generate_adc_signal(adc_signal, 1))
    await dspunit.run_program(500)
    assert np.all(dspunit.dac_out[2] == 0)


@cocotb.test()
async def test_fproc_nojump_diffcore(dut):
    freq = 347.e6
    progdict1 = [{'op' : 'phase_reset'},
            {'op' : 'pulse', 'freq' : freq, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 5, 'dest': 'Q1.rdrv'}, 
            {'op' : 'pulse', 'freq' : freq, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 25, 'dest': 'Q1.rdlo'}, 
            {'op' : 'done_stb', 'label': 'done'}]

    progdict2 = [{'op' : 'declare_reg', 'name' : 'fproc_meas0'},
            {'op' : 'phase_reset'},
            {'op' : 'idle', 'end_time': 25 + 200 + 64},
            {'op' : 'jump_fproc', 'in0': 0, 'alu_op': 'eq', 'jump_label': 'done', 'func_id': 1},
            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
             'start_time': 302, 'dest': 'Q2.qdrv'}, 
            {'op' : 'done_stb', 'label': 'done'}]


    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    channel_configs = load_channel_configs('channel_config.json')
    q1_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q1.qdrv', 'Q1.rdrv', 'Q1.rdlo']}
    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q1_chan_cfgs.items()}
    prog1 = asm.SingleCoreAssembler(q1_chan_cfgs, elem_configs)

    q2_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q2.qdrv', 'Q2.rdrv', 'Q2.rdlo']}
    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q2_chan_cfgs.items()}
    prog2 = asm.SingleCoreAssembler(q2_chan_cfgs, elem_configs)

    prog1.from_list(progdict1)
    prog2.from_list(progdict2)
    prog_exe = prog1.get_program_binaries()
    prog_exe += prog2.get_program_binaries()

    adc_signal = (2**15-1)*np.cos(2*np.pi*freq*(1.e-9*dsp.CLK_CYCLE/dspunit.adc_samples_per_clk)*np.arange(500))
    adc_signal[-1] = 0

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)

    cocotb.start_soon(dspunit.generate_adc_signal(adc_signal, 1))
    await dspunit.run_program(500)
    assert not np.all(dspunit.dac_out[2] == 0)

@cocotb.test()
async def test_fproc_reset_compile(dut):
    qchip = qc.QChip('qubitcfg.json')
    program = [{'name': 'X90', 'qubit': ['Q0']},
               {'name': 'read_test', 'qubit': ['Q0']},
               {'name': 'branch_fproc', 'alu_cond': 'eq', 'cond_lhs': 1, 'func_id': 'Q0.meas', 'scope': ['Q0'],
                'true': [#{'name': 'delay', 't': 146.e-9, 'qubit': ['Q0']},
                             {'name': 'X90', 'qubit': ['Q0']}, 
                             {'name': 'X90', 'qubit': ['Q0']}], 
                    'false': []},
               {'name': 'read_test', 'qubit': ['Q0']}]
    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(1500)
    plt.plot(dspunit.dac_out[0])
    plt.show()
    #assert not np.all(dspunit.dac_out[2] == 0)
    #plt.plot(dspunit.dac_out[0])
    assert not np.all(dspunit.dac_out[0][1250:] == 0)

@cocotb.test()
async def test_fproc_noreset_compile(dut):
    qchip = qc.QChip('qubitcfg.json')
    program = [{'name': 'X90', 'qubit': ['Q0']},
               {'name': 'read_test', 'qubit': ['Q0']},
               {'name': 'branch_fproc', 'alu_cond': 'eq', 'cond_lhs': 0, 'func_id': 'Q0.meas', 'scope': ['Q0'],
                'true': [{'name': 'delay', 't': 146.e-9, 'qubit': ['Q0']},
                             {'name': 'X90', 'qubit': ['Q0']}, 
                             {'name': 'X90', 'qubit': ['Q0']}], 
                    'false': []},
               {'name': 'read_test', 'qubit': ['Q0']}]
    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    #compiled_prog = cm.CompiledProgram(compiler.asm_progs, fpga_config)
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(1500)
    plt.plot(dspunit.dac_out[0])
    plt.show()
    assert np.all(dspunit.dac_out[0][1250:] == 0)

@cocotb.test()
async def test_asm_cw(dut):
    progdict = [
            {'op' : 'phase_reset'},
            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : 'cw', 'amp' : 0.5, 
             'start_time': 50, 'dest': 'Q0.qdrv'}, 
            {'op' : 'pulse', 'freq' : 750.e6, 'phase' : np.pi/2, 'env' : np.ones(200), 'amp' : 0.7, 
             'start_time': 500, 'dest': 'Q0.qdrv'}, 
            {'op' : 'done_stb', 'label': 'done'}]

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    qdrvelemcfg = RFSoCElementCfg(16, interp_ratio=4)
    rdrvelemcfg = RFSoCElementCfg(16, interp_ratio=16)
    rdloelemcfg = RFSoCElementCfg(4, interp_ratio=4)
    channel_configs = load_channel_configs('channel_config.json')
    q0_chan_cfgs = {k: v for k, v in channel_configs.items() if k in ['Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo']}
    elem_configs = {k: RFSoCElementCfg(**v.elem_params) for k, v in q0_chan_cfgs.items()}
    prog = asm.SingleCoreAssembler(q0_chan_cfgs, elem_configs)
    prog.from_list(progdict)
    prog_exe = prog.get_program_binaries()

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(prog_exe)
    await dspunit.run_program(1000)

    sim_pulse = [
            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : np.ones(450*4), 'amp' : 0.5, 
             'start_time': 50, 'dest': 'Q0.rdrv'},
            {'op' : 'pulse', 'freq' : 750.e6, 'phase' : np.pi/2, 'env' : np.ones(200), 'amp' : 0.7, 
             'start_time': 500, 'dest': 'Q0.qdrv'}] 

    dacout_sim = st.generate_sim_dacout(sim_pulse, 16, interp_ratio=INTERP_RATIO, extra_delay=2)
    plt.plot(dspunit.dac_out[0])
    plt.plot(dacout_sim)
    plt.show()

    #TODO: fix amp issue
    assert st.check_dacout_equal(dacout_sim, dspunit.dac_out[0])

@cocotb.test()
async def test_compile_cw(dut):
    qchip = qc.QChip('qubitcfg.json')
    fpga_config = {'alu_instr_clks': 4,
                   'fpga_clk_period': 2.e-9,
                   'jump_cond_clks': 5,
                   'jump_fproc_clks': 5,
                   'pulse_regwrite_clks': 4}
    program = [
            {'name' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : 'cw', 'amp' : 0.99, 
             'start_time': 50, 'twidth': 100.e-9, 'dest': 'Q0.qdrv'}]

    fpga_config = hw.FPGAConfig(**fpga_config)
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    #compiled_prog = cm.CompiledProgram(compiler.asm_progs, fpga_config)
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()


    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    qdrvelemcfg = RFSoCElementCfg(16, interp_ratio=4)
    rdrvelemcfg = RFSoCElementCfg(16, interp_ratio=16)
    rdloelemcfg = RFSoCElementCfg(4, interp_ratio=4)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(1000)


    plt.plot(dspunit.dac_out[0])
    plt.show()

@cocotb.test()
async def test_cond_bitflip(dut):
    qchip = qc.QChip('qubitcfg.json')
    program = [
        {'name': 'X90', 'qubit': ['Q1']},
        {'name': 'read', 'qubit': ['Q1']},
        {'name': 'branch_fproc', 'alu_cond': 'eq', 'cond_lhs': 1, 'func_id': 'Q1.meas', 'scope': ['Q0', 'Q1'],
                    'true': [
                                 {'name': 'X90', 'qubit': ['Q0']}, 
                                 {'name': 'X90', 'qubit': ['Q0']}], 
                    'false': []},
        {'name': 'barrier', 'qubit':['Q0', 'Q1']},
        {'name': 'read', 'qubit': ['Q0']},
        {'name': 'read', 'qubit': ['Q1']},
    
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    #compiled_prog = cm.CompiledProgram(compiler.asm_progs, fpga_config)
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)
    for statement in compiled_prog.program[('Q1.qdrv', 'Q1.rdrv', 'Q1.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(1500)
    #assert np.all(dspunit.dac_out[0][1250:] == 0)


#@cocotb.test()
#async def test_long_twidth(dut):
#    progdict = [
#            {'op' : 'phase_reset'},
#            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : 
#             {'env_func': 'cos_edge_square', 'paradict': {'ramp_fraction': 0.25, 'twidth': 5.12e-7}}, 'amp' : 0.5, 
#             'start_time': 5, 'dest': 0}, 
#            {'op' : 'pulse', 'freq' : 347.e6, 'phase' : 0, 'env' : np.ones(200), 'amp' : 0.5, 
#             'start_time': 25, 'dest': 1}, 
#            {'op' : 'done_stb'}]
#
#    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
#    qdrvelemcfg = RFSoCElementCfg(16)
#    rdrvelemcfg = RFSoCElementCfg(16, interp_ratio=16)
#    rdloelemcfg = RFSoCElementCfg(4, interp_ratio=4)
#    prog.from_list(progdict)
#    prog_exe = prog.get_program_binaries()
#
#    cocotb.start_soon(dsp.generate_clock(dut))
#    await dspunit.load_program([cmd_buf])
#    await dspunit.load_env_buffer(env_buffers[0], 0, 0)
#    await dspunit.load_freq_buffer(freq_buffers[0], 0, 0)
#    await dspunit.load_env_buffer(env_buffers[1], 1, 0)
#    await dspunit.load_freq_buffer(freq_buffers[1], 1, 0)
#    await dspunit.load_env_buffer(env_buffers[2], 2, 0)
#    await dspunit.load_freq_buffer(freq_buffers[2], 2, 0)
#    await dspunit.run_program(500)

@cocotb.test()
async def test_branch_reduce_fproc_eq(dut):
    qchip = qc.QChip('qubitcfg.json')
    program = [
        {'name': 'read', 'qubit': ['Q0']},
        {'name': 'branch_reduce_fproc', 'alu_cond': 'eq', 'cond_lhs': 0, 'func_id': 'Q0.meas', 'scope': ['Q0'],
            'true': [{'name': 'X90', 'qubit': ['Q0']}],
            'false': []}
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(1500)

    assert not np.all(dspunit.dac_out[0] == 0)

def walk_hierarchy(obj, prefix=""):
    for c in obj:
        print(prefix + c._name)
        walk_hierarchy(c, prefix + c._name + ".")


@cocotb.test()
async def test_branch_reduce_fproc_and(dut):
    qchip = qc.QChip('qubitcfg.json')
    # For logic operations `cond_lhs` is the mask that selects the input bits,
    # e.g. 0b101 selects bit 0 and 2, ignores bit 1
    program = [
        {'name': 'read', 'qubit': ['Q0']},
        {'name': 'branch_reduce_fproc', 'alu_cond': 'and', 'cond_lhs': 0b11010, 'func_id': 'Q0.meas', 'scope': ['Q0'],
            'true': [],
            'false': [{'name': 'X90', 'qubit': ['Q0']}]}
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    #dut.dspmod.cores.qubit[0].qubit_inst.dproc.myalu.in1
    await dspunit.run_program(1500)

    assert not np.all(dspunit.dac_out[0] == 0)


@cocotb.test()
async def test_branch_reduce_fproc_maj(dut):
    qchip = qc.QChip('qubitcfg.json')
    # For logic operations `cond_lhs` is the mask that selects the input bits,
    # e.g. 0b101 selects bit 0 and 2, ignores bit 1
    program = [
        {'name': 'X90', 'qubit': ['Q0']},
        {'name': 'X90', 'qubit': ['Q1']},
        {'name': 'X90', 'qubit': ['Q2']},
        {'name': 'read', 'qubit': ['Q0']},
        {'name': 'read', 'qubit': ['Q1']},
        {'name': 'read', 'qubit': ['Q2']},
        {'name': 'branch_reduce_fproc', 'alu_cond': 'maj', 'cond_lhs': 0b111, 'func_id': 'Q0.meas', 'scope': ['Q0', 'Q1', 'Q2'],
            'true': [],
            'false': [{'name': 'X90', 'qubit': ['Q0']}]}
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    #dut.dspmod.cores.qubit[0].qubit_inst.dproc.myalu.in1
    await dspunit.run_program(1500)

    assert not np.all(dspunit.dac_out[0] == 0)


@cocotb.test()
async def test_branch_reduce_fproc_xor(dut):
    qchip = qc.QChip('qubitcfg.json')
    # For logic operations `cond_lhs` is the mask that selects the input bits,
    # e.g. 0b101 selects bit 0 and 2, ignores bit 1
    program = [
        {'name': 'X90', 'qubit': ['Q0']},
        {'name': 'X90', 'qubit': ['Q1']},
        {'name': 'X90', 'qubit': ['Q2']},
        {'name': 'read', 'qubit': ['Q0']},
        {'name': 'read', 'qubit': ['Q1']},
        {'name': 'read', 'qubit': ['Q2']},
        {'name': 'branch_reduce_fproc', 'alu_cond': 'xor', 'cond_lhs': 0b111, 'func_id': 'Q0.meas', 'scope': ['Q0', 'Q1', 'Q2'],
            'true': [],
            'false': [{'name': 'X90', 'qubit': ['Q0']}]}
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    #dut.dspmod.cores.qubit[0].qubit_inst.dproc.myalu.in1
    await dspunit.run_program(1500)

    assert not np.all(dspunit.dac_out[0] == 0)


@cocotb.test()
async def test_branch_fproc_nested(dut):
    qchip = qc.QChip('qubitcfg.json')
    program = [
        {'name': 'read', 'qubit': ['Q0']},
        {'name': 'branch_fproc', 'alu_cond': 'eq', 'cond_lhs': 1, 'func_id': 'Q0.meas', 'scope': ['Q0', 'Q1', 'Q2'],
            'true': [],
            'false': [{'name': 'read', 'qubit': ['Q1']}, 
                {'name': 'branch_fproc', 'alu_cond': 'eq', 'cond_lhs': 1, 'func_id': 'Q1.meas', 'scope': ['Q0', 'Q1', 'Q2'],
                    'true': [],
                    'false': [{'name': 'read', 'qubit': ['Q2']},
                        {'name': 'branch_fproc', 'alu_cond': 'eq', 'cond_lhs': 1, 'func_id': 'Q2.meas', 'scope': ['Q0', 'Q1', 'Q2'],
                            'true': [],
                            'false': [{'name': 'X90', 'qubit': ['Q0']}]
                }]
            }]
        }
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(5000)

    assert not np.all(dspunit.dac_out[0] == 0)


@cocotb.test()
async def test_branch_fproc_maj(dut):
    qchip = qc.QChip('qubitcfg.json')
    program = [
        {'name': 'X90', 'qubit': ['Q0']},
        {'name': 'X90', 'qubit': ['Q1']},
        {'name': 'X90', 'qubit': ['Q2']},
        {'name': 'read', 'qubit': ['Q0']},
        {'name': 'read', 'qubit': ['Q1']},
        {'name': 'read', 'qubit': ['Q2']},
        {
            'name': 'branch_fproc', 
            'alu_cond': 'eq',          # Check if Q0 == 1
            'cond_lhs': 1,         # Qubit 0 outcome
            'func_id': 'Q0.meas', 'scope': ['Q0'], # Not tied to measurement func
            'true': [ # Case: Q0 == 1
                {
                    'name': 'branch_fproc', 
                    'alu_cond': 'eq',      # Check if Q1 == 1 (given Q0 == 1)
                    'cond_lhs': 1,     # Qubit 1 outcome
                    'func_id': 'Q1.meas', 'scope': ['Q0'],
                    'true': [ # Case: Q0 == 1 AND Q1 == 1 --> Majority is 1
                    ],
                    'false': [ # Case: Q0 == 1 AND Q1 == 0
                        {
                            'name': 'branch_fproc', 
                            'alu_cond': 'eq',  # Check if Q2 == 1 (given Q0==1, Q1==0)
                            'cond_lhs': 1, # Qubit 2 outcome
                            'func_id': 'Q2.meas', 'scope': ['Q0'],
                            'true': [ # Case: Q0==1, Q1==0, Q2==1 --> Majority is 1
                            ],
                            'false': [ # Case: Q0==1, Q1==0, Q2==0 --> Majority is 0
                            ]
                        }
                    ]
                }
            ],
            'false': [ # Case: Q0 == 0
                {
                    'name': 'branch_fproc', 
                    'alu_cond': 'eq',      # Check if Q1 == 1 (given Q0 == 0)
                    'cond_lhs': 1,     # Qubit 1 outcome
                    'func_id': 'Q1.meas', 'scope': ['Q0'],
                    'true': [ # Case: Q0 == 0 AND Q1 == 1 
                        {
                            'name': 'branch_fproc', 
                            'alu_cond': 'eq',  # Check if Q2 == 1 (given Q0==0, Q1==1)
                            'cond_lhs': 1, # Qubit 2 outcome
                            'func_id': 'Q2.meas', 'scope': ['Q0'],
                            'true': [ # Case: Q0==0, Q1==1, Q2==1 --> Majority is 1
                            ],
                            'false': [ # Case: Q0==0, Q1==1, Q2==0 --> Majority is 0
                            ]
                        }
                    ],
                    'false': [ # Case: Q0 == 0 AND Q1 == 0 
                        # If Q0=0 and Q1=0, majority is 0 regardless of Q2, issue command for time measurement
                        {
                            'name': 'branch_fproc', 
                            'alu_cond': 'eq',  # Check if Q2 == 1 (given Q0==0, Q1==0)
                            'cond_lhs': 1, # Qubit 2 outcome
                            'func_id': 'Q2.meas', 'scope': ['Q0'],
                            'true': [ # Case: Q0==0, Q1==0, Q2==1 --> Majority is 0
                            ],
                            'false': [ # Case: Q0==0, Q1==0, Q2==0 --> Majority is 0
                                {'name': 'X90', 'qubit': ['Q0']} # or do nothing
                            ]
                        }
                    ]
                }
            ]
        }
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(5000)

    assert not np.all(dspunit.dac_out[0] == 0)


@cocotb.test()
async def test_branch_fproc_xor(dut):
    qchip = qc.QChip('qubitcfg.json')
    program = [
        {'name': 'X90', 'qubit': ['Q0']},
        {'name': 'X90', 'qubit': ['Q1']},
        {'name': 'X90', 'qubit': ['Q2']},
        {'name': 'read', 'qubit': ['Q0']},
        {'name': 'read', 'qubit': ['Q1']},
        {'name': 'read', 'qubit': ['Q2']},
        {
            'name': 'branch_fproc', 
            'alu_cond': 'eq',          # Check if Q0 == 1
            'cond_lhs': 1,             # Compare against 1
            'func_id': 'Q0.meas', 'scope': ['Q0'], # Not tied to measurement func
            'true': [ # Case: Q0 == 1
                {
                    'name': 'branch_fproc', 
                    'alu_cond': 'eq',      # Check if Q1 == 1 (given Q0 == 1)
                    'cond_lhs': 1,     # Qubit 1 outcome
                    'func_id': 'Q1.meas', 'scope': ['Q0'],
                    'true': [ # Case: Q0 == 1 AND Q1 == 1
                        {
                            'name': 'branch_fproc', 
                            'alu_cond': 'eq',  # Check if Q2 == 1 (given Q0==1, Q1==1)
                            'cond_lhs': 1, # Qubit 2 outcome
                            'func_id': 'Q2.meas', 'scope': ['Q0'],
                            'true': [ # Case: Q0==1, Q1==1, Q2==1 --> XOR is 1 (odd number of 1s)
                                {'name': 'X90', 'qubit': ['Q0']} # or do nothing
                            ],
                            'false': [ # Case: Q0==1, Q1==1, Q2==0 --> XOR is 0 (even number of 1s)
                            ]
                        }
                    ],
                    'false': [ # Case: Q0 == 1 AND Q1 == 0
                        {
                            'name': 'branch_fproc', 
                            'alu_cond': 'eq',  # Check if Q2 == 1 (given Q0==1, Q1==0)
                            'cond_lhs': 1, # Qubit 2 outcome
                            'func_id': 'Q2.meas', 'scope': ['Q0'],
                            'true': [ # Case: Q0==1, Q1==0, Q2==1 --> XOR is 0 (even number of 1s)
                            ],
                            'false': [ # Case: Q0==1, Q1==0, Q2==0 --> XOR is 1 (odd number of 1s)
                                {'name': 'X90', 'qubit': ['Q0']} # or do nothing
                            ]
                        }
                    ]
                }
            ],
            'false': [ # Case: Q0 == 0
                {
                    'name': 'branch_fproc', 
                    'alu_cond': 'eq',      # Check if Q1 == 1 (given Q0 == 0)
                    'cond_lhs': 1,     # Qubit 1 outcome
                    'func_id': 'Q1.meas', 'scope': ['Q0'],
                    'true': [ # Case: Q0 == 0 AND Q1 == 1 
                        {
                            'name': 'branch_fproc', 
                            'alu_cond': 'eq',  # Check if Q2 == 1 (given Q0==0, Q1==1)
                            'cond_lhs': 1, # Qubit 2 outcome
                            'func_id': 'Q2.meas', 'scope': ['Q0'],
                            'true': [ # Case: Q0==0, Q1==1, Q2==1 --> XOR is 0 (even number of 1s)
                            ],
                            'false': [ # Case: Q0==0, Q1==1, Q2==0 --> XOR is 1 (odd number of 1s)
                                {'name': 'X90', 'qubit': ['Q0']} # or do nothing
                            ]
                        }
                    ],
                    'false': [ # Case: Q0 == 0 AND Q1 == 0 
                        {
                            'name': 'branch_fproc', 
                            'alu_cond': 'eq',  # Check if Q2 == 1 (given Q0==0, Q1==0)
                            'cond_lhs': 1, # Qubit 2 outcome
                            'func_id': 'Q2.meas', 'scope': ['Q0'],
                            'true': [ # Case: Q0==0, Q1==0, Q2==1 --> XOR is 1 (odd number of 1s)
                                {'name': 'X90', 'qubit': ['Q0']} # or do nothing
                            ],
                            'false': [ # Case: Q0==0, Q1==0, Q2==0 --> XOR is 0 (even number of 1s)
                            ]
                        }
                    ]
                }
            ]
        }
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(5000)

    assert not np.all(dspunit.dac_out[0] == 0)


@cocotb.test()
async def test_branch_fproc_maj_1(dut):
    qchip = qc.QChip('qubitcfg.json')
    program = [
        {'name': 'X90', 'qubit': ['Q0']},
        {'name': 'read', 'qubit': ['Q0']},
        {
            "name": "branch_fproc",
            "alu_cond": "eq",
            "func_id": "Q0.meas",
            "cond_lhs": 1,
            "scope": [
                "Q0"
            ],
            "true": [
                {
                    "name": "X90",
                    "qubit": [
                        "Q0"
                    ]
                }
            ],
            "false": [
                {
                    "name": "X90",
                    "qubit": [
                        "Q0"
                    ]
                }
            ]
        }
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(5000)

    assert not np.all(dspunit.dac_out[0] == 0)


@cocotb.test()
async def test_branch_fproc_maj_3(dut):
    qchip = qc.QChip('qubitcfg.json')
    program = [
        {'name': 'X90', 'qubit': ['Q0']},
        {'name': 'X90', 'qubit': ['Q1']},
        {'name': 'X90', 'qubit': ['Q2']},
        {'name': 'read', 'qubit': ['Q0']},
        {'name': 'read', 'qubit': ['Q1']},
        {'name': 'read', 'qubit': ['Q2']},
        {
            "name": "branch_fproc",
            "alu_cond": "eq",
            "func_id": "Q0.meas",
            "cond_lhs": 1,
            "scope": [
                "Q0"
            ],
            "true": [
                {
                    "name": "branch_fproc",
                    "alu_cond": "eq",
                    "func_id": "Q1.meas",
                    "cond_lhs": 1,
                    "scope": [
                        "Q0"
                    ],
                    "true": [
                        {
                            "name": "X90",
                            "qubit": [
                                "Q0"
                            ]
                        }
                    ],
                    "false": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "X90",
                                    "qubit": [
                                        "Q0"
                                    ]
                                }
                            ],
                            "false": []
                        }
                    ]
                }
            ],
            "false": [
                {
                    "name": "branch_fproc",
                    "alu_cond": "eq",
                    "func_id": "Q1.meas",
                    "cond_lhs": 1,
                    "scope": [
                        "Q0"
                    ],
                    "true": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "X90",
                                    "qubit": [
                                        "Q0"
                                    ]
                                }
                            ],
                            "false": []
                        }
                    ],
                    "false": [
                        {
                            "name": "X90",
                            "qubit": [
                                "Q0"
                            ]
                        }
                    ]
                }
            ]
        }
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(5000)

    assert not np.all(dspunit.dac_out[0] == 0)


@cocotb.test()
async def test_branch_fproc_maj_5(dut):
    qchip = qc.QChip('qubitcfg.json')
    program = [
        {'name': 'X90', 'qubit': ['Q0']},
        {'name': 'X90', 'qubit': ['Q1']},
        {'name': 'X90', 'qubit': ['Q2']},
        {'name': 'X90', 'qubit': ['Q3']},
        {'name': 'X90', 'qubit': ['Q4']},
        {'name': 'read', 'qubit': ['Q0']},
        {'name': 'read', 'qubit': ['Q1']},
        {'name': 'read', 'qubit': ['Q2']},
        {'name': 'read', 'qubit': ['Q3']},
        {'name': 'read', 'qubit': ['Q4']},
        {
            "name": "branch_fproc",
            "alu_cond": "eq",
            "func_id": "Q0.meas",
            "cond_lhs": 1,
            "scope": [
                "Q0"
            ],
            "true": [
                {
                    "name": "branch_fproc",
                    "alu_cond": "eq",
                    "func_id": "Q1.meas",
                    "cond_lhs": 1,
                    "scope": [
                        "Q0"
                    ],
                    "true": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "X90",
                                    "qubit": [
                                        "Q0"
                                    ]
                                }
                            ],
                            "false": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "X90",
                                            "qubit": [
                                                "Q0"
                                            ]
                                        }
                                    ],
                                    "false": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ],
                                            "false": []
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                    "false": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "X90",
                                            "qubit": [
                                                "Q0"
                                            ]
                                        }
                                    ],
                                    "false": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ],
                                            "false": []
                                        }
                                    ]
                                }
                            ],
                            "false": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ],
                                            "false": []
                                        }
                                    ],
                                    "false": []
                                }
                            ]
                        }
                    ]
                }
            ],
            "false": [
                {
                    "name": "branch_fproc",
                    "alu_cond": "eq",
                    "func_id": "Q1.meas",
                    "cond_lhs": 1,
                    "scope": [
                        "Q0"
                    ],
                    "true": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "X90",
                                            "qubit": [
                                                "Q0"
                                            ]
                                        }
                                    ],
                                    "false": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ],
                                            "false": []
                                        }
                                    ]
                                }
                            ],
                            "false": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ],
                                            "false": []
                                        }
                                    ],
                                    "false": []
                                }
                            ]
                        }
                    ],
                    "false": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ],
                                            "false": []
                                        }
                                    ],
                                    "false": []
                                }
                            ],
                            "false": []
                        }
                    ]
                }
            ]
        }
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(5000)

    assert not np.all(dspunit.dac_out[0] == 0)


@cocotb.test()
async def test_branch_fproc_xor_1(dut):
    qchip = qc.QChip('qubitcfg.json')
    program = [
        {'name': 'X90', 'qubit': ['Q0']},
        {'name': 'read', 'qubit': ['Q0']},
        {
            "name": "branch_fproc",
            "alu_cond": "eq",
            "func_id": "Q0.meas",
            "cond_lhs": 1,
            "scope": [
                "Q0"
            ],
            "true": [
                {
                    "name": "X90",
                    "qubit": [
                        "Q0"
                    ]
                }
            ],
            "false": [
                {
                    "name": "X90",
                    "qubit": [
                        "Q0"
                    ]
                }
            ]
        }
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(5000)

    assert not np.all(dspunit.dac_out[0] == 0)


@cocotb.test()
async def test_branch_fproc_xor_2(dut):
    qchip = qc.QChip('qubitcfg.json')
    program = [
        {'name': 'X90', 'qubit': ['Q0']},
        {'name': 'X90', 'qubit': ['Q1']},
        {'name': 'read', 'qubit': ['Q0']},
        {'name': 'read', 'qubit': ['Q1']},
        {
            "name": "branch_fproc",
            "alu_cond": "eq",
            "func_id": "Q0.meas",
            "cond_lhs": 1,
            "scope": [
                "Q0"
            ],
            "true": [
                {
                    "name": "branch_fproc",
                    "alu_cond": "eq",
                    "func_id": "Q1.meas",
                    "cond_lhs": 1,
                    "scope": [
                        "Q0"
                    ],
                    "true": [],
                    "false": [
                        {
                            "name": "X90",
                            "qubit": [
                                "Q0"
                            ]
                        }
                    ]
                }
            ],
            "false": [
                {
                    "name": "branch_fproc",
                    "alu_cond": "eq",
                    "func_id": "Q1.meas",
                    "cond_lhs": 1,
                    "scope": [
                        "Q0"
                    ],
                    "true": [
                        {
                            "name": "X90",
                            "qubit": [
                                "Q0"
                            ]
                        }
                    ],
                    "false": [
                        {
                            "name": "X90",
                            "qubit": [
                                "Q0"
                            ]
                        }
                    ]
                }
            ]
        }
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(5000)

    assert not np.all(dspunit.dac_out[0] == 0)


@cocotb.test()
async def test_branch_fproc_xor_3(dut):
    qchip = qc.QChip('qubitcfg.json')
    program = [
        {'name': 'X90', 'qubit': ['Q0']},
        {'name': 'X90', 'qubit': ['Q1']},
        {'name': 'X90', 'qubit': ['Q2']},
        {'name': 'read', 'qubit': ['Q0']},
        {'name': 'read', 'qubit': ['Q1']},
        {'name': 'read', 'qubit': ['Q2']},
        {
            "name": "branch_fproc",
            "alu_cond": "eq",
            "func_id": "Q0.meas",
            "cond_lhs": 1,
            "scope": [
                "Q0"
            ],
            "true": [
                {
                    "name": "branch_fproc",
                    "alu_cond": "eq",
                    "func_id": "Q1.meas",
                    "cond_lhs": 1,
                    "scope": [
                        "Q0"
                    ],
                    "true": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "X90",
                                    "qubit": [
                                        "Q0"
                                    ]
                                }
                            ],
                            "false": []
                        }
                    ],
                    "false": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [],
                            "false": [
                                {
                                    "name": "X90",
                                    "qubit": [
                                        "Q0"
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ],
            "false": [
                {
                    "name": "branch_fproc",
                    "alu_cond": "eq",
                    "func_id": "Q1.meas",
                    "cond_lhs": 1,
                    "scope": [
                        "Q0"
                    ],
                    "true": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [],
                            "false": [
                                {
                                    "name": "X90",
                                    "qubit": [
                                        "Q0"
                                    ]
                                }
                            ]
                        }
                    ],
                    "false": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "X90",
                                    "qubit": [
                                        "Q0"
                                    ]
                                }
                            ],
                            "false": [
                                {
                                    "name": "X90",
                                    "qubit": [
                                        "Q0"
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(5000)

    assert not np.all(dspunit.dac_out[0] == 0)


@cocotb.test()
async def test_branch_fproc_xor_4(dut):
    qchip = qc.QChip('qubitcfg.json')
    program = [
        {'name': 'X90', 'qubit': ['Q0']},
        {'name': 'X90', 'qubit': ['Q1']},
        {'name': 'X90', 'qubit': ['Q2']},
        {'name': 'X90', 'qubit': ['Q3']},
        {'name': 'read', 'qubit': ['Q0']},
        {'name': 'read', 'qubit': ['Q1']},
        {'name': 'read', 'qubit': ['Q2']},
        {'name': 'read', 'qubit': ['Q3']},
        {
            "name": "branch_fproc",
            "alu_cond": "eq",
            "func_id": "Q0.meas",
            "cond_lhs": 1,
            "scope": [
                "Q0"
            ],
            "true": [
                {
                    "name": "branch_fproc",
                    "alu_cond": "eq",
                    "func_id": "Q1.meas",
                    "cond_lhs": 1,
                    "scope": [
                        "Q0"
                    ],
                    "true": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [],
                                    "false": [
                                        {
                                            "name": "X90",
                                            "qubit": [
                                                "Q0"
                                            ]
                                        }
                                    ]
                                }
                            ],
                            "false": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "X90",
                                            "qubit": [
                                                "Q0"
                                            ]
                                        }
                                    ],
                                    "false": []
                                }
                            ]
                        }
                    ],
                    "false": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "X90",
                                            "qubit": [
                                                "Q0"
                                            ]
                                        }
                                    ],
                                    "false": []
                                }
                            ],
                            "false": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [],
                                    "false": [
                                        {
                                            "name": "X90",
                                            "qubit": [
                                                "Q0"
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ],
            "false": [
                {
                    "name": "branch_fproc",
                    "alu_cond": "eq",
                    "func_id": "Q1.meas",
                    "cond_lhs": 1,
                    "scope": [
                        "Q0"
                    ],
                    "true": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "X90",
                                            "qubit": [
                                                "Q0"
                                            ]
                                        }
                                    ],
                                    "false": []
                                }
                            ],
                            "false": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [],
                                    "false": [
                                        {
                                            "name": "X90",
                                            "qubit": [
                                                "Q0"
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                    "false": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [],
                                    "false": [
                                        {
                                            "name": "X90",
                                            "qubit": [
                                                "Q0"
                                            ]
                                        }
                                    ]
                                }
                            ],
                            "false": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "X90",
                                            "qubit": [
                                                "Q0"
                                            ]
                                        }
                                    ],
                                    "false": []
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(5000)

    assert not np.all(dspunit.dac_out[0] == 0)


@cocotb.test()
async def test_branch_fproc_xor_5(dut):
    qchip = qc.QChip('qubitcfg.json')
    program = [
        {'name': 'X90', 'qubit': ['Q0']},
        {'name': 'X90', 'qubit': ['Q1']},
        {'name': 'X90', 'qubit': ['Q2']},
        {'name': 'X90', 'qubit': ['Q3']},
        {'name': 'X90', 'qubit': ['Q4']},
        {'name': 'read', 'qubit': ['Q0']},
        {'name': 'read', 'qubit': ['Q1']},
        {'name': 'read', 'qubit': ['Q2']},
        {'name': 'read', 'qubit': ['Q3']},
        {'name': 'read', 'qubit': ['Q4']},
        {
            "name": "branch_fproc",
            "alu_cond": "eq",
            "func_id": "Q0.meas",
            "cond_lhs": 1,
            "scope": [
                "Q0"
            ],
            "true": [
                {
                    "name": "branch_fproc",
                    "alu_cond": "eq",
                    "func_id": "Q1.meas",
                    "cond_lhs": 1,
                    "scope": [
                        "Q0"
                    ],
                    "true": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ],
                                            "false": []
                                        }
                                    ],
                                    "false": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [],
                                            "false": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ],
                            "false": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [],
                                            "false": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ]
                                        }
                                    ],
                                    "false": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ],
                                            "false": []
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                    "false": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [],
                                            "false": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ]
                                        }
                                    ],
                                    "false": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ],
                                            "false": []
                                        }
                                    ]
                                }
                            ],
                            "false": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ],
                                            "false": []
                                        }
                                    ],
                                    "false": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [],
                                            "false": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ],
            "false": [
                {
                    "name": "branch_fproc",
                    "alu_cond": "eq",
                    "func_id": "Q1.meas",
                    "cond_lhs": 1,
                    "scope": [
                        "Q0"
                    ],
                    "true": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [],
                                            "false": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ]
                                        }
                                    ],
                                    "false": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ],
                                            "false": []
                                        }
                                    ]
                                }
                            ],
                            "false": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ],
                                            "false": []
                                        }
                                    ],
                                    "false": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [],
                                            "false": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                    "false": [
                        {
                            "name": "branch_fproc",
                            "alu_cond": "eq",
                            "func_id": "Q2.meas",
                            "cond_lhs": 1,
                            "scope": [
                                "Q0"
                            ],
                            "true": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ],
                                            "false": []
                                        }
                                    ],
                                    "false": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [],
                                            "false": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ],
                            "false": [
                                {
                                    "name": "branch_fproc",
                                    "alu_cond": "eq",
                                    "func_id": "Q3.meas",
                                    "cond_lhs": 1,
                                    "scope": [
                                        "Q0"
                                    ],
                                    "true": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [],
                                            "false": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ]
                                        }
                                    ],
                                    "false": [
                                        {
                                            "name": "branch_fproc",
                                            "alu_cond": "eq",
                                            "func_id": "Q4.meas",
                                            "cond_lhs": 1,
                                            "scope": [
                                                "Q0"
                                            ],
                                            "true": [
                                                {
                                                    "name": "X90",
                                                    "qubit": [
                                                        "Q0"
                                                    ]
                                                }
                                            ],
                                            "false": []
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]

    fpga_config = hw.FPGAConfig()
    channel_configs = hw.load_channel_configs('channel_config.json')
    compiler = cm.Compiler(program)
    compiler.run_ir_passes(cm.get_passes(fpga_config, qchip))
    compiled_prog = compiler.compile()
    for statement in compiled_prog.program[('Q0.qdrv', 'Q0.rdrv', 'Q0.rdlo')]:
        print(statement)

    globalasm = asm.GlobalAssembler(compiled_prog, channel_configs, elemconfig_classfactory)
    asmprog = globalasm.get_assembled_program()

    dspunit = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)

    cocotb.start_soon(dsp.generate_clock(dut))
    await dspunit.load_asm_program(asmprog)
    await dspunit.run_program(5000)

    assert not np.all(dspunit.dac_out[0] == 0)
