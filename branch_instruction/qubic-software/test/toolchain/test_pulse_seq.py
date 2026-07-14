import cocotb
import distproc.hwconfig as hw
import qubitconfig.qchip as qc
import qubic.toolchain as tc
import qubic.sim.cocotb_drivers as dsp
import matplotlib.pyplot as plt
import numpy as np
import ipdb
import os
import json

EXTRA_DELAY = 32 #one clock of extra delay

with open(os.path.join(os.environ['COCOTB_BUILD_PATH'], '../sim/gensrc/sim_memory_map.json')) as f:
    memory_map = json.load(f)

@cocotb.test()
async def test_X90_gate(dut):
    channel_config = hw.load_channel_configs('channel_config.json')
    qchip = qc.QChip('qubitcfg.json')
    prog = [{'name': 'X90', 'qubit': 'Q0'}]
    compiled_prog = tc.run_compile_stage(prog, fpga_config=hw.FPGAConfig(), qchip=qchip)
    binary = tc.run_assemble_stage(compiled_prog, channel_config)
    
    cocotb.start_soon(dsp.generate_clock(dut))
    driver = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    await driver.flush_cmd_mem()
    await driver.load_asm_program(binary)

    await driver.run_program(500)

    dac0_ref = np.load('test_outputs/test_X90_gate.npz')['dac0']
    plt.plot(np.roll(dac0_ref, EXTRA_DELAY))
    plt.plot(driver.dac_out[0])
    plt.show()
    #np.savez('test_outputs/test_X90_gate.npz', dac0=driver.dac_out[0])
    assert np.all(np.roll(dac0_ref, EXTRA_DELAY) == driver.dac_out[0])

@cocotb.test()
async def test_X90_gate_dc(dut):
    channel_config = hw.load_channel_configs('channel_config.json')
    qchip = qc.QChip('qubitcfg.json')
    prog = [{'name': 'X90', 'qubit': 'Q0', 'modi': {(0, 'dest'): 'C0.cdrv'}},
            {'name': 'pulse', 'freq': None, 'phase': 0, 'twidth': 32.e-9,
             'amp': -0.5, 'env': None, 'dest': 'C0.dc'},
            {'name': 'pulse', 'freq': None, 'phase': 0, 'twidth': 32.e-9,
             'amp': 0, 'env': None, 'dest': 'C0.dc'}]
    compiled_prog = tc.run_compile_stage(prog, fpga_config=hw.FPGAConfig(), qchip=qchip,
                                         proc_grouping=[('{qubit}.qdrv', '{qubit}.rdrv', '{qubit}.rdlo'),
                                                        ('{qubit}.qdrv2',),
                                                        ('{qubit}.cdrv', '{qubit}.dc')])
    binary = tc.run_assemble_stage(compiled_prog, channel_config)
    
    cocotb.start_soon(dsp.generate_clock(dut))
    driver = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    await driver.flush_cmd_mem()
    await driver.load_asm_program(binary)

    await driver.run_program(500)

    plt.plot(driver.dac_out[3])
    #np.savez('test_outputs/test_X90_plus_dc.npz', dac0=driver.dac_out[3])
    dac0_ref = np.load('test_outputs/test_X90_plus_dc.npz')['dac0']
    plt.plot(np.roll(dac0_ref, 0))
    plt.show()
    assert np.all(np.roll(dac0_ref,0) == driver.dac_out[3])

@cocotb.test()
async def test_simul_pulse(dut):
    channel_config = hw.load_channel_configs('channel_config.json')
    qchip = qc.QChip('qubitcfg.json')
    prog = [{'name': 'X90', 'qubit': 'Q0'},
            {'name': 'pulse', 'freq': 100.e6, 'phase': 0, 'twidth': 50.e-9,
             'amp': 0.5, 'env': {'env_func': 'cos_edge_square', 'paradict': {'ramp_fraction': 0.1}}
             , 'dest': 'Q0.qdrv2'}]
    #ipdb.set_trace()
    compiled_prog = tc.run_compile_stage(prog, fpga_config=hw.FPGAConfig(), qchip=qchip,
                                         proc_grouping=[('{qubit}.qdrv', '{qubit}.rdrv', '{qubit}.rdlo'),
                                                        ('{qubit}.qdrv2',),
                                                        ('{qubit}.cdrv', '{qubit}.dc')])
    binary = tc.run_assemble_stage(compiled_prog, channel_config)
    
    cocotb.start_soon(dsp.generate_clock(dut))
    driver = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    await driver.flush_cmd_mem()
    await driver.load_asm_program(binary)

    await driver.run_program(500)

    #np.savez('test_outputs/test_simul_pulse.npz', dac0=driver.dac_out[0])
    dac0_ref = np.load('test_outputs/test_simul_pulse.npz')['dac0']
    plt.plot(driver.dac_out[0])
    plt.plot(np.roll(dac0_ref, 0))
    plt.show()
    assert np.all(np.roll(dac0_ref, 0) == driver.dac_out[0])

@cocotb.test()
async def test_two_X90_gate(dut):
    channel_config = hw.load_channel_configs('channel_config.json')
    qchip = qc.QChip('qubitcfg.json')
    prog = [{'name': 'X90', 'qubit': 'Q0'},
            {'name': 'X90', 'qubit': 'Q0'}]
    compiled_prog = tc.run_compile_stage(prog, fpga_config=hw.FPGAConfig(), qchip=qchip)
    binary = tc.run_assemble_stage(compiled_prog, channel_config)
    
    cocotb.start_soon(dsp.generate_clock(dut))
    driver = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    await driver.flush_cmd_mem()
    await driver.load_asm_program(binary)

    await driver.run_program(500)

    plt.plot(driver.dac_out[0])
    #np.savez('test_outputs/test_two_X90_gate.npz', dac0=driver.dac_out[0])
    dac0_ref = np.load('test_outputs/test_two_X90_gate.npz')['dac0']
    plt.plot(np.roll(dac0_ref, 32))
    plt.show()
    assert np.all(np.roll(dac0_ref, 32) == driver.dac_out[0])

@cocotb.test()
async def test_two_rdrv(dut):
    channel_config = hw.load_channel_configs('channel_config.json')
    qchip = qc.QChip('qubitcfg.json')
    prog = [{'name': 'X90', 'qubit': 'Q0'},
            {'name': 'barrier', 'qubit': 'Q0'},
            {'name': 'pulse', 'dest': 'Q0.rdrv', 'twidth': 100.e-9, 'amp': 0.5,
             'freq': 5.127e9, 'phase': 0, 'env': np.ones(50)},
            {'name': 'pulse', 'dest': 'Q2.rdrv', 'twidth': 100.e-9, 'amp': 0.5,
             'freq': 6.227e9, 'phase': 0, 'env': {'env_func': 'cos_edge_square', 
                                                  'paradict': {'ramp_fraction': 0.25}}}]
            
    #ipdb.set_trace()
    compiled_prog = tc.run_compile_stage(prog, fpga_config=hw.FPGAConfig(), qchip=qchip)
    binary = tc.run_assemble_stage(compiled_prog, channel_config)
    print(compiled_prog)
    
    cocotb.start_soon(dsp.generate_clock(dut))
    driver = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    await driver.flush_cmd_mem()
    await driver.load_asm_program(binary)

    await driver.run_program(500)

    test_ref = np.load('test_outputs/test_two_rdrv.npz')
    #plt.plot(driver.dac_out[0])
    plt.plot(driver.dac_out[7])
    plt.plot(test_ref['dac_rdrv'])
    plt.show()
    #np.savez('test_outputs/test_two_rdrv.npz', dac0=driver.dac_out[0], dac_rdrv=driver.dac_out[3])
    assert np.all(test_ref['dac_rdrv'] == driver.dac_out[7])

@cocotb.test()
async def test_hw_virtualz(dut):
    channel_config = hw.load_channel_configs('channel_config.json')
    qchip = qc.QChip('qubitcfg.json')
    prog = [{'name': 'declare', 'var': 'phase0', 'dtype': 'phase', 'scope': ['Q0']},
            {'name': 'bind_phase', 'var': 'phase0', 'qubit': 'Q0'},
            {'name': 'X90', 'qubit': 'Q0'},
            {'name': 'virtual_z', 'qubit': 'Q0', 'phase': np.pi},
            {'name': 'X90', 'qubit': 'Q0'}]
    compiled_prog = tc.run_compile_stage(prog, fpga_config=hw.FPGAConfig(), qchip=qchip)
    binary = tc.run_assemble_stage(compiled_prog, channel_config)
    
    cocotb.start_soon(dsp.generate_clock(dut))
    driver = dsp.DSPDriver(dut, memory_map, 16, 16, 4, 16)
    await driver.flush_cmd_mem()
    await driver.load_asm_program(binary)

    await driver.run_program(500)

    dac0_ref = np.load('test_outputs/test_hw_virtualz.npz')['dac0']
    plt.plot(driver.dac_out[0])
    plt.plot(np.roll(dac0_ref, EXTRA_DELAY))
    plt.show()
    #np.savez('test_outputs/test_hw_virtualz.npz', dac0=driver.dac_out[0])
    assert np.all(np.roll(dac0_ref, EXTRA_DELAY) == driver.dac_out[0])
    
