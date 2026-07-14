from qubic.run import CircuitRunner
from qubic.rpc_client import CircuitRunnerClient
from distproc.executable import Executable
import numpy as np
import ipdb
import time
import multiprocessing as mp
from qubic.soc_rpc_server import ServerConfig, run_soc_server
from qubic.job_rpc_server import ServerConfig as JobServerConfig
from qubic.job_rpc_server import run_job_server
from qubic.sim.sim_interface import SimInterface
from qubic.results.primitives import S11
from qubic.results.tools import pack_s11_results
from distproc.hwconfig import load_channel_configs, FPGAConfig
from distproc.ir.instructions import *
import qubic.job_manager as jm
import logging

def test_build_and_run_shots():
    """
    can't do CircuitCounts here since reads per shot is different between qubits
    """
    logging.getLogger(__name__).setLevel(logging.INFO)
    runner = CircuitRunner(SimInterface())

    circuit = [
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q0.qdrv'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q0.rdrv'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q0.rdlo'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q1.rdlo'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q1.rdlo')
        ]

    circuit2 = [
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q0.qdrv'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q1.qdrv'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q0.rdrv'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q0.rdlo'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q1.rdlo'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q1.rdlo')
        ]

    circuits = [circuit, circuit2]

    fpga_config = FPGAConfig()
    channel_configs = load_channel_configs('channel_config.json')

    jobman = jm.JobManager(fpga_config, channel_configs, runner)
    res = jobman.build_and_run_circuits(circuits, n_total_shots=500, reads_per_shot={'Q0.rdlo': 1, 'Q1.rdlo': 2}, 
                                        outputs=['s11', 'shots'], compiler_flags={'resolve_gates': False}, fit_gmm=True)

    print(f's11 Q0 shape: {res["s11"][0]["Q0.rdlo"].shape}')
    print(f's11 Q1 shape: {res["s11"][0]["Q1.rdlo"].shape}')

def test_build_and_run_all():
    logging.getLogger(__name__).setLevel(logging.INFO)
    runner = CircuitRunner(SimInterface())

    circuit = [
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q0.qdrv'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q0.rdrv'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q0.rdlo'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q1.rdlo'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q1.rdlo')
        ]

    circuit2 = [
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q0.qdrv'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q1.qdrv'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q0.rdrv'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q0.rdlo'),
            Pulse(freq=5.e9, amp=0.9, phase=0, env={'env_func': 'square', 'paradict': {}},
                  twidth=100.e-9, dest='Q1.rdlo'),
        ]

    circuits = [circuit, circuit2]

    fpga_config = FPGAConfig()
    channel_configs = load_channel_configs('channel_config.json')

    jobman = jm.JobManager(fpga_config, channel_configs, runner)
    res = jobman.build_and_run_circuits(circuits, n_total_shots=500, reads_per_shot={'Q0.rdlo': 1, 'Q1.rdlo': 1}, 
                                        outputs=['s11', 'shots', 'counts'], compiler_flags={'resolve_gates': False}, fit_gmm=True)

    print(res['counts'])
