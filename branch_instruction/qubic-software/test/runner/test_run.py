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
import logging

def test_run_circuit_batch():
    logging.getLogger(__name__).setLevel(logging.INFO)
    runner = CircuitRunner(SimInterface())
    executable = Executable()
    executable.add_mem_buffer('command0', data=bytes(160))
    executable.add_mem_buffer('command1', data=np.arange(50).tobytes())
    executable.add_mem_buffer('command3', data=np.arange(50).tobytes())
    executable.add_result_chan('Q0.rdlo', mem_name='accmem0', reads_per_shot=2)
    executable.add_result_chan('Q1.rdlo', mem_name='accmem0', reads_per_shot=1)
    s11 = runner.run_circuit_batch([executable], 500)[0]

    s11_ideal = {'Q0.rdlo': (np.arange(1000) - 500) + 1j*(np.arange(1000, 0, -1) - 500),
                 'Q1.rdlo': (np.arange(500) - 250) + 1j*(np.arange(500, 0, -1) - 250)}
    s11_ideal['Q0.rdlo'] = np.reshape(s11_ideal['Q0.rdlo'], (500, 2))
    s11_ideal['Q1.rdlo'] = np.reshape(s11_ideal['Q1.rdlo'], (500, 1))
    for chan in s11_ideal.keys():
        assert np.array_equal(s11[chan], s11_ideal[chan], equal_nan=True)

def test_run_circuit_batch_packed():
    logging.getLogger(__name__).setLevel(logging.INFO)
    runner = CircuitRunner(SimInterface())
    executable = Executable()
    executable.add_mem_buffer('command0', data=bytes(160))
    executable.add_mem_buffer('command1', data=np.arange(50).tobytes())
    executable.add_mem_buffer('command3', data=np.arange(50).tobytes())
    executable.add_result_chan('Q0.rdlo', mem_name='accmem0', reads_per_shot=2)
    executable.add_result_chan('Q1.rdlo', mem_name='accmem0', reads_per_shot=1)
    s11 = runner.run_circuit_batch([executable], 500)
    s11 = pack_s11_results(s11)

    s11_ideal = {'Q0.rdlo': (np.arange(1000) - 500) + 1j*(np.arange(1000, 0, -1) - 500),
                 'Q1.rdlo': (np.arange(500) - 250) + 1j*(np.arange(500, 0, -1) - 250)}
    s11_ideal['Q0.rdlo'] = s11_ideal['Q0.rdlo'].reshape((1, 500, 2))
    s11_ideal['Q1.rdlo'] = s11_ideal['Q1.rdlo'].reshape((1, 500, 1))
    for chan in s11_ideal.keys():
        assert np.array_equal(s11[chan], s11_ideal[chan], equal_nan=True)

def test_run_circuit_batch_rpc():
    logging.getLogger(__name__).setLevel(logging.INFO)
    port = 8905
    server_config = ServerConfig(ip='localhost',
                                 xsa_commit='',
                                 port=port,
                                 log_level='info',
                                 jobserver_mode=False,
                                 platform='sim')

    proc = mp.Process(target=run_soc_server, args=(server_config,))
    proc.start()

    try:
        time.sleep(10)
        runner = CircuitRunnerClient('localhost', port)
        
        executable = Executable()
        executable.add_mem_buffer('command0', data=bytes(160))
        executable.add_mem_buffer('command1', data=np.arange(50).tobytes())
        executable.add_mem_buffer('command3', data=np.arange(50).tobytes())
        executable.add_result_chan('Q0.rdlo', mem_name='accmem0', reads_per_shot=2)
        executable.add_result_chan('Q1.rdlo', mem_name='accmem0', reads_per_shot=1)
        #ipdb.set_trace()
        s11 = runner.run_circuit_batch([executable], 500)[0]

        s11_ideal = {'Q0.rdlo': (np.arange(1000) - 500) + 1j*(np.arange(1000, 0, -1) - 500),
                     'Q1.rdlo': (np.arange(500) - 250) + 1j*(np.arange(500, 0, -1) - 250)}
        print(s11_ideal.keys())
        s11_ideal['Q0.rdlo'] = s11_ideal['Q0.rdlo'].reshape((500, 2))
        s11_ideal['Q1.rdlo'] = s11_ideal['Q1.rdlo'].reshape((500, 1))
        for chan in s11_ideal.keys():
            assert np.array_equal(s11[chan], S11(s11_ideal[chan]), equal_nan=True)

    except Exception as e:
        proc.terminate()
        raise e

    proc.terminate()

def test_multiboard_rpc():
    logging.getLogger(__name__).setLevel(logging.INFO)
    soc_port0 = 8905
    soc_port1 = 8906
    jobserver_port = 8907
    server_config0 = ServerConfig(ip='localhost',
                                 xsa_commit='',
                                 port=soc_port0,
                                 log_level='debug',
                                 jobserver_mode=True,
                                 platform='sim')

    server_config1 = ServerConfig(ip='localhost',
                                 xsa_commit='',
                                 port=soc_port1,
                                 log_level='debug',
                                 jobserver_mode=True,
                                 platform='sim')

    jobserver_config = JobServerConfig(host_ip='localhost', host_port=jobserver_port, log_level='debug',
                                       boards={
                                           'board0': {
                                               'ip': 'localhost',
                                               'port': soc_port0,},
                                           'board1': {
                                               'ip': 'localhost',
                                               'port': soc_port1}
                                           },
                                       sync_graph={
                                           'board0.mst': 'board1.slv'})


    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
    server_procs = []
    proc = mp.Process(target=run_soc_server, args=(server_config0,))
    proc.start()
    server_procs.append(proc)

    proc = mp.Process(target=run_soc_server, args=(server_config1,))
    proc.start()
    server_procs.append(proc)

    time.sleep(10)

    proc = mp.Process(target=run_job_server, args=(jobserver_config,))
    proc.start()
    server_procs.append(proc)

    executable = Executable()
    executable.add_mem_buffer('command0', board='board0', data=bytes(160))
    executable.add_mem_buffer('command1', board='board1', data=np.arange(50).tobytes())
    executable.add_mem_buffer('command3', board='board0', data=np.arange(50).tobytes())
    executable.add_result_chan('Q0.rdlo', board='board0', mem_name='accmem0', reads_per_shot=2)
    executable.add_result_chan('Q1.rdlo', board='board1', mem_name='accmem0', reads_per_shot=1)

    try:
        time.sleep(10)
        runner = CircuitRunnerClient('localhost', jobserver_port)
        
        executable = Executable()
        executable.add_mem_buffer('command0', board='board0', data=bytes(160))
        executable.add_mem_buffer('command1', board='board1', data=np.arange(50).tobytes())
        executable.add_mem_buffer('command3', board='board0', data=np.arange(50).tobytes())
        executable.add_result_chan('Q0.rdlo', board='board0', mem_name='accmem0', reads_per_shot=2)
        executable.add_result_chan('Q1.rdlo', board='board1', mem_name='accmem0', reads_per_shot=1)
        s11 = runner.run_circuit_batch([executable], 500)[0]

        s11_ideal = {'Q0.rdlo': (np.arange(1000) - 500) + 1j*(np.arange(1000, 0, -1) - 500),
                     'Q1.rdlo': (np.arange(500) - 250) + 1j*(np.arange(500, 0, -1) - 250)}
        print(s11_ideal.keys())
        s11_ideal['Q0.rdlo'] = s11_ideal['Q0.rdlo'].reshape((500, 2))
        s11_ideal['Q1.rdlo'] = s11_ideal['Q1.rdlo'].reshape((500, 1))
        for chan in s11_ideal.keys():
            assert np.array_equal(s11[chan], S11(s11_ideal[chan]), equal_nan=True)

    except Exception as e:
        for proc in server_procs:
            proc.terminate()
        raise e

    for proc in server_procs:
        proc.terminate()

def test_multiboard_ring():
    logging.getLogger(__name__).setLevel(logging.INFO)
    soc_port0 = 8905
    soc_port1 = 8906
    soc_port2 = 8907
    jobserver_port = 8908
    server_config0 = ServerConfig(ip='localhost',
                                  xsa_commit='',
                                  port=soc_port0,
                                  log_level='debug',
                                  jobserver_mode=True,
                                  platform='sim')

    server_config1 = ServerConfig(ip='localhost',
                                  xsa_commit='',
                                  port=soc_port1,
                                  log_level='debug',
                                  jobserver_mode=True,
                                  platform='sim')

    server_config2 = ServerConfig(ip='localhost',
                                  xsa_commit='',
                                  port=soc_port2,
                                  log_level='debug',
                                  jobserver_mode=True,
                                  platform='sim')

    jobserver_config = JobServerConfig(host_ip='localhost', host_port=jobserver_port, log_level='debug',
                                       boards={
                                           'board0': {
                                               'ip': 'localhost',
                                               'port': soc_port0,},
                                           'board1': {
                                               'ip': 'localhost',
                                               'port': soc_port1},
                                           'board2': {
                                               'ip': 'localhost',
                                               'port': soc_port2}
                                           },
                                       sync_graph={
                                           'board0.mst': 'board1.slv',
                                           'board1.mst': 'board2.slv',
                                           'board2.mst': 'board0.slv'})


    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
    server_procs = []
    proc = mp.Process(target=run_soc_server, args=(server_config0,))
    proc.start()
    server_procs.append(proc)

    proc = mp.Process(target=run_soc_server, args=(server_config1,))
    proc.start()
    server_procs.append(proc)

    proc = mp.Process(target=run_soc_server, args=(server_config2,))
    proc.start()
    server_procs.append(proc)

    time.sleep(10)

    proc = mp.Process(target=run_job_server, args=(jobserver_config,))
    proc.start()
    server_procs.append(proc)

    try:
        time.sleep(10)
        runner = CircuitRunnerClient('localhost', jobserver_port)
        
        executable = Executable()
        executable.add_mem_buffer('command0', board='board0', data=bytes(160))
        executable.add_mem_buffer('command1', board='board1', data=np.arange(50).tobytes())
        executable.add_mem_buffer('command3', board='board0', data=np.arange(50).tobytes())
        executable.add_mem_buffer('command3', board='board2', data=np.arange(50).tobytes())
        executable.add_result_chan('Q0.rdlo', board='board0', mem_name='accmem0', reads_per_shot=2)
        executable.add_result_chan('Q1.rdlo', board='board1', mem_name='accmem0', reads_per_shot=1)
        s11 = runner.run_circuit_batch([executable], 500)[0]

        s11_ideal = {'Q0.rdlo': (np.arange(1000) - 500) + 1j*(np.arange(1000, 0, -1) - 500),
                     'Q1.rdlo': (np.arange(500) - 250) + 1j*(np.arange(500, 0, -1) - 250)}
        print(s11_ideal.keys())
        s11_ideal['Q0.rdlo'] = s11_ideal['Q0.rdlo'].reshape((500, 2))
        s11_ideal['Q1.rdlo'] = s11_ideal['Q1.rdlo'].reshape((500, 1))
        for chan in s11_ideal.keys():
            assert np.array_equal(s11[chan], S11(s11_ideal[chan]), equal_nan=True)

    except Exception as e:
        for proc in server_procs:
            proc.terminate()
        raise e

    for proc in server_procs:
        proc.terminate()
