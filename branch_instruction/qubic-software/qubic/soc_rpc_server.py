import warnings
try:
    import qubic.rfsoc.pl_interface as pl
except ImportError:
    warnings.warn('could not import pl_interface, run in sim mode only')
from qubic.sim.sim_interface import SimInterface
from qubic.rfsoc.sync_interface import SyncInteface
import xmlrpc.server
from qubic.run import CircuitRunner
import logging
import argparse
from functools import partial, update_wrapper
from attrs import define, field
import yaml
from typing import List, Dict, Tuple

@define
class ServerConfig:
    xsa_commit: str
    ip: str
    port: int
    log_level: str
    jobserver_mode: bool = False
    dac_current_values: Dict = field(default=None)
    dac_nyquist_zone: Dict = field(default=None)
    adc_nyquist_zone: int = 1
    lmk_freq: float = 500.18
    platform: str = 'rfsoc'

    """
    Class for configuring/initializing ZCU216 RPC server. Wraps
    server_config.yaml

    Attributes
    ----------
    xsa_commit: str
        commit hash of gateware build (directory w/ XSA/configs)
    ip: str
        server IP
    port: int
        server port
    log_level: str
    enable_batching: bool 
        if True, allow clients to submit batched jobs, otherwise
        an intermediate job server (job_rpc_server.py) is 
        required.
    dac_current_values: Dict[Tuple, int]
        dictionary of DAC channels to modify current. Format 
        is (block, tile): <current in uA>
    dac_nyquist_zone: int
    adc_nyquist_zone: int
    lmk_freq: float
    """

    def __attrs_post_init__(self):
        if self.dac_current_values is not None:
            self.dac_current_values = {
                    tuple(int(i) for i in key.split(',')): value for key, value in self.dac_current_values.items()}
        if self.dac_nyquist_zone is not None:
            dac_nyquist_zone_dict = {}
            for key, value in self.dac_nyquist_zone.items():
                if key.lower() == 'default':
                    dac_nyquist_zone_dict['default'] = value
                else:
                    dac_nyquist_zone_dict[tuple(int(i) for i in key.split(','))] = value

            self.dac_nyquist_zone = dac_nyquist_zone_dict

def run_soc_server(server_config: ServerConfig):
    """
    Start an xmlrpc server that exposes an instance of CircuitRunner over a 
    network. Intended to be run from the RFSoC ARM core python (pynq) 
    environment. IP should only be accessible locally!

    Parameters
    ----------
    server_config: ServerConfig
    """
    if server_config.log_level is not None:
        if server_config.log_level.lower() == 'info':
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

        elif server_config.log_level == 'debug':
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

        else:
            raise Exception(f'log level: {server_config.log_level} not supported')

    if server_config.platform == 'rfsoc':
        pl_driver = pl.PLInterface(commit_hash=server_config.xsa_commit,
                                   dac_current_values=server_config.dac_current_values,
                                   dac_nyquist_zone=server_config.dac_nyquist_zone,
                                   adc_nyquist_zone=server_config.adc_nyquist_zone,
                                   lmk_freq=server_config.lmk_freq,
                                   load_xsa=True)

    elif server_config.platform == 'rfsoc_multiboard':
        pl_driver = pl.PLInterfaceSync(commit_hash=server_config.xsa_commit,
                                       dac_current_values=server_config.dac_current_values,
                                       dac_nyquist_zone=server_config.dac_nyquist_zone,
                                       adc_nyquist_zone=server_config.adc_nyquist_zone,
                                       lmk_freq=server_config.lmk_freq,
                                       load_xsa=True)

    elif server_config.platform == 'sim':
        pl_driver = SimInterface()

    else:
        raise Exception(f'unsupported platform: {server_config.platform}')

    runner = CircuitRunner(pl_driver)
    server = xmlrpc.server.SimpleXMLRPCServer((server_config.ip, server_config.port), 
                                              logRequests=True, allow_none=True)


    if server_config.jobserver_mode:
        sync_interface = SyncInteface(pl_driver)
        server.register_function(runner.load_executable)
        server.register_function(runner.start_program)
        server.register_function(partial(runner.run_circuit, from_server=True), name='run_circuit')
        server.register_function(partial(runner.wait_and_readback, from_server=True), name='wait_and_readback')
        server.register_function(partial(runner.load_and_run_acq, from_server=True), name='load_and_run_acq')
        server.register_function(sync_interface.read_ptp_corrval)
        server.register_function(sync_interface.write_ptp_corrval)
        server.register_function(sync_interface.ptp_enable_tx)
        server.register_function(sync_interface.ptp_disable_tx)
        server.register_function(sync_interface.ptp_read_tx_clockcount)
        server.register_function(sync_interface.ptp_read_rx_clockcount)
        server.register_function(sync_interface.ptp_read_dsp_clockcount)


    else:
        server.register_function(partial(runner.run_circuit_batch, from_server=True), name='run_circuit_batch')
        server.register_function(partial(runner.load_and_run_acq, from_server=True), name='load_and_run_acq')

    print('RPC server running on {}:{}'.format(server_config.ip, server_config.port))

    server.serve_forever()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('server_config')
    args = parser.parse_args()

    with open(args.server_config) as f:
        server_config = ServerConfig(**yaml.safe_load(f))
 
    print(f'starting RPC Server on {server_config.ip}:{server_config.port}')

    run_soc_server(server_config)


