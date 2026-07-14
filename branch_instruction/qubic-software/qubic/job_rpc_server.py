import xmlrpc.server
import logging
import argparse
import yaml
import time
import concurrent.futures as cf
import networkx as nx
from attrs import define, field
from typing import List, Dict, Iterable, Iterator
from distproc.executable import Executable, executable_from_dict
import numpy as np

ACC_BUF_SIZE = 1000

@define
class ServerConfig:
    """
    Wrapper around jobserver_config.yaml (example in `scripts/') 

    Attributes
    ----------
    host_ip: str
        host IP of the job server
    host_port: int
        host port of the job server
    boards: Dict[str, Dict]
        Boards controlled by this server. Each board is keyed
        by its name (as defined in the channel_config file); attributes
        are 'master'/'slave', IP, and port
    log_level: str
        'info' (default) or 'debug'
    """
    host_ip: str
    host_port: int
    boards: Dict[str, Dict[str, Dict]]
    sync_graph: Dict[str, str]
    log_level: str = 'info'

@define
class _PTPEdge:
    master: str
    slave: str
    master_interface: str
    slave_interface: str


class _SyncGraph:
    """
    Maintains a NetworkX DiGraph that encodes the topology of a multi-board synchronized 
    system. Directed edges specify a master/slave relationship between two board interfaces.

    Attributes
    ----------
    graph: nx.DiGraph
        nodes: strings specifying board names
        edges: _PTPEdge objects which specify the connected boards,
            as well as the named interfaces on each board that implement these connections.
            
    """

    def __init__(self, board_names: Iterable[str], graph_edges: Dict[str, str]):
        self.graph = nx.DiGraph()
        for board in board_names:
            self.graph.add_node(board, cur_corrval=0)

        for master, slave in graph_edges.items():
            master_board, master_interface = master.split('.')
            slave_board, slave_interface = slave.split('.')
            if master_board not in self.graph.nodes:
                raise Exception(f'{master_board} not found!')
            if slave_board not in self.graph.nodes:
                raise Exception(f'{slave_board} not found!')
            self.graph.add_edge(master_board, slave_board, ptp_edge=_PTPEdge(master=master_board, slave=slave_board, 
                                                        master_interface=master_interface, slave_interface=slave_interface))

        if not nx.is_weakly_connected(self.graph):
            raise Exception(f'PTP Sync graph is not connected: \n'\
                            f'    nodes: {self.graph.nodes} \n'\
                            f'    edges: {self.graph.edges}')

        # check for any root nodes, otherwise just start with the first board
        zero_pred_list = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        if len(zero_pred_list) > 0:
            self.start_board = zero_pred_list[0]
        else:
            self.start_board = list(board_names)[0]

    def ptp_edge_iterator(self) -> Iterator[_PTPEdge]:
        return (self.graph.edges[edge]['ptp_edge'] for edge in nx.edge_bfs(self.graph, source=self.start_board, orientation=None))

    def get_board_corrval(self, board: str) -> int:
        return self.graph.nodes[board]['cur_corrval']

    def set_board_corrval(self, board: str, value: int):
        self.graph.nodes[board]['cur_corrval'] = value


class _BoardClient:

    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        # self._busy = False

    def connect(self):
        self.proxy = xmlrpc.client.ServerProxy('http://' + self.ip + ':' + str(self.port), allow_none=True)

class JobServer:
    """
    Class for managing a multi-board job server with synchronization. General flow is:

        1. User submits job (list of compiled executables) to this server via RPC using 
           `run_circuit_batch`
        2. Server initiates PTP offset correction to synchronize boards
        3. For each circuit:
            a. upload partial executable to each board
            b. for each batch of 1000 shots:
                1) initiate timed trigger on each board
                2) wait for results
            c. concatenate all results
        4. Return batched results to user, in same format as the normal CircuitRunner

    In principle, can support a single master board with arbitrary number of slaves; although 
    only 2-board configurations have been tested.

    """

    def __init__(self, board_configs: Dict[str, Dict], sync_graph: Dict[str, str]):
        """
        Parameters
        ----------
        board_configs: Dict[str, Dict]
            dictionary of boards, keyed by board name 
            values are Dicts, with fields: mode (master or slave),
                ip, port 
        """
        self._boards = {}
        self._loaded_boards = []
        self._result_channels = {}
        for boardname, config in board_configs.items():
            self._boards[boardname] =  _BoardClient(**config)
        self._sync_graph = _SyncGraph(self._boards.keys(), sync_graph)
        self._init_ptp()

    def _init_ptp(self):
        for boardname in self._boards.keys():
            self._boards[boardname].connect()

        print(list(self._sync_graph.graph.edges))
        print(list(self._sync_graph.graph.nodes))
        print(list(self._sync_graph.ptp_edge_iterator()))
        for ptp_pair in self._sync_graph.ptp_edge_iterator():
            logging.getLogger(__name__).debug(f'doing PTP correction for boards {ptp_pair.master}, {ptp_pair.slave}')
            delay, offset = self._measure_ptp_corr(ptp_pair)
            slave_corrval = int(self._boards[ptp_pair.slave].proxy.read_ptp_corrval())
            slave_corrval = int(slave_corrval-2**64) if (slave_corrval>>(64-1))&1 else slave_corrval
            logging.getLogger(__name__).debug(f'    slave_corval: {slave_corrval}')
            # self.master_board.proxy.write_ptp_corrval('0')
            
            # where to add master offset?
            correction = slave_corrval - offset + self._sync_graph.get_board_corrval(ptp_pair.master)
            logging.getLogger(__name__).debug(f'    correction: {correction}') 
            self._sync_graph.set_board_corrval(ptp_pair.slave, correction)
            self._boards[ptp_pair.slave].proxy.write_ptp_corrval(str(int(slave_corrval - offset)))
            
    def _measure_ptp_corr(self, ptp_pair: _PTPEdge):
        master = self._boards[ptp_pair.master]
        slave = self._boards[ptp_pair.slave]
        slave.proxy.ptp_disable_tx(ptp_pair.slave_interface)
        master.proxy.ptp_enable_tx(ptp_pair.master_interface)
        t1 = int(master.proxy.ptp_read_tx_clockcount(ptp_pair.master_interface))
        t2 = int(slave.proxy.ptp_read_rx_clockcount(ptp_pair.slave_interface))
        master.proxy.ptp_disable_tx(ptp_pair.master_interface)
        slave.proxy.ptp_enable_tx(ptp_pair.slave_interface)
        t3 = int(slave.proxy.ptp_read_tx_clockcount(ptp_pair.slave_interface))
        t4 = int(master.proxy.ptp_read_rx_clockcount(ptp_pair.master_interface))
        delay = ((t4-t1)-(t3-t2))/2
        offset = ((t2-t1)-(t4-t3))/2

        logging.getLogger(__name__).debug(f'ptp times: {t1}, {t2}, {t3}, {t4}')
        logging.getLogger(__name__).debug(f'    offset: {offset}')
        logging.getLogger(__name__).debug(f'    delay: {delay}')

        return delay, offset

    def _load_executable(self, executable_dict: Dict, load_commands: bool = True, 
                         load_freqs: bool = True, load_envs: bool = True, zero: bool = True):
        """
        Load executable into all boards, returns when finished
        """

        executable = executable_from_dict(executable_dict)
        cur_boards = executable.boards
        futures = []
        with cf.ThreadPoolExecutor(max_workers=len(cur_boards)) as e:
            for board in cur_boards:
                futures.append(e.submit(self._boards[board].proxy.load_executable, executable.get_board_executable(board).to_dict(), 
                               load_commands, load_freqs, load_envs, zero))
 
        # this is to throw errors if any future has an error
        for future in futures:
            _ = future.result()
        self._loaded_boards = cur_boards
        self._result_channels = executable.result_channels

    def _start_program(self, nshots: int, clock_count: int = None):
        futures = []
        with cf.ThreadPoolExecutor(max_workers=len(self._loaded_boards)) as e:
            for board in self._loaded_boards:
                logging.getLogger(__name__).debug(f'starting program on board {board}')
                futures.append(e.submit(self._boards[board].proxy.start_program, nshots, str(clock_count)))

        # this is to throw errors if any future has an error
        for future in futures:
            _ = future.result()

    def _wait_and_readback(self, reads_per_shot: int | Dict = None):
        futures = {}
        with cf.ThreadPoolExecutor(max_workers=len(self._loaded_boards)) as e:
            for board in self._loaded_boards:
                logging.getLogger(__name__).debug(f'waiting for result on board {board}')
                futures[board] = e.submit(self._boards[board].proxy.wait_and_readback, reads_per_shot)

        results = {}
        print(futures.keys())
        for board in futures.keys():
            results.update(futures[board].result())
            logging.getLogger(__name__).debug(f'received result on board {board}')

        return results

    def _run_circuit(self, n_total_shots, reads_per_shot, timeout_per_shot=8, waitclkcnt=2**24):
        if isinstance(reads_per_shot, int):
            for _, chan in self._result_channels.items():
                chan.reads_per_shot = reads_per_shot
        elif isinstance(reads_per_shot, dict):
            for channame, n_reads in reads_per_shot.items():
                if channame in self._result_channels:
                    self._result_channels[channame].reads_per_shot = n_reads
        else:
            if reads_per_shot is not None:
                raise Exception(f'reads per shot: {reads_per_shot} invalid type')
 
        logging.getLogger(__name__).info(f'starting circuit with {n_total_shots} shots')

        if len(self._result_channels) == 0:
            shots_per_run = n_total_shots
        else:
            max_reads_per_shot = max(acc_chan.reads_per_shot for acc_chan in self._result_channels.values())
            shots_per_run = min(ACC_BUF_SIZE//max_reads_per_shot, n_total_shots)

        n_runs = int(np.ceil(n_total_shots/shots_per_run))
        s11 = {ch: bytes() for ch in self._result_channels}
                         
        for i in range(n_runs): 
            logging.getLogger(__name__).debug(f'reading master clockcount')
            dspclkcnt = int(self._boards[self._loaded_boards[0]].proxy.ptp_read_dsp_clockcount())
            self._start_program(shots_per_run if i < (n_runs - 1) else n_total_shots - i*shots_per_run, 
                                clock_count=str(dspclkcnt + waitclkcnt))
            time.sleep(waitclkcnt*2.e-9 + 0.01)
            result = self._wait_and_readback(reads_per_shot)
            for ch in self._result_channels.keys():
                s11[ch] += result[ch].data
 
        logging.getLogger(__name__).info('done circuit')
        return s11
    

    def run_circuit_batch(self, 
                          executables: List[Executable], 
                          n_total_shots: int, 
                          reads_per_shot: int | Dict[str, int] = None, 
                          timeout_per_shot: float = 8,
                          reload_cmd: bool = True, 
                          reload_freq: bool = True, 
                          reload_env: bool = True, 
                          zero_between_reload: bool = True) -> Dict:
        """
        Runs a batch of circuits given by a list of compiled executables. Each circuit is run n_total_shots
        times. `reads_per_shot` and `n_total_shots` are passed directly into run_circuit, and must
        be the same for all circuits in the batch. The parameters reload_cmd, reload_freq, reload_env, and 
        zero_between_reload control which of these fields is rewritten circuit-to-circuit (everything is 
        rewritten initially). Leave these all at True (default) for maximum safety, to ensure that QubiC 
        is in a clean state before each run. Depending on the circuits, some of these can be turned off 
        to save time.

        TODO: consider throwing some version of all the args here into a BatchedCircuitRun or somesuch
        object

        Parameters
        ----------
        executables : List[Executable]
            list of executables to run
        n_total_shots : int
            number of shots per circuit
        reads_per_shot : int | Dict[str, int]
            number of values per shot per channel to read back from accbuf. Unless
            there is mid-circuit measurement involved this is typically 1. If `int`, assumed
            to be the same for all channels, else can be a per-channel dict.
        timeout_per_shot : float
            maximum allowable wall clock time (in seconds) per single shot of the circuit
        reload_cmd : bool
            if True, reload command buffer between circuits
        reload_freq : bool
            if True, reload freq buffer between circuits
        reload_env: bool
            if True, reload env buffer between circuits

        Returns
        -------
        dict:
            Complex IQ shots for each accbuf in chanlist; each array has 
            shape (len(raw_asm_list), n_total_shots, reads_per_shot)
        """
        self._init_ptp()
        s11 = []

        for i, executable in enumerate(executables):
            logging.getLogger(__name__).info(f'starting circuit {i}/{len(executables)-1}')
            if i==0:
                self._load_executable(executable, True, True, True, True)
            else:
                self._load_executable(executable, reload_cmd, reload_freq, reload_env, zero_between_reload)

            if len(self._loaded_boards) == 1:
                s11_i = self._boards[self._loaded_boards[0]].proxy.run_circuit(n_total_shots, reads_per_shot, timeout_per_shot)

            else:
                s11_i = self._run_circuit(n_total_shots, reads_per_shot, timeout_per_shot)

            s11.append(s11_i)

        logging.getLogger(__name__).info('batch finished')
        return s11

    def load_and_run_acq(self, 
                         executable_dict: Dict, 
                         n_total_shots: int = 1, 
                         nsamples: int = 8192, 
                         acq_chans: Dict[str, int] = {'0': 0, '1': 1}, 
                         trig_delay: float = 0, 
                         decimator: int = 0, 
                         return_acc: bool = False):
        executable = executable_from_dict(executable_dict)
        if len(executable.boards) > 1:
            raise Exception('Multi-board mode not supported for ACQ!')
        self._init_ptp()
        return self._boards[executable.boards[0]].proxy.load_and_run_acq(executable_dict, n_total_shots, nsamples, acq_chans, trig_delay, decimator, return_acc)

def run_job_server(server_config):
    """
    Run the RPC job server using the provided job server config. Exposes `JobServer.run_circuit_batch` and 
    `JobServer.load_and_run_acq`. Boards are assumed to be running their own RPC servers using `qubic.soc_rpc_server`

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

    job_server = JobServer(server_config.boards, server_config.sync_graph)
    rpc_server = xmlrpc.server.SimpleXMLRPCServer((server_config.host_ip, server_config.host_port), logRequests=True, allow_none=True)
    rpc_server.register_function(job_server.run_circuit_batch)
    rpc_server.register_function(job_server.load_and_run_acq)

    logging.getLogger(__name__).info('RPC job server running on {}:{}'.format(server_config.host_ip, 
                                                                              server_config.host_port))
    rpc_server.serve_forever()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('server_config')
    args = parser.parse_args()

    with open(args.server_config) as f:
        server_config = ServerConfig(**yaml.safe_load(f))
        server_config.boards = {name: params for boarddict in server_config.boards for name, params in boarddict.items()}
        for board, params in server_config.boards.items():
            server_config.boards[board] = params


    run_job_server(server_config)

