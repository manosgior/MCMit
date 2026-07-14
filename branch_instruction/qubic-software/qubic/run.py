import warnings

from distproc.executable import Executable, executable_from_dict
from qubic.results.primitives import get_result_class
import numpy as np
from tqdm import tqdm
import logging
from qubic.abstract_runner import AbstractCircuitRunner
from typing import List, Dict

TIMEOUT = 600 #timeout per 1000 shots
POLL_INTERVAL = 0.05

# todo: load these from some sort of config tied to bitfile
CLK_PERIOD = 2.e-9
MAX_NSAMPLES = 16384
ADC_SAMPLES_PER_CLK = 4

class CircuitRunner(AbstractCircuitRunner):
    """
    Class for taking a program in binary/ASM form and running it on 
    the FPGA. Currently, this class is meant to be run on the QubiC FPGA 
    PS + pynq system. It will load and configure the specified PL bitfile,
    and can then be used to configure PL memory and registers, and read 
    back data from experiments.

    Attributes
    ----------
    _pl_driver: pl.PLInterface 
        used for low level access to memory and registers
    loaded_channels: list 
        channels with a program currently loaded
    """

    def __init__(self, pl_driver):
        self._pl_driver = pl_driver

        self.result_channels = []
        self._cur_nshots = None


    def load_and_run(self, executable: Executable, n_total_shots: int): 
        """
        Load circuit described by rawasm "binary", then run for n_total_shots. 

        Parameters
        ----------
        executable: Executable
        n_total_shots: int
            number of shots to run. Program is restarted from the beginning 
            for each new shot

        Returns
        -------
        dict:
            Complex IQ shots for each accbuf in chanlist; each array has 
            shape `(n_total_shots, reads_per_shot)`
        """
        self.load_executable(Executable)
        return self.run_circuit(n_total_shots)

    def load_executable(self, executable: Executable | dict, load_commands: bool = True, 
                        load_freqs: bool = True, load_envs: bool = True, zero: bool = True):
        """
        Load the circuit described by `executable`, which is the output of 
        the final distributed proc assembler stage. Loads command memory, env memory
        and freq buffer memory, according to specified input parameters. Before circuit is loaded, 
        if zero=True, all channels are zeroed out using zero_command_buf()

        Parameters
        ----------
        executable: Executable | dict
            Compiled program binary. Can be an Executable object or its dictionary
            representation (for xmlrpc support).
        zero: bool
            if True, (default), zero out all cmd buffers before loading circuit
        load_commands: bool
            if True, (default), load command buffers
        load_freqs: bool
            if True, (default), load freq buffers
        load_envs: bool
            if True, (default), load env buffers
        """
        if isinstance(executable, dict):
            executable = executable_from_dict(executable)

        if zero:
            self.zero_command_buf()

        self._cur_nshots = None
        for memname, data in executable.get_binaries_fromboard().items():
            if load_commands and 'command' in memname:
                self._pl_driver.write_mem_buf(memname, data)
            elif load_envs and 'env' in memname:
                self._pl_driver.write_mem_buf(memname, data)
            elif load_freqs and 'freq' in memname:
                self._pl_driver.write_mem_buf(memname, data)
            else:
                self._pl_driver.write_mem_buf(memname, data)

        for regname, value in executable.get_registers_fromboard().items():
            self._pl_driver.write_reg(regname, value)

        self.loaded_result_channels = executable.result_channels
    
    def start_program(self, nshots, clock_count=None):
        self._cur_nshots = nshots
        self._pl_driver.start_program(nshots, clock_count)

    def wait_and_readback(self, reads_per_shot=None, from_server=False) -> Dict[str, np.ndarray]:
        """
        Returns
        -------
        Dict[str, bytes]
            results for each channel
        """
        if isinstance(reads_per_shot, int):
            for _, chan in self.loaded_result_channels.items():
                chan.reads_per_shot = reads_per_shot
        elif isinstance(reads_per_shot, dict):
            for channame, n_reads in reads_per_shot.items():
                if channame in self.loaded_result_channels:
                    self.loaded_result_channels[channame].reads_per_shot = n_reads
        else:
            if reads_per_shot is not None:
                raise Exception(f'reads per shot: {reads_per_shot} invalid type')

        # s11 = {channame: np.nan*np.zeros((1, self._cur_nshots, chan.reads_per_shot), dtype=np.complex128) 
        #        for channame, chan in self.loaded_result_channels.items()}
        return self._pl_driver.wait_and_readback(self.loaded_result_channels, self._cur_nshots)
 
    def zero_command_buf(self, core_inds: List[str | int] = None):
        """
        Loads command memory with dummy asm program: reset phase, 
        output done signal, then idle. This is useful/necessary if 
        a new program is loaded on a subset of cores such that the 
        previous program is not completely overwritten (e.g. you 
        are loading a program that runs only on core 2, and the 
        previous program used cores 2 and 3).

        Parameters
        ----------
        core_inds: list
            list of channels (proc cores) to load. Defaults to
            all channels in currently loaded gateware.
        """
        zero_prog = Executable({':' + mem: bytes(16) for mem in self._pl_driver.get_program_memories(core_inds)}) 
        mems = zero_prog.get_binaries_fromboard()
        for memname, data in mems.items():
            self._pl_driver.write_mem_buf(memname, data)

    def run_circuit_batch(self, 
                          executables: List[Executable | Dict], 
                          n_total_shots: int, 
                          reads_per_shot: int = None,
                          timeout_per_shot: float = 8,
                          reload_cmd: bool = True, 
                          reload_freq: bool = True, 
                          reload_env: bool = True, 
                          zero_between_reload: bool = True,
                          from_server: bool = False):
        """
        Runs a batch of circuits given by a list of compiled executables. Each circuit is run n_total_shots
        times. `reads_per_shot` and `n_total_shots` are passed directly into `run_circuit`, and must
        be the same for all circuits in the batch. The parameters `reload_cmd`, `reload_freq`, `reload_env`, and 
        `zero_between_reload` control which of these fields is rewritten circuit-to-circuit (everything is 
        rewritten initially). Leave these all at `True` (default) for maximum safety, to ensure that QubiC 
        is in a clean state before each run. Depending on the circuits, some of these can be turned off 
        to save time.

        TODO: consider throwing some version of all the args here into a BatchedCircuitRun or somesuch
        object

        Parameters
        ----------
        executables: List[Executable, Dict]
            List of compiled program binaries. Each element can be an Executable object or its dictionary
            representation (for xmlrpc support).
        n_total_shots: int
            number of shots per circuit
        reads_per_shot: int | dict
            number of values per shot per channel to read back from accbuf. If dict, indexed
            by channel name (e.g. `Q0.rdlo`). If int, assumed to be the same across channels. 
            Unless multiple circuits were rastered pre-compilation or there is mid-circuit 
            measurement involved this is typically 1
        timeout_per_shot: float
            job will time out if time to take a single shot exceeds this value in seconds 
            (this likely means the job is hanging due to timing issues in the program or gateware)
        reload_cmd: bool
            if True, reload command buffer between circuits
        reload_freq: bool
            if True, reload freq buffer between circuits
        reload_env: bool
            if True, reload env buffer between circuits
        from_server: bool
            set to true if calling over RPC. If True, pack returned s11 arrays into
            byte objects
        Returns
        -------
        dict:
            Complex IQ shots for each accbuf in chanlist; each array has 
            shape `(len(executables), n_total_shots, reads_per_shot)`
        """
        results = []

        for i, raw_asm in enumerate(tqdm(executables)):
            logging.getLogger(__name__).info(f'starting circuit {i}/{len(executables)-1}')
            if i==0:
                self.load_executable(raw_asm, True, True, True, True)
                self._pl_driver.set_default_regs()
            else:
                self.load_executable(raw_asm, zero=zero_between_reload, load_commands=reload_cmd,
                                     load_freqs=reload_freq, load_envs=reload_env)
            
            results_i = self.run_circuit(n_total_shots, reads_per_shot, timeout_per_shot, from_server)
            results.append(results_i)

        logging.getLogger(__name__).info('batch finished')
        return results

    def load_and_run_acq(self, 
                         raw_asm_prog: Executable, 
                         n_total_shots: int = 1, 
                         nsamples: int = 8192, 
                         acq_chans: Dict[str, int] = {'0':0,'1':1}, 
                         trig_delay: float = 0, 
                         decimator: int = 0, 
                         return_acc: bool = False, 
                         from_server: bool = False):
        """
        Load the program given by `raw_asm_prog` and acquire raw (or downconverted) adc traces.

        Parameters
        ----------
        raw_asm_prog: Executable | Dict
            Compiled program binary to run. See `load_executable` for details.
        n_total_shots: int
            number of shots to run. Program is restarted from the beginning 
            for each new shot
        nsamples: int
            number of samples to read from the acq buffer
        acq_chans: dict
            current channel mapping is:

                '0': ADC_237_2 (main readout ADC)
                '1': ADC_237_0 (other ADC connected in gateware)
                TODO: figure out DLO channels, etc and what they mean
        trig_delay: float
            time to delay acquisition, relative to circuit start.
            NOTE: this value, when converted to units of clock cycles, is a 
            16-bit value. So, it maxes out at CLK_PERIOD*(2**16) = 131.072e-6
        decimator: int
            decimation interval when sampling. e.g. 0 means full sample rate, 1
            means capture every other sample, 2 means capture every third sample, etc
        return_acc: bool
            if True, return a single acc (integrated + accumulated readout) value per shot,
            on each loaded channel. Default is False.
        from_server: bool
            set to true if calling over RPC. If True, pack returned acq arrays into
            byte objects

        Returns
        -------
        tuple | Dict
            - if `return_acc` is `False`:

                - dict:
                    array of acq samples for each channel in acq_chans with shape (n_total_shots, nsamples)

            - if `return_acc` is `True`:

                - tuple:
                    - dict:
                        array of acq samples for each channel in acq_chans with shape `(n_total_shots, nsamples)`
                    - dict:
                        array of acc values for each loaded channel with length `n_total_shots`

        """
        self.load_executable(raw_asm_prog)
        return self.run_circuit_acq(n_total_shots, nsamples, acq_chans, trig_delay, decimator, return_acc, from_server)

    def run_circuit(self, 
                    n_total_shots: int, 
                    reads_per_shot: int | Dict[str, int] = None, 
                    timeout_per_shot: float = 8, 
                    from_server: bool = False):
        """
        Run the currently loaded program and acquire integrated IQ shots. Program is
        run `n_total_shots` times, in batches of size `shots_per_run` (i.e. `shots_per_run` runs of the program
        are executed in logic before each readback/restart cycle). The current gateware 
        is limited to ~1000 reads in its IQ buffer, which generally means 
        shots_per_run = 1000//reads_per_shot

        Parameters
        ----------
        n_total_shots: int
            number of shots to run. Program is restarted from the beginning 
            for each new shot
        reads_per_shot: int | dict
            number of values per shot per channel to read back from accbuf. If `dict`, indexed
            by str(channel_number) (same indices as `raw_asm_list`). If `int`, assumed to be 
            the same across channels. Unless multiple circuits were rastered pre-compilation or 
            there is mid-circuit measurement involved this is typically 1
        timeout_per_shot: float
            job will time out if time to take a single shot exceeds this value in seconds 
            (this likely means the job is hanging due to timing issues in the program or gateware)
        from_server: bool
            set to true if calling over RPC. If `True`, pack returned s11 arrays into
            byte objects

        Returns
        -------
        dict:
            Complex IQ shots for each accbuf in `chanlist`; each array has 
            shape `(n_total_shots, reads_per_shot)`
        """
        if isinstance(reads_per_shot, int):
            for _, chan in self.loaded_result_channels.items():
                chan.reads_per_shot = reads_per_shot
        elif isinstance(reads_per_shot, dict):
            for channame, n_reads in reads_per_shot.items():
                if channame in self.loaded_result_channels:
                    self.loaded_result_channels[channame].reads_per_shot = n_reads
        else:
            if reads_per_shot is not None:
                raise Exception(f'reads per shot: {reads_per_shot} invalid type')
 
        logging.getLogger(__name__).info(f'starting circuit with {n_total_shots} shots')

        if len(self.loaded_result_channels) == 0:
            shots_per_run = n_total_shots
        else:
            shots_per_run = min(self._pl_driver.get_mem_size(res_chan.mem_name) 
                                // (res_chan.reads_per_shot * get_result_class(res_chan.dtype).word_size()) 
                                for res_chan in self.loaded_result_channels.values())

        n_runs = int(np.ceil(n_total_shots/shots_per_run))
        results = {ch: bytes() for ch in self.loaded_result_channels.keys()}

        for i in range(n_runs): 
            #result = self._pl_driver.run_prog_acc(self.loaded_result_channels, shots_per_run, timeout_per_shot=timeout_per_shot)
            self.start_program(shots_per_run if i < (n_runs - 1) else n_total_shots - i*shots_per_run)
            cur_result = self.wait_and_readback()
            for ch in self.loaded_result_channels.keys():
                results[ch] += cur_result[ch]
 
        logging.getLogger(__name__).info('done circuit')

        if not from_server:
            for ch in results.keys():
                results[ch] = get_result_class(self.loaded_result_channels[ch].dtype)(results[ch], n_total_shots)

        return results

    def run_circuit_acq(self,
                        n_total_shots: int = 1, 
                        nsamples: int = 8192, 
                        acq_chans: Dict[str, int] = {'0':0,'1':1}, 
                        trig_delay: float = 0, 
                        decimator: int = 0, 
                        return_acc: bool = False, 
                        from_server: bool = False):
        """
        Run the currently loaded program and acquire raw (or downconverted) adc traces.

        Parameters
        ----------
        n_total_shots: int
            number of shots to run. Program is restarted from the beginning 
            for each new shot
        nsamples: int
            number of samples to read from the acq buffer
        acq_chans: dict
            current channel mapping is:

                '0': ADC_237_2 (main readout ADC)
                '1': ADC_237_0 (other ADC connected in gateware)
                TODO: figure out DLO channels, etc and what they mean
        trig_delay: float
            time to delay acquisition, relative to circuit start.
            NOTE: this value, when converted to units of clock cycles, is a 
            16-bit value. So, it maxes out at CLK_PERIOD*(2**16) = 131.072e-6
        decimator: int
            decimation interval when sampling. e.g. 0 means full sample rate, 1
            means capture every other sample, 2 means capture every third sample, etc
        return_acc: bool
            if True, return a single acc (integrated + accumulated readout) value per shot,
            on each loaded channel. Default is False.
        from_server: bool
            set to true if calling over RPC. If True, pack returned acq arrays into
            byte objects

        Returns
        -------
        tuple | Dict
            - if return_acc is False:

                - dict:
                    array of acq samples for each channel in `acq_chans` with shape `(n_total_shots, nsamples)`

            - if return_acc is True:

                - tuple:
                    - dict:
                        array of acq samples for each channel in acq_chans with shape `(n_total_shots, nsamples)`
                    - dict:
                        array of acc values for each loaded channel with length `n_total_shots`

        """
        if nsamples > MAX_NSAMPLES:
            raise RuntimeError(f'{nsamples} exceeds max_nsamples length of {MAX_NSAMPLES}')
        
        if return_acc:
            acc_chans = self.loaded_result_channels
        else:
            acc_chans = {}
        acq_data, acc_data = self._pl_driver.run_prog_acq(n_total_shots, nsamples, acq_chans, acc_chans,
                                                int(trig_delay/CLK_PERIOD), decimator)

        if from_server:
            for ch in acq_data.keys():
                acq_data[ch] = acq_data[ch].tobytes()
            for ch in acc_data.keys():
                acc_data[ch] = acc_data[ch].tobytes()

        if return_acc:
            return acq_data, acc_data

        else: 
            return acq_data

 
