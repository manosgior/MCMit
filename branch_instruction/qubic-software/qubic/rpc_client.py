import xmlrpc.client
import numpy as np
import logging
from typing import List, Dict
from qubic.abstract_runner import AbstractCircuitRunner
from qubic.results.tools import pack_s11_results
from distproc.executable import Executable
from qubic.results.primitives import get_result_class, PrimitiveResult

class CircuitRunnerClient(AbstractCircuitRunner):
    """
    CircuitRunner instance that can be run from a remote machine (i.e. not on
    the RFSoC ARM core) over RPC. Used for submitting compiled circuits to a QubiC board
    and receiving the resulting (integrated IQ or ADC timestream) data. Should be a drop-in 
    replacement for CircuitRunner for most experiments. Exposes the following methods from
    CircuitRunner:

        run_circuit
        run_circuit_batch
        load_circuit
        load_and_run_acq
    """

    def __init__(self, ip, port=9095):
        self.proxy = xmlrpc.client.ServerProxy('http://' + ip + ':' + str(port), allow_none=True)

    def run_circuit_batch(self, 
                          executables: List[Executable], 
                          n_total_shots: int, 
                          reads_per_shot: int | Dict = None, 
                          timeout_per_shot: float = 8,
                          reload_cmd: bool = True, 
                          reload_freq: bool = True, 
                          reload_env: bool = True, 
                          zero_between_reload: bool = True) -> List[Dict[str, PrimitiveResult]]:
        """
        Runs a batch of circuits given by a list of compiled executables. Each circuit is run `n_total_shots`
        times. `reads_per_shot` and `n_total_shots` are passed directly into `run_circuit`, and must
        be the same for all circuits in the batch. The parameters `reload_cmd`, `reload_freq`, `reload_env`, and 
        `zero_between_reload` control which of these fields is rewritten circuit-to-circuit (everything is 
        rewritten initially). Leave these all at `True` (default) for maximum safety, to ensure that QubiC 
        is in a clean state before each run. Depending on the circuits, some of these can be turned off 
        to save time.

        Parameters
        ----------
        executables : List[Executable]
            list of executables to run
        n_total_shots: int
            number of shots per circuit
        reads_per_shot: int | Dict[str, int]
            number of values per shot per channel to read back from accbuf. If `dict`, indexed
            by `str(channel_number)` (same indices as `raw_asm_list`). If `int`, assumed to be 
            the same across channels, else can be a per-channel dict. Unless multiple circuits 
            were rastered pre-compilation or there is mid-circuit measurement involved this is typically 1
        timeout_per_shot: float
            job will time out if time to take a single shot exceeds this value in seconds 
            (this likely means the job is hanging due to timing issues in the program or gateware)
        reload_cmd: bool
            if True, reload command buffer between circuits
        reload_freq: bool
            if True, reload freq buffer between circuits
        reload_env: bool
            if True, reload env buffer between circuits

        Returns
        -------
        List[Dict]:
            Measurement results. A dictionary of results is returned for each circuit in the batch.  
            This dict is keyed by  `ResultChannel` names in the `Executable`; values are arraylike objects 
            containing the measurement results for that channel. Result type is given by the  `dtype` field 
            in the `ResultChannel` object. These are always numpy-arraylike; a full listing  of available result 
            types is given in `qubic.results.primitives`. Results have shape `(n_total_shots, reads_per_shot)`.

            Putting this together, a returned collection of results might looks like:
                `[{'Q0.rdlo': np.ndarray((n_shots, reads_per_shot)), 'Q1.rdlo': np.ndarray((n_shots, reads_per_shot))...}, 
                  {'Q0.rdlo': np.ndarray((n_shots, reads_per_shot)), ...}, ...]`

        """
        # TODO: consider throwing some version of all the args here into a BatchedCircuitRun or somesuch
        # object
        serialized_exe = [exec.to_dict() for exec in executables]
        
        packed_results = self.proxy.run_circuit_batch(serialized_exe, n_total_shots, reads_per_shot, float(timeout_per_shot),
                                     reload_cmd, reload_freq, reload_env, zero_between_reload)
        results = []
        for i, packed_result in enumerate(packed_results):
            results.append({ch: get_result_class(executables[i].result_channels[ch].dtype)(data.data, n_total_shots) for ch, data in packed_result.items()})

        return results

    def load_and_run_acq(self, 
                         raw_asm_prog: Executable, 
                         n_total_shots: int = 1, 
                         nsamples: int = 8192, 
                         acq_chans: Dict = {'0': 0, '1': 1}, 
                         trig_delay: float = 0, 
                         decimator: int = 0, 
                         return_acc: bool = False) -> tuple | Dict:
        """
        Load the program given by raw_asm_prog and acquire raw (or downconverted) adc traces.

        Parameters
        ----------
        raw_asm_prog: dict
            ASM binary to run. See load_circuit for details.
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

        Returns
        -------
        tuple | Dict
            - if `return_acc` is `False`:

                - dict:
                    array of acq samples for each channel in acq_chans with shape `(n_total_shots, nsamples)`

            - if `return_acc` is `True`:

                - tuple:
                    - dict:
                        array of acq samples for each channel in `acq_chans` with shape `(n_total_shots, nsamples)`
                    - dict:
                        array of acc values for each loaded channel with length `n_total_shots`

        """
        data = self.proxy.load_and_run_acq(raw_asm_prog.to_dict(), n_total_shots, nsamples, acq_chans, trig_delay, decimator, return_acc)

        if return_acc:
            acq_data = data[0]
            acc_data = data[1]
        else:
            acq_data = data
            acc_data = {}

        for ch in acq_data.keys():
            acq_data[ch] = np.reshape(np.frombuffer(acq_data[ch].data, dtype=np.int32), (n_total_shots, nsamples))
        for ch in acc_data.keys():
            acc_data[ch] = np.frombuffer(acc_data[ch].data, dtype=np.complex128)

        if return_acc:
            return acq_data, acc_data
        else:
            return acq_data
