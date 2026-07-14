from abc import ABC, abstractmethod
from typing import List, Dict

class AbstractCircuitRunner(ABC):
    """
    Defines interface for loading and running batches of compiled circuits. Used primarily
    by `job_manager` and other higher level software requiring a circuit runner as input. 
    Current implementations include qubic.run.CircuitRunner (for running circuits/submitted jobs
    locally on the RFSoC), and qubic.rpc_client (for submitting/running circuits over RPC)
    """

    def __init__(self):
        pass

    @abstractmethod
    def run_circuit_batch(self, 
                          raw_asm_list: List[Dict], 
                          n_total_shots: int, 
                          reads_per_shot: int | Dict = 1, 
                          timeout_per_shot: int = 8,
                          reload_cmd: bool = True, 
                          reload_freq: bool = True, 
                          reload_env: bool = True, 
                          zero_between_reload: bool = True) -> Dict:
        """
        Runs a batch of circuits given by a list of raw_asm "binaries". Each circuit is run n_total_shots
        times. `reads_per_shot`, `n_total_shots`, and `delay_per_shot` are passed directly into `run_circuit`, and must
        be the same for all circuits in the batch. The parameters reload_cmd, reload_freq, reload_env, and 
        `zero_between_reload` control which of these fields is rewritten circuit-to-circuit (everything is 
        rewritten initially). Leave these all at `True` (default) for maximum safety, to ensure that QubiC 
        is in a clean state before each run. Depending on the circuits, some of these can be turned off 
        to save time.

        Parameters
        ----------
        raw_asm_list : list
            list of raw_asm binaries to run
        n_total_shots : int
            number of shots per circuit
        reads_per_shot : int | dict
            number of values per shot per channel to read back from accbuf. If dict, indexed
            by str(channel_number) (same indices as raw_asm_list). If int, assumed to be 
            the same across channels. Unless multiple circuits were rastered pre-compilation or 
            there is mid-circuit measurement involved this is typically 1
        timeout_per_shot : float
            job will time out if time to take a single shot exceeds this value in seconds 
            (this likely means the job is hanging due to timing issues in the program or gateware)
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
            shape `(len(raw_asm_list), n_total_shots, reads_per_shot)`
        """
        pass

    @abstractmethod
    def load_and_run_acq(self, 
                         raw_asm_prog: List[Dict], 
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
        raw_asm_prog : dict
            ASM binary to run. See load_circuit for details.
        n_total_shots : int
            number of shots to run. Program is restarted from the beginning 
            for each new shot
        nsamples : int
            number of samples to read from the acq buffer
        acq_chans : dict
            current channel mapping is:
                '0': ADC_237_2 (main readout ADC)
                '1': ADC_237_0 (other ADC connected in gateware)
                TODO: figure out DLO channels, etc and what they mean
        trig_delay : float
            time to delay acquisition, relative to circuit start.
            NOTE: this value, when converted to units of clock cycles, is a 
            16-bit value. So, it maxes out at `CLK_PERIOD*(2**16) = 131.072e-6`
        decimator : int
            decimation interval when sampling. e.g. 0 means full sample rate, 1
            means capture every other sample, 2 means capture every third sample, etc
        return_acc : bool
            if True, return a single acc (integrated + accumulated readout) value per shot,
            on each loaded channel. Default is `False`.

        Returns
        -------
        if `return_acc` is `False`:
            dict:
                array of acq samples for each channel in acq_chans with shape `(n_total_shots, nsamples)`

        if `return_acc` is `True`:
            tuple:
                dict:
                    array of acq samples for each channel in acq_chans with shape `(n_total_shots, nsamples)`
                dict:
                    array of acc values for each loaded channel with length `n_total_shots`

        """
        pass
