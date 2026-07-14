import qubic.job_manager as jm
import numpy as np
import trueq as tq
import qubic.trueq.transpiler as tp
import copy
from qubitconfig.qchip import QChip
from typing import Dict, List
from distproc.compiler import CompiledProgram, CompilerFlags
from distproc.hwconfig import ChannelConfig, FPGAConfig
from qubic.abstract_runner import AbstractCircuitRunner
from qubic.state_disc import GMMManager
from qubic.results import CircuitCounts

class TrueQJobManager:
    """
    Wrapper around JobManager class for compiling and running TrueQ circuits
    """

    def __init__(self, 
                 fpga_config: FPGAConfig, 
                 qchip: QChip, 
                 channel_configs: Dict[str, ChannelConfig], 
                 circuit_runner: AbstractCircuitRunner, 
                 gmm_manager: GMMManager = None, 
                 target_platform: str = 'rfsoc'):

        self.job_manager = jm.JobManager(fpga_config, channel_configs,
                                         circuit_runner, qchip, gmm_manager, target_platform)

    def build_and_run_circuits(self, 
                               trueq_circuits: tq.circuits.Circuit | tq.circuits.CircuitCollection,
                               label_to_qubit: Dict[int, str] | List[str], 
                               n_total_shots: int, 
                               entangler: str = 'cz',
                               outputs: List[str] = ['trueq_results'], 
                               compiler_flags: Dict[str, bool] | CompilerFlags = None,
                               gateware_twirl = False,
                               tq_compile = True,
                               fit_gmm: bool = False, 
                               reads_per_shot: int | Dict[str, int] = 1, 
                               qchip: QChip = None, 
                               delay_before_circuit: float = 500.e-6,
                               reload_cmd: bool = True, 
                               reload_freq: bool = True, 
                               reload_env: bool = True, 
                               zero_between_reload: bool = True) -> Dict:
        """
        Parameters
        ----------
        trueq_circuit : trueq.circuits.Circuit or CircuitCollection
        label_to_qubit : dict or list
            if dict, keys are trueq labels (ints) and values are qubitids ('Q0', 'Q1', etc)
            if list of qubitids, label 0 is assumed to be the first element, etc
        n_total_shots : int
            number of shots to run for each circuit
        entangler : str
            either 'cz' or 'cnot'
        outputs : list
            list of 's11', 'shots', 'trueq_results', and/or 'counts'
        compiler_flags: dict | CompilerFlags
            see CompilerFlags definition for allowed keys
        gateware_twirl: bool
        tq_compile: bool
            If True, run the TrueQ compiler to compile to native gates. Otherwise,
            the circuit is assumed to have already been compiled and is transpiled as is.
        fit_gmm : bool
        reads_per_shot : int
        qchip : qubitconfig.qchip.QChip
            if provided, override self.qchip for compilation
        delay_before_circuit: float
            qubit relaxation delay applied to each circuit during compilation
        reload_cmd: bool
        reload_freq: bool
        reload_env: bool
        zero_between_reload: bool

        Returns
        -------
            dict
                results with keys/types matching the provided 'outputs'
        """

        qubic_outputs = copy.copy(outputs)
        if 'trueq_results' in outputs:
            qubic_outputs.remove('trueq_results')
            qubic_outputs.append('counts')

        qubic_circuits = tp.transpile(trueq_circuits, label_to_qubit, entangler, gateware_twirl, tq_compile, delay_before_circuit)
        if isinstance(qubic_circuits[0], dict):
            qubic_circuits = [qubic_circuits]

        qubic_output_dict = self.job_manager.build_and_run_circuits(qubic_circuits, n_total_shots, qubic_outputs, 
                                                                    compiler_flags=compiler_flags, fit_gmm=fit_gmm,
                                                                    reads_per_shot=reads_per_shot, qchip=qchip,
                                                                    reload_cmd=reload_cmd, reload_env=reload_env,
                                                                    reload_freq=reload_freq, 
                                                                    zero_between_reload=zero_between_reload)

        output_dict = {}
        for output_type in outputs:
            if output_type == 'trueq_results':
                output_dict['trueq_results'] = batchcounts_to_results(qubic_output_dict['counts'], label_to_qubit)
            else:
                output_dict[output_type] = qubic_output_dict[output_type]

        return output_dict



    def collect_trueq_results(self, circuit_list, n_total_shots):
        pass


def batchcounts_to_results(batchcounts: List[CircuitCounts], label_to_qubit: Dict[int, str] | List):
    """
    Convert native list of CircuitCounts object to TrueQ Results objects

    Parameters
    ----------
        batchcounts : List[CircuitCounts]
        label_to_qubit : list or dict
            mapping from trueq labels (indices) to physical qubits
    """
    tq_results = []
    if isinstance(label_to_qubit, dict):
        qubit_to_label = {qubit: label for label, qubit in label_to_qubit.items()}
    elif isinstance(label_to_qubit, list) or isinstance(label_to_qubit, np.ndarray):
        qubit_to_label = {qubit: label for label, qubit in enumerate(label_to_qubit)}

    for circuitcounts in batchcounts:
        if list(circuitcounts.bitstring_dict.values())[0].shape[0] != 1:
            raise Exception('Multiple reads per shot not supported!')

        qubic_to_tq_bitstring = {}
        sorted_tq_labels = sorted([label for qubit, label in qubit_to_label.items() if qubit in circuitcounts.qubits])
        tq_labels_to_index = {label: sorted_tq_labels.index(label) for label in sorted_tq_labels}
        for tq_bitstring in circuitcounts.bitstring_dict.keys():
            #enumerate all bitstrings as tq strings, then backsolve for qubic mapping
            qb_bitstring = ''.join([tq_bitstring[tq_labels_to_index[qubit_to_label[circuitcounts.qubits[i]]]] for i in range(len(tq_bitstring))])
            qubic_to_tq_bitstring[qb_bitstring] = tq_bitstring

        result_dict = {qubic_to_tq_bitstring[bitstring]: counts for 
                       bitstring, counts in circuitcounts.bitstring_dict.items()}
        tq_results.append(tq.Results(result_dict))

    return tq_results


