from collections import OrderedDict
import copy
import numpy as np
from typing import Dict, List, Tuple
import pygsti
from qubic.pygsti.transpiler import transpile
import qubic.state_disc as sd
from distproc.compiler import CompiledProgram, CompilerFlags
from distproc.hwconfig import ChannelConfig, FPGAConfig
from qubic.abstract_runner import AbstractCircuitRunner
from qubitconfig.qchip import QChip
from qubic.job_manager import JobManager
from qubic.results import CircuitCounts

class PyGSTiJobManager(JobManager):
    """
    Wrapper around JobManager class for compiling and running PyGSTi circuits
    """
    def __init__(self,                  
                 fpga_config: FPGAConfig, 
                 channel_configs: Dict[str, ChannelConfig], 
                 circuit_runner: AbstractCircuitRunner,
                 qchip: QChip,
                 gmm_manager: sd.GMMManager = None, 
                 target_platform: str = 'rfsoc'):

        if gmm_manager is None:
            raise(ValueError("PyGSTiJobManager needs a gmm_manager!"))
        super().__init__(fpga_config, channel_configs, circuit_runner, qchip,
                 gmm_manager, target_platform)

    def build_and_run_circuits(self, pygsti_circuits: List[pygsti.circuits.Circuit], 
                               qubit_map: List[str] | Dict[int, str],
                               n_total_shots: int, 
                               outputs : List[str] = ['pygsti_results'], 
                               compiler_flags: Dict[str, bool] | CompilerFlags = None, 
                               gateware_twirl = False,
                               fit_gmm: bool = False, 
                               reads_per_shot: int = 1, 
                               qchip: QChip = None, 
                               delay_before_circuit: float = 500.e-6,
                               reload_cmd: bool = True, 
                               reload_freq: bool = True, 
                               reload_env: bool = True, 
                               zero_between_reload: bool = True) -> dict:
        """
        Compile and run provided list of circuits. Output data products/analysis are controlled 
        by 'output' parameter. 

        Parameters
        ----------
        pygsti_circuits: list
            list of pygsti circuits to run
        qubit_map: List[str] | Dict[int, str]
            Maps pygsti quantum registers to physical qubits. If list
            of physical qubits, register 0 is assumed to correspond to the
            first element, etc. If dict, register -> qubit mapping is given
            directly as key value pairs.
        n_total_shots: int
            number of shots to run for each circuit
        outputs: list
            list of 's11', 'shots', 'pygsti_results', and/or 'counts'
        compiler_flags: dict | CompilerFlags
            see CompilerFlags definition for allowed keys
        gateware_twirl: bool = False
        fit_gmm: bool
        qchip: qubitconfig.qchip.QChip
            if provided, override self.qchip for compilation
        reads_per_shot : int
            number of reads (measurements) per qubit in each instruction
        reload_cmd: bool
        reload_freq: bool
        reload_env: bool
        zero_between_reload: bool

        Returns
        -------
        dict
            results with keys/types matching the provided 'outputs'

        """
        if reads_per_shot != 1:
            raise Exception(f'Multiple reads_per_shot ({reads_per_shot}) not supported')
        qubic_outputs = copy.copy(outputs)
        if 'pygsti_results' in outputs:
            qubic_outputs.remove('pygsti_results')
            qubic_outputs.append('counts')
        
        qubic_circuits = []
        for circuit in pygsti_circuits:
            qubic_circuits.append(transpile(circuit, qubit_map, gateware_twirl, delay_before_circuit))

        qubic_output_dict = super().build_and_run_circuits(qubic_circuits, n_total_shots, qubic_outputs, 
                                                                    compiler_flags=compiler_flags, fit_gmm=fit_gmm,
                                                                    reads_per_shot=reads_per_shot, qchip=qchip,
                                                                    reload_cmd=reload_cmd, reload_env=reload_env,
                                                                    reload_freq=reload_freq, 
                                                                    zero_between_reload=zero_between_reload)

        output_dict = {}
        for output_type in outputs:
            if output_type == 'pygsti_results':
                output_dict['pygsti_results'] = map_results(pygsti_circuits, qubic_output_dict['counts'], qubit_map)
            else:
                output_dict[output_type] = qubic_output_dict[output_type]

        return output_dict


def map_results(circuits: List[pygsti.circuits.Circuit], 
                batchcounts: List[CircuitCounts], 
                label_to_qubit: Dict[int, str] | List):
    """
    Convert native list of CircuitCounts objects to a PyGSTi Dataset object

    Parameters
    ----------
    circuits: List[pygsti.circuits.Circuit]
    batchcounts : List[CircuitCounts]
    label_to_qubit : list or dict
        mapping from pygsti registers (indices) to physical qubits
    """
    pygsti_dataset = pygsti.data.DataSet()
    if isinstance(label_to_qubit, dict):
        qubit_to_label = {qubit: label for label, qubit in label_to_qubit.items()}
    elif isinstance(label_to_qubit, list) or isinstance(label_to_qubit, np.ndarray):
        qubit_to_label = {qubit: label for label, qubit in enumerate(label_to_qubit)}

    for circuit, circuitcounts in zip(circuits, batchcounts):
        if list(circuitcounts.bitstring_dict.values())[0].shape[0] != 1:
            raise Exception('Multiple reads per shot not supported!')

        qubic_to_pygsti_bitstring = {}
        sorted_pygsti_labels = sorted([label for qubit, label in qubit_to_label.items() if qubit in circuitcounts.qubits])
        pygsti_labels_to_index = {label: sorted_pygsti_labels.index(label) for label in sorted_pygsti_labels}
        for pygsti_bitstring in circuitcounts.bitstring_dict.keys():
            #enumerate all bitstrings as pygsti strings, then backsolve for qubic mapping
            qb_bitstring = ''.join([pygsti_bitstring[pygsti_labels_to_index[qubit_to_label[circuitcounts.qubits[i]]]] for i in range(len(pygsti_bitstring))])
            qubic_to_pygsti_bitstring[qb_bitstring] = pygsti_bitstring

        result_dict = {qubic_to_pygsti_bitstring[bitstring]: counts for 
                       bitstring, counts in circuitcounts.bitstring_dict.items()}
        pygsti_dataset[circuit] = result_dict

    return pygsti_dataset

