from collections import OrderedDict
from distproc.command_gen import twos_complement
from typing import Dict, List, Tuple
from qubic.results.tools import herald_shots
import numpy as np
import logging
import itertools
import copy

try:
    import ipdb
except ImportError:
    pass

class CircuitCounts:
    """
    Object for storing shot count results (i.e. number of '000', '100', etc) from a batch of 
    experiments. Intended to only store counts from a single circuit, but possibly with multiple
    reads.

    CircuitCounts are summed along the nshots dimension, so bitstring count values have shape 
    (reads_per_shot,)

    The keys of `count_dict` and `bitstring_dict` have results in the same order as `qubits`.

    Attributes
    ----------
    qubits: List[str]
        list of qubits in dataset
    count_dict: Dict[Tuple, int] 
        dictionary of counts indexed by bitstring formatted as tuple of sequential gmm labels.
        Tuple is indexed in the same order as `qubits`
    bitstring_dict: Dict[str, int]
        dictionary of counts indexed by bitstring literal

    TODO:
        add heralded bitstring dict
    """

    def __init__(self, shot_dict: Dict[str, np.ndarray], gmm_labels: list = [0, 1]):
        """
        Parameters
        ----------
        shot_dict : Dict[np.ndarray]
            dictionary of classified shots, indexed by qubit
            values have shape (nshots, reads_per_shot)

        gmm_labels : list
            list of possible shot classification values
        """
        self.qubits = sorted(shot_dict.keys())
        self.shot_dict = OrderedDict()
        for qubit in self.qubits:
            self.shot_dict[qubit] = shot_dict[qubit]

        self.bit_tuples = [bittuple for bittuple in itertools.product(*[gmm_labels for i in range(len(self.qubits))])]

        self._generate_counts()

        self._bitstring_dict = None

    def _generate_counts(self):
        count_dict = OrderedDict()
        shot_array = np.array([shots for shots in self.shot_dict.values()]) #this should have dims (nqubits, nshots, reads_per_shot)
        for bit_tuple in self.bit_tuples:
            #shape (n_circuits, nshots, reads_per_shot); i.e. does this measurement satisfy the current bit tuple?
            bitstring_sat_mask = np.asarray([c for c in (shot_array[i] == bit_tuple[i] for i in range(len(self.qubits)))])
            bitstring_sat = np.prod(bitstring_sat_mask, axis=0) 
            count_dict[bit_tuple] = np.sum(bitstring_sat, axis=0) #sum over nshots, shape is (reads_per_shot,)

        self.count_dict = count_dict

    @property 
    def bitstring_dict(self):
        """
        lazy generation + store
        """
        if self._bitstring_dict is None:
            self._bitstring_dict = OrderedDict()
            for bit_tuple in self.bit_tuples:
                bitstring = ''.join([str(bit) for bit in bit_tuple])
                self._bitstring_dict[bitstring] = self.count_dict[bit_tuple]

        return self._bitstring_dict

    def get_mitigated_results(self, mitig_mat: np.ndarray):
        """

        Parameters
        ----------
        mitig_mat: np.ndarray
            shape (len(self.count_dict), len(self.count_dict)); (2**n_qubits, 2**n_qubits) for two state readout.
            multiplies the measurement vector to get mitigated results
            
        """
        mitig_counts = copy.deepcopy(self)
        meas_vec = np.array(list(self.count_dict.values()))
        mitig_meas_vec = mitig_mat @ meas_vec
        for i, bittuple in enumerate(mitig_counts.count_dict.keys()):
            mitig_counts.count_dict[bittuple] = mitig_meas_vec[i]

        mitig_counts._bitstring_dict = None
        return mitig_counts

    def __str__(self):
        return str(self.bitstring_dict)

    def __repr__(self):
        return f'CircuitCounts(qubits: {self.qubits}, counts: {self.count_dict})'

def shots_to_batchcounts(shot_dict: Dict[str, np.ndarray], 
                         gmm_labels: List = [0, 1],
                         herald: bool = False,
                         herald_read_ind: int = 0) -> List[CircuitCounts]:
    """
    Given a dictionary of classified shots from a batch, compute bitstring results 
    (`CircuitCounts`) for each circuit.

    Parameters
    ----------
    shot_dict : Dict[np.ndarray]
        dictionary of classified shots, indexed by qubit
        values have shape (n_circuits, n_shots, reads_per_shot)
    gmm_labels : list
        list of possible shot classification values

    Returns
    -------
    List[CircuitCounts]
        list of CircuitCounts objects, one per circuit in batch
    """
    n_circuits = len(list(shot_dict.values())[0])
    batched_counts = []

    for i in range(n_circuits):
        shot_dict_i = {}
        for qubit in shot_dict.keys():
            if not np.any(np.isnan(shot_dict[qubit][i])):
                shot_dict_i[qubit] = shot_dict[qubit][i]
            else:
                if not np.all(np.isnan(shot_dict[qubit][i])):
                    logging.getLogger(__name__).warning(f'qubit: {qubit}, circuit: {i}: \
                            partial data found, skipping bitstring')

        if herald:
            shot_dict_i = herald_shots(shot_dict_i, herald_read_ind)

        batched_counts.append(CircuitCounts(shot_dict_i, gmm_labels))

    return batched_counts
