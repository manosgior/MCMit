from collections import OrderedDict
from distproc.command_gen import twos_complement
from typing import Dict, List, Tuple
import numpy as np
import logging
import itertools
import copy
from qubic.results.primitives import S11

def herald_shots(shot_dict: Dict[str, np.ndarray], herald_read_ind: int = 0, remove_herald: bool = True) -> Dict[str, np.ndarray]:
    """
    Parameters
    ----------
    shot_dict : Dict[np.ndarray]
        dictionary of classified shots, indexed by qubit
        values have shape (n_shots, reads_per_shot)
    herald_read_ind: int
        index of herald read, defaults to 0
    remove_herald: bool
        if True (default), remove herald read from the data,
        so returned data will have shape (n_shots, reads_per_shot-1)

    Returns
    -------
    Dict[str, np.ndarray]
        heralded shots
    """
    shot_dict_shape = list(shot_dict.values())[0].shape
    n_shots = shot_dict_shape[0]
    n_qubits = len(shot_dict)
    shot_dict = copy.deepcopy(shot_dict)

    herald_reads = np.zeros((n_qubits,  n_shots))
    for i, shots in enumerate(shot_dict.values()):
        herald_reads[i] = shots[:, herald_read_ind]

    herald_mask = np.all(herald_reads == 0, axis=0) # shots 

    for qubit in shot_dict.keys():
        shot_dict[qubit] = shot_dict[qubit][herald_mask]
        if remove_herald:
            shot_dict[qubit] = np.delete(shot_dict[qubit], herald_read_ind, axis=1)

    return shot_dict

def pack_s11_results(s11: List[Dict[str, S11]]) -> Dict[str, S11]:
    """
    Convert the list of dictionaries containing downconverted/accumulated data (S11 results) from:

        [{'ch': np.ndarray((n_shots, reads_per_shot))...}, {'ch': np.ndarray((n_shots, reads_per_shot))}]

    to the older format:
        
        s11 = {'ch': np.ndarray((n_circuits, n_shots, n_reads_per_shot))...}

    Gaps in data (e.g. due to differing number of `reads_per_shot` across different circuits) are 
    filled with np.nan.

    Parameters
    ----------
    s11: List[Dict[str, S11]]
        raw data returned by the board

    Returns
    -------
    Dict[str, S11]

    """
    acc_channels = set().union(*list(set(s11_i.keys()) for s11_i in s11)) # union of all proc channels in batch

    max_rps_perchan = {}
    for chan in acc_channels:
        chan_rps = [s11[i][chan].shape[1] for i in range(len(s11)) if chan in s11[i].keys()]
        if len(chan_rps) > 1:
            max_rps_perchan[chan] = max(*chan_rps)
        else:
            max_rps_perchan[chan] = chan_rps[0]

    if len(s11[0].values()) == 0:
        nshots = 0
    else:
        nshots = list(s11[0].values())[0].shape[0]
    # this is an array of nans with shape (n_circuits, n_shots, max_reads_per_shot[chan])
    s11_packed = {chan: np.nan*np.zeros((len(s11), nshots, max_rps_perchan[chan]), dtype=np.complex128) for chan in acc_channels}

    for i, s11_i in enumerate(s11):
        for chan in acc_channels:
            if chan in s11_i.keys():
                s11_packed[chan][i][:,:s11_i[chan].shape[1]] = s11_i[chan]

    return s11_packed

