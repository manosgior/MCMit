import numpy as np
from qubic.results.tools import herald_shots
from qubic.results.counts import shots_to_batchcounts
import ipdb

def test_heralding():
    shots = {'Q0': np.zeros((100, 2)),
             'Q1': np.zeros((100, 2))}

    shots['Q0'][10:15, 0] = 1
    shots['Q1'][12:20, 0] = 1
    shots['Q0'][50:65, 0] = 1

    shots['Q0'][:, 1] = np.arange(100)
    shots['Q1'][:, 1] = np.arange(100) + 100

    shots = herald_shots(shots)

    herald_mask = np.ones(100).astype(bool)
    herald_mask[10:20] = False
    herald_mask[50:65] = False
    correct_q0 = np.arange(100)[herald_mask]
    correct_q1 = correct_q0 + 100

    assert np.all(correct_q0 == np.squeeze(shots['Q0']).astype(int))
    assert np.all(correct_q1 == np.squeeze(shots['Q1']).astype(int))

def test_heralded_bitstring():
    shots = {'Q0': np.zeros((1, 100, 2)),
             'Q1': np.zeros((1, 100, 2))}

    shots['Q0'][:, 10:15, 0] = 1
    shots['Q1'][:, 12:20, 0] = 1
    shots['Q0'][:, 50:65, 0] = 1

    shots['Q0'][:, :, 1] = np.arange(100) % 2
    shots['Q1'][:, :, 1] = (np.arange(100) + 1) % 2 

    counts = shots_to_batchcounts(shots, herald=True)

    assert counts[0].bitstring_dict['01'] == 37
    assert counts[0].bitstring_dict['10'] == 38



