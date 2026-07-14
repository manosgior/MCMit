import numpy as np
import qubic.results.tools as rs
from qubic.results.primitives import S11

def test_pack_s11():
    s11 = [
            {'Q0.rdlo': S11(np.arange(100).reshape(50,2)), 'Q1.rdlo': S11(np.arange(50).reshape(50,1))},
            {'Q1.rdlo': S11(np.arange(100).reshape(50,2))}, 
            {'Q0.rdlo': S11(np.arange(50).reshape(50,1)), 'Q1.rdlo': S11(np.arange(50).reshape(50,1))}
            ]
    packed_results_ideal = {
            'Q0.rdlo': S11(np.array([np.arange(100).reshape(50, 2), np.nan*np.zeros((50, 2)), np.zeros((50, 2))])),
            'Q1.rdlo': S11(np.array([np.zeros((50, 2)), np.arange(100).reshape((50, 2)), np.zeros((50, 2))]))
            }
    packed_results_ideal['Q0.rdlo'][2][:, 0] = np.arange(50)
    packed_results_ideal['Q0.rdlo'][2][:, 1] = np.nan
    packed_results_ideal['Q1.rdlo'][0][:, 0] = np.arange(50)
    packed_results_ideal['Q1.rdlo'][0][:, 1] = np.nan
    packed_results_ideal['Q1.rdlo'][2][:, 0] = np.arange(50)
    packed_results_ideal['Q1.rdlo'][2][:, 1] = np.nan
    packed_results = rs.pack_s11_results(s11)
    for chan in s11[0].keys():
        assert np.array_equal(packed_results[chan], packed_results_ideal[chan], equal_nan=True)
