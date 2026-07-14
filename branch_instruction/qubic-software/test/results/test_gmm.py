import numpy as np
from qubic.results.tools import pack_s11_results
from qubic.results.primitives import S11
import qubic.state_disc as sd
import distproc.hwconfig as hw

def test_gmm_default_chanmap():
    s11 = [
            {'Q0.rdlo': S11(np.arange(100).reshape(50,2)), 'Q1.rdlo': S11(np.arange(50).reshape(50,1))},
            {'Q1.rdlo': S11(np.arange(100).reshape(50,2))}, 
            {'Q0.rdlo': S11(np.arange(50).reshape(50,1)), 'Q1.rdlo': S11(np.arange(50).reshape(50,1))}
            ]

    packed_results = pack_s11_results(s11)

    channel_config = hw.load_channel_configs('../toolchain/channel_config.json')
    gmm_manager = sd.GMMManager(chanmap_or_chan_cfgs=channel_config)
    gmm_manager.fit(packed_results)
    print(gmm_manager.gmm_dict)
