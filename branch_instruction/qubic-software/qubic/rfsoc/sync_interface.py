class SyncInteface:
    """
    Class for handling PTP synchronization in software. Intended to run
    on the ZCU216, with functions called over xml-RPC.
    """

    def __init__(self, pl_driver):
        self._pl_driver = pl_driver

    def read_ptp_corrval(self):
        return str(self._pl_driver.read_64b('copper_corr64'))

    def write_ptp_corrval(self, corrval):
        corrval = int(corrval)
        self._pl_driver.write_reg('copper_clkcnt_sw', 1)
        self._pl_driver.write_64b('copper_corr64', corrval)

    def ptp_enable_tx(self, suffix=None):
        suffix = _get_suffix(suffix)
        self._pl_driver.write_reg(f'copper_tx_en{suffix}', 1)
        self._pl_driver.write_reg(f'copper_tx_rst{suffix}', 1)
        self._pl_driver.write_reg(f'copper_tx_rst{suffix}', 1)
        self._pl_driver.write_reg(f'copper_tx_rst{suffix}', 0)

    def ptp_disable_tx(self, suffix=None):
        suffix = _get_suffix(suffix)
        self._pl_driver.write_reg(f'copper_tx_en{suffix}', 0)

    def ptp_read_tx_clockcount(self, suffix=None) -> int:
        suffix = _get_suffix(suffix)
        return str(self._pl_driver.read_64b(f'copper_tx_clkcnt{suffix}'))

    def ptp_read_rx_clockcount(self, suffix=None) -> int:
        suffix = _get_suffix(suffix)
        return str(self._pl_driver.read_64b(f'copper_rx_clkcnt{suffix}'))

    def ptp_read_dsp_clockcount(self) -> int:
        return str(self._pl_driver.read_64b('copper_dspclkcnt'))
    
def _get_suffix(suffix):
    if suffix is None or suffix == '':
        return ''
    elif suffix[0] == '_':
        return suffix
    else:
        return f'_{suffix}'

