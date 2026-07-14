import logging
from typing import List, Dict, Tuple
from distproc.executable import Executable, ResultChannel
import numpy as np

RESULT_MEM_SIZE = 2048

class SimInterface:

    def __init__(self):
        pass

    def write_mem_buf(self, name, mem_vals, start_addr=0) -> None:
        logging.getLogger(__name__).info(f'sim interface: writing mem {name}: {mem_vals}')

    def start_program(self, nshots: int, clock_count=None) -> None:
        logging.getLogger(__name__).info(f'sim interface: starting program with {nshots} shots'
                                         f'clock_count: {clock_count}')

    def write_reg(self, name, value):
        logging.getLogger(__name__).info(f'sim interface: writing register {name}: {value}')

    def get_mem_size(self, name):
        return RESULT_MEM_SIZE

    def set_default_regs(self):
        pass

    def get_program_memories(self, core_inds=None):
        return [f'command{i}' for i in range(8)]

    def wait_and_readback(self, 
                          acc_read_chans: Dict[str, ResultChannel],
                          nshots: int,
                          timeout_per_shot: float = 8):
        logging.getLogger(__name__).info(f'sim interface: wait and readback on channels {acc_read_chans}, timeout_per_shot: {timeout_per_shot}')

        res_dict = {channame: np.asarray(list(zip(np.arange(nshots*chan.reads_per_shot, 0, -1) - ((nshots*chan.reads_per_shot)//2),
                                                  np.arange(nshots*chan.reads_per_shot)-((nshots*chan.reads_per_shot)//2)))).astype(np.uint32)
                    for channame, chan in acc_read_chans.items()}
        return {channame: data.tobytes() for channame, data in res_dict.items()}

    def read(self, name):
        return 0

    def read_64b(self, name):
        return 0

    def write_64b(self, name, value):
        pass

