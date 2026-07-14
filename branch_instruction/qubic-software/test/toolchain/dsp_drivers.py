import cocotb
import random
import numpy as np
import matplotlib.pyplot as plt
from cocotb.triggers import Timer, RisingEdge
import distproc.command_gen as cg
from sim_tools import unravel_dac, ravel_adc, twoscomp_to_signed
from xmlrpc.client import Binary

N_MAX_CMD = 30 #for flushing cmd buffer
N_CLKS = 100000
CLK_CYCLE = 1
N_PROC = 4
CMD_WRITE_WORDS = 5

async def generate_clock(dut):
    while True:
        dut.clk.value = 0
        await Timer(CLK_CYCLE, units='ns')
        dut.clk.value = 1
        await Timer(CLK_CYCLE, units='ns')

class DSPDriver:
    """
    Class for running a program on simulated
    instances of a dsp_unit module in cocotb

    Attributes
    ----------
        dut : SimHandleBase
            top-level dsp_unit_sim module 
            under test
        self.mon_signals : dict -> [str : SimHandleBase]
            Dictionary of signals to monitor
                key: user assigned name
                value: SimHandleBase object (e.g. dut.dpr.regs)
        self.mon_data : dict -> [str : list]
            Dictionary of data corresponding to mon_signals;
            list indexed by clock cycle
        self.dac_i : numpy array
            shape: (n_dspunit, nsamples). values are signed 16-bit dac_i out
        self.dac_q : numpy array
            shape: (n_dspunit, nsamples). values are signed 16-bit dac_q out
    """
    def __init__(self, dut, dac_samples_per_clk, dac_nbits, adc_samples_per_clk, \
            adc_nbits, mon_signals=None):
        """
        TODO: make dac_samples_per_clk, etc optionally arrays to support different 
        sample rates
        """
        self._dut = dut
        self.mon_signals = {}
        self.mon_data = {}
        self._n_dac = len(self._dut.dac)
        self._dac_out_signal = self._dut.dac
        self.dac_samples_per_clk = dac_samples_per_clk
        self.adc_samples_per_clk = adc_samples_per_clk
        self.dac_nbits = dac_nbits
        self.adc_nbits = adc_nbits
        self.n_cores = len(self._dut.acc_read_data)
        if mon_signals is not None:
            for name, sig in mon_signals.items():
                self.add_mon(name, sig)

    @property
    def n_dac(self):
        return self._n_dac

    def add_mon(self, name, sig):
        self.mon_signals.update({name: sig})
        self.mon_data.update({name: []})

    async def capture_acc(self, ncycles):
        self.acc_vals = [np.array([], dtype=np.complex128) for i in range(self.n_cores)]
        for i in range(ncycles):
            await RisingEdge(self._dut.clk)
            for j in range(self.n_cores):
                if int(self._dut.we_accbuf.value[j]) != 0:
                    self._dut._log.info(f'acc_trigger {j}')
                    acc_raw = self._dut.data_accbuf[j].value
                    self.acc_vals[j] = np.append(self.acc_vals[j],
                                                 twoscomp_to_signed(acc_raw & (0xFFFFFFFF), nbits=32) 
                                                 + 1j*twoscomp_to_signed((acc_raw >> 32) & (0xFFFFFFFF), nbits=32))

    async def flush_cmd_mem(self, ncmd=N_MAX_CMD):
        cmd_lists = []
        for i in range(self.n_cores):
            cmd_lists.append(np.zeros(ncmd, dtype=np.int32).tobytes())
        await self.load_program(cmd_lists)

    async def load_asm_program(self, asm_prog):
        """
        Load a full asm_prog (dict) outputted by GlobalAssembler
        """
        self._dut._log.info('loading asm program')
        for chan_key, chan_asm in asm_prog.items():
            await self.load_cmd_buffer(chan_asm['cmd_buf'], int(chan_key))

            for elem_ind, env_buf in enumerate(chan_asm['env_buffers']):
                await self.load_env_buffer(env_buf, elem_ind, int(chan_key))

            for elem_ind, freq_buf in enumerate(chan_asm['freq_buffers']):
                await self.load_freq_buffer(freq_buf, elem_ind, int(chan_key))


    async def load_cmd_buffer(self, cmd_buffer, core_ind):
        if isinstance(cmd_buffer, Binary):
            cmd_buffer = cmd_buffer.data

        self._dut.reset.value = 1
        self._dut.mem_write_en.value = 1
        self._dut.mem_write_sel.value = 0

        dt = np.dtype(np.uint32)
        dt = dt.newbyteorder('little')
        cmd_list = np.frombuffer(cmd_buffer, dtype=dt)
        core_ind = int(core_ind)
        self._dut.proc_write_sel.value = core_ind

        for i, cmd in enumerate(cmd_list):
            self._dut.mem_write_data.value = int(cmd)
            self._dut.mem_write_addr.value = i
            await RisingEdge(self._dut.clk)

        self._dut.mem_write_en.value = 0

    async def load_program(self, cmd_lists):
        """
        cmd_lists : list of lists
            Each element n is a list of commands for the nth
            DSP unit
        """
        if not (isinstance(cmd_lists[0], bytes)):
            raise Exception('cmd_lists must be byte array')
        self._dut.reset.value = 1
        self._dut.mem_write_en.value = 1
        self._dut.mem_write_sel.value = 0
        for i, cmd_list in enumerate(cmd_lists):
            dt = np.dtype(np.uint32)
            dt = dt.newbyteorder('little')
            cmd_list = np.frombuffer(cmd_list, dtype=dt)
            self._dut.proc_write_sel.value = i
            for j, cmd in enumerate(cmd_list):
                self._dut.mem_write_data.value = int(cmd)
                self._dut.mem_write_addr.value = j
                await RisingEdge(self._dut.clk)

        self._dut.mem_write_en.value = 0

    #async def load_env(self, env_buffer_list):
    #    """
    #    Load full envelope for program
    #    """
    #    for i, env_buffer in enumerate(env_buffer_list):
    #        await self.load_unit_env(env_buffer, i)
        

    async def load_env_buffer(self, env_buffer, elem_ind, core_ind, wave_start_addr=0):
        """
        Load envelope for a single DSP unit
        env_buffer is packed 32-bit I and Q values, one dt sample per element

        elem_ind indexes the process element: 
            currently 0 is qdrv, 1 is rdrv, and 2 is rdlo
        """
        if isinstance(env_buffer, Binary):
            env_buffer = env_buffer.data
        self._dut.mem_write_en.value = 1
        self._dut.mem_write_sel.value = 1 + elem_ind*2 
        self._dut.proc_write_sel.value = core_ind
        wave_addr = wave_start_addr
        env_buffer = np.frombuffer(env_buffer, dtype=np.uint32)
        for sample in env_buffer:
            self._dut.mem_write_data.value = int(sample)
            self._dut.mem_write_addr.value = wave_addr
            await RisingEdge(self._dut.clk)
            wave_addr += 1
        
        self._dut.mem_write_en.value = 0

    async def load_freq_buffer(self, freq_buffer, elem_ind, core_ind):
        if isinstance(freq_buffer, Binary):
            freq_buffer = freq_buffer.data
        self._dut.mem_write_en.value = 1
        self._dut.mem_write_sel.value = 2 + elem_ind*2 
        self._dut.proc_write_sel.value = core_ind
        freq_buffer = np.frombuffer(freq_buffer, dtype=np.uint32)
        addr = 0
        for sample in freq_buffer:
            self._dut.mem_write_data.value = int(sample)
            self._dut.mem_write_addr.value = addr
            await RisingEdge(self._dut.clk)
            addr += 1
        
        self._dut.mem_write_en.value = 0

    async def reset(self):
        """
        Reset all of the proc cores and DSP elements
        """
        await RisingEdge(self._dut.clk)
        await RisingEdge(self._dut.clk)
        self._dut.reset.value = 1
        await RisingEdge(self._dut.clk)
        await RisingEdge(self._dut.clk)
        self._dut.reset.value = 0

    async def monitor_outputs(self, ncycles):
        """
        Monitor program output for ncycles clocks.
        Sets class attributes dac_i and dac_q, each of
        which is a (n_dac, n_samples) numpy array of
        DAC values. Also populates self.mon_data, if any
        mon signals have been declared
        """
        dac_out = []
        for i in range(ncycles):
            await RisingEdge(self._dut.clk)
            for name, sig in self.mon_signals:
                self.mon_data[name].append(sig.value)
            dac_out.append([int(val) for val in self._dac_out_signal.value[::-1]])

        dac_out = np.transpose(np.asarray(dac_out))
        self.dac_out = np.empty((self.n_dac, ncycles*self.dac_samples_per_clk))
        for i in range(self.n_dac):
            self.dac_out[i] = unravel_dac(dac_out[i], self.dac_samples_per_clk, self.dac_nbits)

    async def capture_demod_chain(self, ncycles):
        rdlo_x = []
        rdlo_y = []
        mixed_x = []
        mixed_y = []
        for i in range(ncycles):
            await RisingEdge(self._dut.clk)
            rdlo_x.append([int(self._dut.dspif.rdlo_signal_x[i].value) for i in range(self.n_cores)])
            rdlo_y.append([int(self._dut.dspif.rdlo_signal_y[i].value) for i in range(self.n_cores)])
            mixed_x.append([int(self._dut.dspif.adc_mix_dlo_x[i].value) for i in range(self.n_cores)])
            mixed_y.append([int(self._dut.dspif.adc_mix_dlo_y[i].value) for i in range(self.n_cores)])

        rdlo_x = np.transpose(np.asarray(rdlo_x))
        rdlo_y = np.transpose(np.asarray(rdlo_y))
        mixed_x = np.transpose(np.asarray(mixed_x))
        mixed_y = np.transpose(np.asarray(mixed_y))

        self.rdlo = np.empty((self.n_cores, ncycles*4), dtype=np.complex128)
        self.mixed = np.empty((self.n_cores, ncycles*4), dtype=np.complex128)
        for i in range(self.n_cores):
            self.rdlo[i] = unravel_dac(rdlo_x[i], 4, 16) + 1j*unravel_dac(rdlo_y[i], 4, 16)
            self.mixed[i] = unravel_dac(mixed_x[i], 4, 16) + 1j*unravel_dac(mixed_y[i], 4, 16)


    async def generate_adc_signal(self, adc_signal, adc_ind):
        """
        Signal is signed integer normalized to 2**15-1
        """
        adc_signal = ravel_adc(adc_signal, self.adc_samples_per_clk, self.adc_nbits)
        for i in range(len(adc_signal)):
            self._dut.adc[adc_ind].value = int(adc_signal[i])
            await(RisingEdge(self._dut.clk))

    async def read_acq_buf(self, nvalues, adc_ind, start_addr=0):
        acq_buf = np.zeros(nvalues)
        for i in range(nvalues):
            self._dut.buf_read_addr.value = i + start_addr
            acq_buf[i] = self._dut.acq_read_data[adc_ind].value
            await(RisingEdge(self._dut.clk))

        return unravel_dac(acq_buf, self.adc_samples_per_clk, self.adc_nbits)

    async def read_acc_buf(self, nvalues, proc_ind=0, start_addr=0):
        acc_buf = np.zeros(nvalues, dtype=np.complex128)
        for i in range(nvalues):
            self._dut.buf_read_addr.value = i + start_addr
            await(RisingEdge(self._dut.clk))
            await(RisingEdge(self._dut.clk))
            raw_val = self._dut.acc_read_data[proc_ind].value
            acc_buf[i] = twoscomp_to_signed(raw_val & (0xFFFFFFFF), nbits=32) \
                + 1j*twoscomp_to_signed((raw_val >> 32) & (0xFFFFFFFF), nbits=32)

        return acc_buf
            

    async def run_program(self, ncycles, nshots=1):
        """
        For backwards compatibility with earlier tests; can be used 
        to run simple programs without external (fproc) input.
        """
        self._dut.nshot.value = nshots
        await self.reset()
        self._dut.stb_start.value = 1
        await RisingEdge(self._dut.clk)
        self._dut.stb_start.value = 0
        await self.monitor_outputs(ncycles)

