import numpy as np
from distproc.command_gen import twos_complement

CORDIC_DELAY = 47 #in clks
PHASEIN_DELAY = 1
QCLK_DELAY = 4
CSTROBE_DELAY = 2 #in clks
PHASE_RST_DELAY = 9
CLK_CYCLE = 2 # in ns
N_CLKS = 50000
ENV_BITS = 16


def unravel_dac(dac_out, samples_per_clk, nbits):
    dac_out_unravel = []
    for val in dac_out:
        for i in range(samples_per_clk):
            sliced_val = (int(val) >> (i*nbits)) & (2**nbits - 1)
            dac_out_unravel.append(twoscomp_to_signed(sliced_val, nbits))

    return np.asarray(dac_out_unravel)

def ravel_adc(adc_samples, samples_per_clk, nbits):
    adc_samples = np.pad(adc_samples, (0, len(adc_samples)%samples_per_clk))
    adc_ravel = np.zeros(len(adc_samples)//samples_per_clk, dtype=np.uint64)
    #adc_samples *= 2**(nbits-1) - 1
    adc_samples = np.array(adc_samples, dtype=int)
    for i in range(len(adc_ravel)):
        for j in range(samples_per_clk):
            # need to convert output to python int here to avoid overflow...
            adc_ravel[i] += int(twos_complement(adc_samples[samples_per_clk*i+j], nbits)) << (nbits * j)

    return adc_ravel

def twoscomp_to_signed(value, nbits=16):
    sval = value & (2**(nbits - 1) - 1)
    sval += -1*(value & (2**(nbits - 1)))
    return sval


def check_dacout_equal(dac_out_sim, dac_out, tol=.005):
    tol = tol*np.max(dac_out)
    max_len = max(len(dac_out), len(dac_out_sim))
    dac_out = np.pad(dac_out, (0, max_len-len(dac_out)))
    dac_out_sim = np.pad(dac_out_sim, (0, max_len-len(dac_out_sim)))
    return np.all(np.abs(dac_out - dac_out_sim) < tol)

# def dac_debug_plots(program, dac_out):
#     dac_i_sim, dac_q_sim = generate_sim_output(program)
#     plt.plot(dac_i, '-', label='I')
#     plt.plot(dac_q, '-', label='Q')
#     plt.plot(dac_i_sim, ':', label='Sim I')
#     plt.plot(dac_q_sim, ':', label='Sim Q')
#     plt.legend()
#     plt.xlabel('Time (ns)')
#     plt.show()

def generate_sim_dacout(pulse_sequence, samples_per_clk, extra_delay=0, interp_ratio=1, ncycles=N_CLKS):
    dac_out_sim = np.zeros(ncycles)
    scale_factor = 2**(ENV_BITS - 1)
    for pulse in pulse_sequence:
        pulse_length = interp_ratio*len(pulse['env']) #length in samples
        sample_inds = np.arange(0, pulse_length)
        start_time = samples_per_clk*pulse['start_time'] + samples_per_clk*(CSTROBE_DELAY + QCLK_DELAY + PHASEIN_DELAY)
        phases = pulse['phase'] + 2*np.pi*(CLK_CYCLE/samples_per_clk)\
                *1.e-9*(sample_inds + start_time - samples_per_clk*(PHASE_RST_DELAY))*pulse['freq']
        if interp_ratio>1:
            pulse['env'] = np.vstack([pulse['env'] for i in range(interp_ratio)]).T.flatten()
        env_i = scale_factor*pulse['amp']*np.real(pulse['env'])[:pulse_length]
        env_q = scale_factor*pulse['amp']*np.imag(pulse['env'])[:pulse_length]
        pulse_i = env_i*np.cos(phases) - env_q*np.sin(phases)
        pulse_q = env_q*np.cos(phases) + env_i*np.sin(phases)

        dac_out_sim[(CORDIC_DELAY + extra_delay)*samples_per_clk + start_time : \
                (CORDIC_DELAY + extra_delay)*samples_per_clk + start_time + pulse_length] = pulse_i

    return dac_out_sim.astype(int)

