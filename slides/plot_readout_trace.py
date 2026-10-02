"""Slide figure: real readout traces for |0>, |1>, and a |1> that decays mid-readout (I quadrature only).

Data: the 5-qubit multiplexed raw IQ dataset (DRaw_C_{Tr,Te}), qubit 4, first 1000 ns (500 samples @ 500 MHz).
Qubit 4 is digitally demodulated at a refined IF and rotated so the |1>-|0> separation lies on I; only I is plotted.
  - |0>, |1>     : mean over all test shots with q4 in that state and the other qubits in |0> (35k each)
  - decaying |1> : one real single test shot labelled |1> (DECAY_SHOT), found by fitting a step-down template
The training file is used only to calibrate the IF and the I/Q rotation.
Run from the repo root:  python slides/plot_readout_trace.py   (needs h5py; set READOUT_DATA to override)
"""
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve, firwin

DATA = Path(os.environ.get("READOUT_DATA",
                           "/home/manosgior/Documents/GitHub/KLiNQ/qubit_readout_klinq/data/five_qubit_data"))
TRAIN, TEST = DATA / "DRaw_C_Tr_v0-001", DATA / "DRaw_C_Te_v0-002"
SHOTS_TRAIN, SHOTS_TEST = 15000, 35000   # shots per 5-qubit state; files are sorted by state
OUT = Path(__file__).resolve().parent / "figures"

QUBIT, IF_COARSE = 4, -127e6             # coarse IF from runners/_colleague_prep.calibrate (1 MHz FFT grid)
FS, N_SAMPLES, WIN = 500e6, 500, 25      # 1000 ns readout, 50 ns windows
N_WIN = N_SAMPLES // WIN
WIN_NS = WIN / FS * 1e9
STATE1 = 1 << QUBIT
DECAY_SHOT = 23945                        # index within the |q4=1> test block
BLUE, RED, SHADE = "#3B8BDB", "#D9572B", "#F0EDE6"
PAD = 100                                 # extra samples read past 1000 ns so the FIR has no edge artifact there
t = np.arange(N_SAMPLES + PAD) / FS
LOWPASS = firwin(101, 5e6, fs=FS)       # same FIR as runners/_colleague_prep, applied zero-phase 


def windows(f, key, start, n, if_freq, lowpass=False):
    """Demodulate shots [start, start+n) at if_freq and average into 50 ns windows.
    lowpass=True first rejects the other qubits' tones (smooth curves for display)."""
    x = f[key][start:start + n, :N_SAMPLES + PAD, :]
    z = (x[..., 0] + 1j * x[..., 1]) * np.exp(-2j * np.pi * if_freq * t)
    if lowpass:
        z = fftconvolve(z, LOWPASS[None, :], mode="same", axes=1)
    return z[:, :N_SAMPLES].reshape(n, N_WIN, WIN).mean(2)


with h5py.File(TRAIN, "r") as f:
    # Refine the IF from the residual phase slope of (|1> - |0>) on the plateau, then rotate the separation onto +I
    t_win = (np.arange(N_WIN) + 0.5) * WIN / FS
    d = windows(f, "X_train", STATE1 * SHOTS_TRAIN, 5000, IF_COARSE).mean(0) - windows(f, "X_train", 0, 5000, IF_COARSE).mean(0)
    if_freq = IF_COARSE + np.polyfit(t_win[5:], np.unwrap(np.angle(d))[5:], 1)[0] / (2 * np.pi)
    d = windows(f, "X_train", STATE1 * SHOTS_TRAIN, 5000, if_freq).mean(0) - windows(f, "X_train", 0, 5000, if_freq).mean(0)
    rot = np.exp(-1j * np.angle(d[5:].mean()))

with h5py.File(TEST, "r") as f:
    i0 = (windows(f, "X_test", 0, SHOTS_TEST, if_freq, lowpass=True) * rot).real
    i1 = (windows(f, "X_test", STATE1 * SHOTS_TEST, SHOTS_TEST, if_freq, lowpass=True) * rot).real
m0, m1, decay = i0.mean(0), i1.mean(0), i1[DECAY_SHOT]
plateau = m1[N_WIN // 2:].mean()
ring_up_ns = (np.argmax(m1 - m0 >= 0.9 * (plateau - m0[N_WIN // 2:].mean())) + 0.5) * WIN_NS  # 90% of plateau

# Prepend the t=0 point (no signal before the readout pulse) so curves start at the origin
x_ns = np.concatenate([[0], (np.arange(N_WIN) + 0.5) * WIN_NS])
with_origin = lambda v: np.concatenate([[m0[0]], v])

plt.rcParams.update({"font.family": "Lato", "font.size": 20})
fig, ax = plt.subplots(figsize=(12, 5))
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("0.45")

ax.axvspan(0, ring_up_ns, color=SHADE, zorder=0)
ax.plot(x_ns, with_origin(m0), color=BLUE, lw=3.5, label=r"$|0\rangle$")
ax.plot(x_ns, with_origin(m1), color=RED, lw=3.5, label=r"$|1\rangle$")
ax.plot(x_ns, with_origin(decay), color=RED, lw=3.5, ls=(0, (1.2, 1.2)), label=r"$|1\rangle$ that decays mid-readout")

lo, hi = min(m0.min(), decay.min()), max(m1.max(), decay.max())
ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.25 * (hi - lo))
ax.set_yticks([])
ax.set_ylabel("I (a.u.)", color="0.3")
ax.set_xlim(0, N_SAMPLES / FS * 1e9)
ax.set_xticks([0, 250, 500, 750, 1000], ["0", "250", "500", "750", "1000 ns"])
ax.tick_params(axis="x", length=0, pad=10, colors="0.2")
ax.text(ring_up_ns / 2, hi + 0.12 * (hi - lo), "ring-up", ha="center", va="center", color="0.25")
ax.text(700, (plateau + m0.mean()) / 2, "decay", ha="center", va="center", color="0.25")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, handlelength=2.2, columnspacing=2.5)


OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT / f"readout_trace.{ext}", dpi=300, bbox_inches="tight", transparent=(ext != "png"))
print(f"IF {if_freq / 1e6:.3f} MHz, ring-up {ring_up_ns:.0f} ns")
