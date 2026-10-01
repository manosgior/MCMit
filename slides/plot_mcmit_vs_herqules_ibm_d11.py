"""Slide version of Fig. (a) "IBM Heron": MCMit vs HERQULES logical error rate, d=11 only.

Reads the corrected results in lattice-sim/ (these match the published figure; the copies in
lattice-sim/experiment_results/ are older). No simulation is re-run.
Run from the repo root:  python slides/plot_mcmit_vs_herqules_ibm_d11.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LATTICE = ROOT / "lattice-sim"
OUT = Path(__file__).resolve().parent / "figures"
DISTANCE = 11
SERIES = [  # (label, csv, color, hatch)
    ("MCMit", "mcm_tradeoff_mcm_ibm.csv", "#77C38A", "//"),
    ("HERQULES", "mcm_tradeoff_herqules_ibm.csv", "#F5B07A", "\\\\"),
]
BAR_WIDTH = 0.38

plt.rcParams.update({"font.size": 18, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 4.5))

for i, (label, csv, color, hatch) in enumerate(SERIES):
    df = pd.read_csv(LATTICE / csv)
    df = df[df["distance"] == DISTANCE].sort_values("measure_latency_ns")
    x = np.arange(len(df)) + (i - 0.5) * BAR_WIDTH
    ax.bar(x, df["logical_error_rate"], width=BAR_WIDTH, color=color, hatch=hatch,
           edgecolor="black", linewidth=1.2, label=label, zorder=2)
    print(label, dict(zip(df["measure_latency_ns"], df["logical_error_rate"])))

durations = sorted(df["measure_latency_ns"])
ax.set_yscale("log")
ax.set_xticks(np.arange(len(durations)), [str(d) for d in durations])
ax.set_xlabel("MCM duration (ns)")
ax.set_ylabel("Logical error rate")
ax.minorticks_off()
ax.set_yticks([0.05, 0.1, 0.2, 0.5], ["0.05", "0.1", "0.2", "0.5"])
ax.set_ylim(0.04, 1.1)  # headroom for the one-row legend
ax.grid(axis="y", color="0.85", zorder=0)
ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=16)
ax.text(0.5, 1.04, "Lower is better ↓", transform=ax.transAxes, ha="center", va="bottom",
        fontweight="bold", color="blue")

fig.tight_layout()
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT / f"mcmit_vs_herqules_ibm_d{DISTANCE}.{ext}", dpi=300, transparent=(ext != "png"))
