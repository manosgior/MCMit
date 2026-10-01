"""Slide version of Fig. 11's constant-depth GHZ panel: MCMit vs Qiskit M3 fidelity (Raw dropped).

Reads the published ibm_fez results; nothing is re-run.
Run from the repo root:  python slides/plot_ghz_mcmit_vs_m3.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "software_mitigation_fidelity.csv"
OUT = Path(__file__).resolve().parent / "figures"
SERIES = [  # (method in CSV, legend label, color, marker)
    ("MCMit", "MCMit", "#77C38A", "o"),
    ("Qiskit M3", "Qiskit M3 [3]", "#F5B07A", "s"),
]

df = pd.read_csv(CSV)
df = df[df["Benchmark"] == "Constant-depth GHZ"]

plt.rcParams.update({"font.size": 18, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 4.5))

for method, label, color, marker in SERIES:
    s = df[df["Method"] == method].sort_values("N")
    ax.plot(s["N"], s["Fidelity"], marker=marker, color=color, linewidth=3, markersize=10,
            markeredgecolor="black", markeredgewidth=1.2, label=label, zorder=3)

ns = sorted(df["N"].unique())
ax.set_xticks(ns[::2])
ax.set_xlabel("Number of qubits")
ax.set_ylabel("Fidelity")
ax.set_ylim(0.3, 1.05)
ax.set_yticks([0.4, 0.6, 0.8, 1.0])
ax.grid(axis="y", color="0.85", zorder=0)
ax.legend(loc="lower left", frameon=False, fontsize=16)
ax.text(0.5, 1.04, "Higher is better ↑", transform=ax.transAxes, ha="center", va="bottom",
        fontweight="bold", color="blue")

fig.tight_layout()
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT / f"ghz_mcmit_vs_m3.{ext}", dpi=300, transparent=(ext != "png"))
