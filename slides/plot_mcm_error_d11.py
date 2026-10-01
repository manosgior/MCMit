"""Slide version of Fig. (a) "MCM error rate": d=11 only, Google Sycamore circuit-level noise.

Reads the stored results; no simulation is re-run.
Run from the repo root:  python slides/plot_mcm_error_d11.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "lattice-sim" / "experiment_results" / "mcm_error_ler_google.csv"
OUT = Path(__file__).resolve().parent / "figures"
DISTANCE = 11
COLOR = "#77C38A"  # d=11 green from the paper figure, slightly deepened for projectors

df = pd.read_csv(CSV)
df = df[df["distance"] == DISTANCE].sort_values("measure_error")
p = df["measure_error"].to_numpy()
ler = df["logical_error_rate"].to_numpy()
x = np.arange(len(p))

plt.rcParams.update({"font.size": 18, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.bar(x, ler, width=0.6, color=COLOR, hatch="//", edgecolor="black", linewidth=1.2, zorder=2)

ax.set_yscale("log")
ax.set_xticks(x, [f"{v:g}" for v in p])
ax.set_xlabel("MCM error probability")
ax.set_ylabel("Logical error rate")
ax.grid(axis="y", which="major", color="0.85", zorder=0)
ax.minorticks_off()
ax.set_yticks([0.1, 0.2, 0.4], ["0.1", "0.2", "0.4"])
ax.set_ylim(0.07, 0.45)

ax.text(0.5, 1.04, "Lower is better ↓", transform=ax.transAxes, ha="center", va="bottom",
        fontweight="bold", color="blue")

fig.tight_layout()
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT / f"mcm_error_ler_d{DISTANCE}.{ext}", dpi=300, transparent=(ext != "png"))
print(f"Saved to {OUT}")
