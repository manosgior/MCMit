"""Slide version of the MECH "classical latency impact" panel: average of QFT/QAOA/VQE/BV.

Uses the stored MECH compiler results (3x3 grid of 7x7 square chiplets) and the same
depth model as evaluation/motivation/MECH/MECH_line_plot.ipynb; no compilation is re-run.
Run from the repo root:  python slides/plot_mech_classical_latency_avg.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MECH = ROOT / "evaluation" / "motivation" / "MECH"
OUT = Path(__file__).resolve().parent / "figures"
BENCHMARKS = ["qft", "qaoa", "vqe", "bv"]
MEAS_LATENCY = 10  # measurement / local CNOT latency ratio
CLASSICAL_LATENCIES = [2, 3, 4, 5, 6, 7, 8]
COLOR = "#77C38A"  # same green as the other slide plots

exp = json.load(open(MECH / "sample_exp_data" / "squ3377.json"))
base = json.load(open(MECH / "sample_baseline_data" / "squ3377_level_3.json"))["level_3"]


def depth_difference(b, c_l):
    e = exp[b]
    exp_depth = e["depth"] + e["shuttle_num"] * (MEAS_LATENCY - 2) * 2 + e["meas_num"] * c_l
    return 1 - exp_depth / base[b]["depth"]


diff = np.array([[depth_difference(b, c) for c in CLASSICAL_LATENCIES] for b in BENCHMARKS])
mean = diff.mean(axis=0)

plt.rcParams.update({"font.size": 18, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.axhline(0, color="0.35", linestyle="--", linewidth=1.2, zorder=1)
ax.plot(CLASSICAL_LATENCIES, mean, "-o", color=COLOR, linewidth=3, markersize=10,
        markeredgecolor="black", markeredgewidth=1.2, zorder=3)

ax.set_xticks(CLASSICAL_LATENCIES)
ax.set_xlabel("Latency ratio (classical feedback / local CNOT)", fontsize=16)
ax.set_ylabel("Depth difference")
ax.grid(axis="y", color="0.85", zorder=0)
ax.text(0.5, 1.04, "Higher is better ↑", transform=ax.transAxes, ha="center", va="bottom",
        fontweight="bold", color="blue")

fig.tight_layout()
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT / f"mech_classical_latency_avg.{ext}", dpi=300, transparent=(ext != "png"))

for c, m in zip(CLASSICAL_LATENCIES, mean):
    print(f"ratio {c}: mean depth difference {m:+.3f}")
