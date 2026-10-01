"""Slide version of the XOR / classical-feedback latency table.

Values are copied from the paper table (no raw data file): feedback latency = 189 ns of
other feedback-path latency + branch-instruction latency (16 ns per XORed qubit).
Run from the repo root:  python slides/plot_xor_feedback_latency.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
N_QUBITS = [1, 8, 16, 32]
BRANCH_NS = [16, 128, 256, 512]
FEEDBACK_NS = [205, 317, 445, 701]
OTHER_NS = [f - b for f, b in zip(FEEDBACK_NS, BRANCH_NS)]  # 189 ns in every row
MCMIT_GREEN = "#5DAA45"

plt.rcParams.update({"font.size": 18, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 4.5))

x = np.arange(len(N_QUBITS))
ax.bar(x, OTHER_NS, width=0.6, color="#D9D9D9", hatch="..", edgecolor="black", linewidth=1.2,
       label="Pulse preparation, ADCs, and DACs", zorder=2)
ax.bar(x, BRANCH_NS, bottom=OTHER_NS, width=0.6, color="#F5B07A", hatch="\\\\", edgecolor="black",
       linewidth=1.2, label="Branch instructions", zorder=2)

for xi, total in zip(x, FEEDBACK_NS):
    increase = int((total - FEEDBACK_NS[0]) / FEEDBACK_NS[0] * 100)  # truncated, as in the paper table
    if xi == 0:  # QubiC at N=1 and MCMit have the same latency
        ax.text(xi, total + 15, "= MCMit", ha="center", va="bottom", fontweight="bold", color=MCMIT_GREEN)
    else:
        ax.text(xi, total + 15, f"+{increase}%", ha="center", va="bottom", fontweight="bold")

# Bracket over all bars: every bar is QubiC
BRACKET_Y = 860
ax.plot([-0.3, -0.3, 3.3, 3.3], [BRACKET_Y - 30, BRACKET_Y, BRACKET_Y, BRACKET_Y - 30], color="black", linewidth=1.5)
ax.text(1.5, BRACKET_Y + 10, "QubiC [4]", ha="center", va="bottom", fontweight="bold")

ax.set_xticks(x, [str(n) for n in N_QUBITS])
ax.set_xlabel("Qubits in XOR operation (N)")
ax.set_ylabel("Feedback latency (ns)")
ax.set_ylim(0, 960)
ax.set_yticks([0, 200, 400, 600, 800])
ax.grid(axis="y", color="0.85", zorder=0)
ax.legend(loc="upper left", bbox_to_anchor=(0, 0.86), frameon=False, fontsize=14)
ax.text(0.5, 1.04, "Lower is better ↓", transform=ax.transAxes, ha="center", va="bottom",
        fontweight="bold", color="blue")

fig.tight_layout()
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT / f"xor_feedback_latency.{ext}", dpi=300, transparent=(ext != "png"))
print(OTHER_NS)
