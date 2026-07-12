"""Does eliminating MCMs change how many shots you need?

We compare, at matched total shots, the sampling error of:
  (A) the ORIGINAL dynamic circuit (real mid-circuit measurements), and
  (B) the OPTIMIZED circuit from the merged engine (MCMs eliminated),
both estimating the same final-readout distribution.  If the error-vs-shots
curves overlap, MCM elimination is shot-neutral (distribution-preserving).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "applications"))

from collections import Counter
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator

from BranchingPropagation import BranchingConstantPropagation as BCP
from constant_depth_GHZ import create_constant_depth_ghz
from quantum_teleportation import create_repeated_teleportation_circuit as REP

SIM = AerSimulator()


def marginal_final(counts):
    out = Counter()
    for b, c in counts.items():
        out[b.split()[0] if " " in b else b[0]] += c
    t = sum(out.values())
    return {k: v / t for k, v in out.items()}


def first_reg_dist(counts):
    """Marginal over the leftmost (last-added) classical register."""
    out = Counter()
    for b, c in counts.items():
        out[b.split()[0] if " " in b else b] += c
    t = sum(out.values())
    return {k: v / t for k, v in out.items()}


def tvd(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))


def build_optimized(qc):
    """Single subcircuit from the merged engine: prepare ref state + read out."""
    rep = BCP.optimize_merged(qc)
    sv = rep["ref_state"]
    out = QuantumCircuit(*qc.qregs, *qc.cregs)
    out.append(StatePreparation(sv.to_state_vector()), list(out.qubits))
    for q, c in rep["terminal_measures"]:
        out.measure(out.qubits[q], out.clbits[c])
    return out, rep["mcm_eliminated"]


def run(qc, shots):
    return SIM.run(transpile(qc, SIM), shots=shots).result().get_counts()


def main():
    print("GHZ n=5  ideal = {00000:0.5, 11111:0.5}; estimating full readout distribution")
    ghz = create_constant_depth_ghz(5)
    opt, k = build_optimized(ghz)
    ideal = {"0" * 5: 0.5, "1" * 5: 0.5}
    print(f"  MCMs eliminated: {k};  measure original vs optimized TVD-to-ideal at matched shots")
    print(f"  {'shots':>7} | {'orig_TVD':>9} | {'opt_TVD':>9}")
    for S in [128, 512, 2048, 8192, 32768]:
        # average over a few repeats to compare variance fairly
        ot = sum(tvd(first_reg_dist(run(ghz, S)), ideal) for _ in range(5)) / 5
        pt = sum(tvd(first_reg_dist(run(opt, S)), ideal) for _ in range(5)) / 5
        print(f"  {S:>7} | {ot:>9.4f} | {pt:>9.4f}")

    print("\nREPEATED teleport n=4  ideal final = {0:1.0}")
    tel = REP(4)
    opt, k = build_optimized(tel)
    print(f"  MCMs eliminated: {k}")
    print(f"  {'shots':>7} | {'orig P(0)':>9} | {'opt P(0)':>9}")
    for S in [128, 512, 2048, 8192]:
        op = sum(marginal_final(run(tel, S)).get("0", 0) for _ in range(5)) / 5
        pp = sum(marginal_final(run(opt, S)).get("0", 0) for _ in range(5)) / 5
        print(f"  {S:>7} | {op:>9.4f} | {pp:>9.4f}")


if __name__ == "__main__":
    main()
