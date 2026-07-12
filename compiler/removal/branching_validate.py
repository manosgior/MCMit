import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "applications"))

from collections import Counter
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from BranchingPropagation import BranchingConstantPropagation as BCP
from ConstantPropagation import ConstantPropagation as CP

from quantum_teleportation import (
    create_repeated_teleportation_circuit as REP,
    create_ladder_teleportation_circuit as LAD,
)


def mid_measures(qc):
    """Mid-circuit measure = a measure with any non-measure op after it."""
    last_nonmeas = -1
    for i, inst in enumerate(qc.data):
        if inst.operation.name.lower() not in ("measure", "barrier", "delay"):
            last_nonmeas = i
    return sum(1 for i, inst in enumerate(qc.data)
               if inst.operation.name == "measure" and i < last_nonmeas)


def final_zero_prob(inst):
    measured_q = None
    pre = QuantumCircuit(*inst.qregs, *inst.cregs)
    for d in inst.data:
        if d.operation.name == "measure":
            measured_q = inst.find_bit(d.qubits[0]).index
        else:
            pre.append(d.operation, d.qubits, d.clbits)
    sv = Statevector(pre)
    n = inst.num_qubits
    return sum(p for bs, p in sv.probabilities_dict().items() if bs[n - 1 - measured_q] == "0")


def branching_p0(qc, tree, samples=32):
    cache, freq = {}, Counter()
    for _ in range(samples):
        inst = BCP.generate_instance(tree, qc)
        sig = tuple((d.operation.name, tuple(inst.find_bit(q).index for q in d.qubits))
                    for d in inst.data)
        if sig not in cache:
            cache[sig] = final_zero_prob(inst)
        freq[sig] += 1
    return sum(cache[s] * c / samples for s, c in freq.items()), len(cache)


def original_aer_p0(qc, shots=8000):
    SIM = AerSimulator()
    res = SIM.run(transpile(qc, SIM), shots=shots).result()
    out = Counter()
    for bitstr, c in res.get_counts().items():
        out[bitstr.split()[0] if " " in bitstr else bitstr[0]] += c
    return out.get("0", 0) / sum(out.values())


def main():
    for name, builder in [("REPEATED", REP), ("LADDER", LAD)]:
        print(f"\n{'='*94}\n{name} teleportation   (ideal final readout = '0' w.p. 1.0)\n{'='*94}", flush=True)
        print(f"{'n':>3} {'nq':>3} | {'MCM_in':>6} | {'orig_pass_elim':>14} {'branch_elim':>11} | "
              f"{'nodes':>5} {'splits':>6} {'leaves':>6} {'uniq':>4} | {'branch_P0':>9} {'aer_P0':>7}", flush=True)
        print("-" * 94, flush=True)
        for n in range(5, 11):
            qc = builder(n)
            m_in = mid_measures(qc)
            orig_elim = m_in - mid_measures(CP.optimize(qc, max_amplitudes=2**20, max_ent_group_size=64))

            tree = BCP.optimize(qc)
            st = BCP.count_stats(tree)
            branch_elim = m_in - 0  # instances carry only terminal read-out

            p0, uniq = branching_p0(qc, tree)
            aer = original_aer_p0(qc) if n <= 7 else float("nan")
            ok = "OK" if abs(p0 - 1.0) < 1e-6 else "FAIL"
            print(f"{n:>3} {qc.num_qubits:>3} | {m_in:>6} | {orig_elim:>14} {branch_elim:>11} | "
                  f"{st['nodes']:>5} {st['splits']:>6} {st['leaves']:>6} {uniq:>4} | "
                  f"{p0:>9.5f} {aer:>7.4f}  {ok}", flush=True)


if __name__ == "__main__":
    main()
