import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "applications"))

from BranchingPropagation import BranchingConstantPropagation as BCP
from ConstantPropagation import ConstantPropagation as CP

from constant_depth_GHZ import create_constant_depth_ghz
from long_range_CNOT import create_dynamic_CNOT_circuit


def mid_measures(qc):
    last_nonmeas = -1
    for i, inst in enumerate(qc.data):
        if inst.operation.name.lower() not in ("measure", "barrier", "delay"):
            last_nonmeas = i
    return sum(1 for i, inst in enumerate(qc.data)
               if inst.operation.name == "measure" and i < last_nonmeas)


def leaf_distribution_check(tree):
    """Walk the merged DAG, accumulate P(leaf)*Born over terminal qubits.
    Returns (P_all_zero, P_all_one, P_other) summed over leaves, and #leaves."""
    p_allzero = p_allone = p_other = 0.0
    leaves = 0
    seen_path = {}

    # accumulate path probabilities (DAG: a node may be reached by several paths)
    def walk(node, prob):
        nonlocal p_allzero, p_allone, p_other, leaves
        if node.split:
            walk(node.child_one, prob * node.p_one)
            walk(node.child_zero, prob * (1 - node.p_one))
        else:
            leaves += 1
            sv = node.leaf_state
            T = [q for q, _ in node.terminal_measures]
            az = ao = 0.0
            for k, v in sv.state.items():
                pr = abs(v) ** 2
                if all(not k[q] for q in T):
                    az += pr
                elif all(k[q] for q in T):
                    ao += pr
            p_allzero += prob * az
            p_allone += prob * ao
            p_other += prob * (1 - az - ao)
    walk(tree, 1.0)
    return p_allzero, p_allone, p_other, leaves


def main():
    for name, builder in [("constant_depth_GHZ", create_constant_depth_ghz),
                          ("long_range_CNOT", create_dynamic_CNOT_circuit)]:
        print(f"\n{'='*100}\n{name}\n{'='*100}", flush=True)
        print(f"{'n':>3} {'nq':>3} | {'MCM_in':>6} {'orig_elim':>9} {'branch_elim':>11} | "
              f"{'nodes':>6} {'splits':>6} {'leaves(subcirc)':>15} | "
              f"{'P(0..0)':>8} {'P(1..1)':>8} {'P(other)':>8} | {'opt_time_s':>10}", flush=True)
        print("-" * 100, flush=True)
        for n in range(5, 26, 2):
            qc = builder(n)
            m_in = mid_measures(qc)
            orig_elim = m_in - mid_measures(CP.optimize(qc, max_amplitudes=2**20, max_ent_group_size=64))

            t0 = time.perf_counter()
            tree = BCP.optimize(qc)
            dt = time.perf_counter() - t0

            st = BCP.count_stats(tree)
            pz, po, pother, leaves = leaf_distribution_check(tree)
            ok = "OK" if abs(pother) < 1e-6 and abs(pz - 0.5) < 1e-6 else "CHECK"
            print(f"{n:>3} {qc.num_qubits:>3} | {m_in:>6} {orig_elim:>9} {m_in:>11} | "
                  f"{st['nodes']:>6} {st['splits']:>6} {leaves:>15} | "
                  f"{pz:>8.4f} {po:>8.4f} {pother:>8.4f} | {dt:>10.4f}  {ok}", flush=True)


if __name__ == "__main__":
    main()
