import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "applications"))

from ConstantPropagation import ConstantPropagation
from util.ProbabilisticGate import ProbabilisticGate
from constant_depth_GHZ import create_constant_depth_ghz
from long_range_CNOT import create_dynamic_CNOT_circuit


def count_ops(qc):
    """Count measurements (recursing into if_else branches) and prob_gates."""
    meas = 0
    reset = 0
    prob = 0
    for inst in qc.data:
        name = inst.operation.name.lower()
        if isinstance(inst.operation, ProbabilisticGate):
            prob += 1
        elif name == "measure":
            meas += 1
        elif name == "reset":
            reset += 1
        elif name == "if_else":
            for blk in inst.operation.params:
                if blk is None:
                    continue
                m, r, p = count_ops(blk)
                meas += m
                reset += r
                prob += p
    return meas, reset, prob


def total_measures(qc):
    """All measure ops, recursing into control-flow blocks."""
    m, r, p = count_ops(qc)
    return m, r, p


def analyze(name, builder, sizes, max_amp, max_grp):
    print(f"\n{'='*78}\n{name}   (max_amplitudes={max_amp}, max_ent_group_size={max_grp})\n{'='*78}")
    # mid-circuit measures = total measures minus the n terminal measures
    print(f"{'n':>3} | {'MCM_in':>6} {'reset_in':>8} | {'MCM_out':>7} {'reset_out':>9} {'prob_out':>8} | {'MCM_elim':>8}")
    print("-"*78)
    for n in sizes:
        qc = builder(n)
        m_in, r_in, p_in = total_measures(qc)
        try:
            opt = ConstantPropagation.optimize(qc, max_amplitudes=max_amp, max_ent_group_size=max_grp)
        except Exception as e:
            print(f"{n:>3} | {m_in:>6} {r_in:>8} | ERROR: {type(e).__name__}: {e}")
            continue
        m_out, r_out, p_out = total_measures(opt)
        # terminal measures stay = n (GHZ measure_all) or 2 (cnot cr3). Report raw.
        print(f"{n:>3} | {m_in:>6} {r_in:>8} | {m_out:>7} {r_out:>9} {p_out:>8} | {m_in - m_out:>8}")


if __name__ == "__main__":
    ghz_sizes = [n for n in range(5, 26, 2)]
    cnot_sizes = [n for n in range(5, 26, 2)]

    for (max_amp, max_grp) in [(1024, 16), (2**20, 64)]:
        analyze("constant_depth_GHZ", create_constant_depth_ghz, ghz_sizes, max_amp, max_grp)
        analyze("long_range_CNOT", create_dynamic_CNOT_circuit, cnot_sizes, max_amp, max_grp)
