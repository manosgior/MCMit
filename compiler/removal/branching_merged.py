import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "applications"))

from BranchingPropagation import BranchingConstantPropagation as BCP

from constant_depth_GHZ import create_constant_depth_ghz
from long_range_CNOT import create_dynamic_CNOT_circuit
from quantum_teleportation import (
    create_repeated_teleportation_circuit as REP,
    create_ladder_teleportation_circuit as LAD,
)


def time_branching(qc, limit_s=20.0):
    """Time the exponential engine; returns (seconds, leaves_paths) or (None, None) if too big."""
    import signal
    t0 = time.perf_counter()
    try:
        tree = BCP.optimize(qc, max_nodes=2_000_000)
    except RuntimeError:
        return None, None
    return time.perf_counter() - t0, None


def main():
    for name, builder, rng in [
        ("constant_depth_GHZ", create_constant_depth_ghz, range(5, 26, 2)),
        ("long_range_CNOT", create_dynamic_CNOT_circuit, range(5, 26, 2)),
        ("REPEATED_teleport", REP, range(5, 11)),
        ("LADDER_teleport", LAD, range(5, 11)),
    ]:
        print(f"\n{'='*92}\n{name}\n{'='*92}", flush=True)
        print(f"{'n':>3} {'nq':>3} | {'MCM_elim':>8} {'subcirc':>7} {'recon(state/dist)':>17} "
              f"{'passes':>6} | {'merged_t_s':>10} {'branch_t_s':>11} {'speedup':>8}", flush=True)
        print("-" * 92, flush=True)
        for n in rng:
            qc = builder(n)
            t0 = time.perf_counter()
            rep = BCP.optimize_merged(qc)
            tm = time.perf_counter() - t0

            # exponential engine, only while it is still cheap enough to be fair
            tb, _ = (time_branching(qc) if n <= 17 else (None, None))
            sp = f"{tb/tm:>7.0f}x" if tb else "   n/a"
            tb_s = f"{tb:>11.3f}" if tb else "        ---"
            recon = f"{str(rep['verified_reconverge_state']):>5}/{str(rep['verified_reconverge_dist']):<5}"
            sub = rep["subcircuits"] if rep["subcircuits"] is not None else "?"
            print(f"{n:>3} {qc.num_qubits:>3} | {rep['mcm_eliminated']:>8} {str(sub):>7} {recon:>17} "
                  f"{rep['passes']:>6} | {tm:>10.4f} {tb_s} {sp:>8}", flush=True)


if __name__ == "__main__":
    main()
