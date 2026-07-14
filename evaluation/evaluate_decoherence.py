"""
evaluate_decoherence.py
-----------------------
Analytically models the decoherence impact of repeated circuit execution for
two 32-qubit applications:
  - Constant-depth GHZ state preparation
  - Long-range CNOT

N = number of sequential circuit repetitions (instances) in the application.
Each repetition incurs a total classical feedback overhead of `latency` ns
during which all qubits are idle.  Over N repetitions, total idle time is:

    total_idle_ns = N × latency_per_instance_ns

Fidelity accounts for the number of qubits whose state matters during feedback
(n_affected), using a product-of-survival model:

    p_per_qubit = idleError(total_idle_s, T1, T2)
    fidelity    = (1 − p_per_qubit) ^ n_affected

Circuit-specific n_affected (at 32 qubits):
  - Constant-depth GHZ : n_qubits / 2 = 16  (every other qubit is a data qubit;
    odd qubits are measured mid-circuit and reset, so only even qubits
    accumulate relevant decoherence during classical feedback phases)
  - Long-range CNOT    : 2  (only qubit 0 (control) and qubit n−1 (target)
    hold state that matters for the final answer; all ancilla qubits are
    measured and discarded before the feedback corrections are applied)

This model collapses to the calibrated single-qubit formula (1 − p) when
n_affected = 1, and naturally makes GHZ decohere faster than CNOT.

No quantum simulation is required.

Output: results/feedback_latency_impact_32q.csv  (same schema as feedback_latency_impact_16q.csv)
"""

import csv
import numpy as np
from analysis.fidelity import idleError

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

N_QUBITS = 32   # circuit size

# Total per-instance classical feedback latency (ns).
# MCMit achieves constant-time processing; Qubic has longer branch-selection
# overhead. These represent the *aggregate* idle window per circuit execution.
LATENCY_MCMIT = 205.0   # ns
LATENCY_QUBIC = 669.0   # ns

# Realistic qubit T1/T2 (SOTA IBM Eagle/Heron generation, median values).
T1_S = 170e-6   # seconds
T2_S = 130e-6   # seconds  (T2 ≤ 2·T1, physically valid)

# Instance sweep
N_VALUES = [10, 50, 100, 250, 500, 750, 1000]

OUTPUT_CSV = "results/feedback_latency_impact_32q.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Analytical decoherence fidelity
# ──────────────────────────────────────────────────────────────────────────────

def decoherence_fidelity(n_instances: int,
                         latency_ns: float,
                         n_affected: int = 1,
                         t1: float = T1_S,
                         t2: float = T2_S) -> float:
    """
    Fidelity = (1 − p_per_qubit)^n_affected, where p_per_qubit is the thermal
    relaxation error for a single qubit over N sequential instances.

    Parameters
    ----------
    n_instances : N — number of sequential circuit repetitions
    latency_ns  : total per-instance feedback latency (ns)
    n_affected  : number of data qubits that hold state during feedback
    t1, t2      : qubit coherence times (seconds)
    """
    total_idle_s = n_instances * latency_ns * 1e-9
    p_per_qubit  = idleError(total_idle_s, t1, t2)
    fidelity     = (1.0 - p_per_qubit) ** n_affected
    return float(max(0.0, fidelity))

# ──────────────────────────────────────────────────────────────────────────────
# Main sweep
# ──────────────────────────────────────────────────────────────────────────────

# Benchmark definitions: (name, n_affected_qubits)
# GHZ:  every other qubit is a data qubit → n_qubits / 2 = 16
# CNOT: only the control (q0) and target (q_{n-1}) hold state → 2
BENCHMARKS = [
    ('Constant-depth GHZ', N_QUBITS // 2),
    ('Long-range CNOT',    2),
]

METHODS = {
    'MCMit': LATENCY_MCMIT,
    'Qubic': LATENCY_QUBIC,
}

def run_all():
    rows = []

    for bench_name, n_affected in BENCHMARKS:
        for method_name, base_latency in METHODS.items():
            for n in N_VALUES:
                fid = decoherence_fidelity(
                    n_instances=n,
                    latency_ns=base_latency,
                    n_affected=n_affected,
                )
                total_ns = n * base_latency
                print(
                    f"  [{bench_name:25s}] {method_name}  N={n:4d}  "
                    f"n_affected={n_affected:2d}  total_idle={total_ns:8.1f} ns  fidelity={fid:.4f}"
                )
                rows.append({
                    'Benchmark': bench_name,
                    'N':         n,
                    'Method':    method_name,
                    'Fidelity':  fid,
                })

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Benchmark', 'N', 'Method', 'Fidelity'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to: {OUTPUT_CSV}")


if __name__ == '__main__':
    print(f"Analytical decoherence model — {N_QUBITS} qubits")
    print(f"  T1 = {T1_S*1e6:.0f} µs   T2 = {T2_S*1e6:.0f} µs")
    print(f"  LATENCY_MCMIT = {LATENCY_MCMIT} ns   LATENCY_QUBIC = {LATENCY_QUBIC} ns\n")
    run_all()
