"""
evaluate_feedback_latency.py
=============================
Fig. 10 driver (§8.2), simulation counterpart: impact of classical feedback
latency on constant-depth GHZ and long-range CNOT fidelity, modelling ONLY
the thermal-relaxation (T1/T2) error accrued while qubits idle during
classical feedback -- no gate or readout noise, matching the paper's "for
the sake of fairness" setup.

Model: an application executes N instances of the benchmark circuit; each
conditional (feedback) operation stalls all qubits for the controller's
feedback latency. The N instances' idle windows all act on the same payload
qubits, so we insert one delay of N x latency ns after every conditional and
attach a thermal-relaxation error of exactly that duration to it.

Default latencies (ns) come from Table 2's measured controller feedback:
MCMit's branch_reduce_fproc is constant-latency (205 ns) regardless of input
size; Qubic's scales with the XOR width (445 ns at 16 measurement inputs).

Note: both circuit generators only support odd qubit counts, so the paper's
"16-qubit" configuration runs at 17 qubits (nearest valid size).

Output: results/feedback_latency_impact_16q.csv schema
    Benchmark,N,Method,Fidelity

Example
-------
    python -m evaluation.evaluate_feedback_latency --output results/feedback_latency_impact_16q.csv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

from applications.constant_depth_GHZ import create_constant_depth_ghz, get_perfect_ghz_distribution
from applications.long_range_CNOT import create_dynamic_CNOT_circuit, get_perfect_distribution_long_range_cnot
from evaluation.fidelity import fidelity
from evaluation.distribution_processing import process_distribution_ghz, process_distribution_long_range_cnot

T1_S = 170e-6
T2_S = 130e-6


def add_feedback_delays(circuit: QuantumCircuit, delay_ns: float) -> QuantumCircuit:
    """Insert an all-qubit delay after every conditional (feedback) operation."""
    new_qc = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)
    for instr in circuit.data:
        new_qc.append(instr)
        if getattr(instr.operation, "condition", None) is not None:
            new_qc.delay(delay_ns, new_qc.qubits, unit="ns")
    return new_qc


def idle_only_noise_model(delay_ns: float) -> NoiseModel:
    """Thermal relaxation of exactly `delay_ns` attached to delay instructions.
    (Aer does not scale the error by the actual delay length, so the model
    must be rebuilt for each delay duration used.)"""
    nm = NoiseModel()
    err = thermal_relaxation_error(T1_S, T2_S, delay_ns * 1e-9)
    nm.add_all_qubit_quantum_error(err, ["delay"])
    return nm


def evaluate(circuit: QuantumCircuit, perfect: dict, post, n_instances: int,
             latency_ns: float, shots: int) -> float:
    total_delay_ns = n_instances * latency_ns
    delayed = add_feedback_delays(circuit, total_delay_ns)
    sim = AerSimulator(noise_model=idle_only_noise_model(total_delay_ns))
    tqc = transpile(delayed, sim, optimization_level=1)
    counts = sim.run(tqc, shots=shots).result().get_counts()
    return fidelity(perfect, post(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nqubits", type=int, default=16)
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--latency-mcmit", type=float, default=205.0, help="ns per feedback event (Table 2)")
    parser.add_argument("--latency-qubic", type=float, default=445.0, help="ns per feedback event (Table 2, 16-input XOR)")
    parser.add_argument("--instances", type=int, nargs="+", default=[10, 50, 100, 250, 500, 750, 1000])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Both circuit generators only support odd qubit counts; a "16-qubit"
    # request runs at 17 (nearest valid size).
    n = args.nqubits if args.nqubits % 2 == 1 else args.nqubits + 1
    benchmarks = [
        ("Constant-depth GHZ", create_constant_depth_ghz(n),
         get_perfect_ghz_distribution(n, args.shots), process_distribution_ghz),
        ("Long-range CNOT", create_dynamic_CNOT_circuit(n),
         get_perfect_distribution_long_range_cnot(args.shots), process_distribution_long_range_cnot),
    ]
    methods = [("MCMit", args.latency_mcmit), ("Qubic", args.latency_qubic)]

    rows = []
    for bench_label, qc, perfect, post in benchmarks:
        for method, latency in methods:
            for n in args.instances:
                fid = evaluate(qc, perfect, post, n, latency, args.shots)
                rows.append({"Benchmark": bench_label, "N": n, "Method": method, "Fidelity": fid})
                print(f"{bench_label},{n},{method},{fid:.4f}", flush=True)

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Benchmark", "N", "Method", "Fidelity"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
