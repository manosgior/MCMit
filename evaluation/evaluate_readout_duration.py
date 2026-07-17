"""
evaluate_readout_duration.py
=============================
Fig. 12 driver (§8.3): impact of readout duration on the fidelity of the
repeated and ladder quantum-teleportation benchmarks, modelling ONLY the
thermal-relaxation (T1/T2) error accrued while qubits idle during each
mid-circuit measurement window -- no gate or readout-bitflip noise, matching
the paper's "for the sake of fairness" setup.

Model: every mid-circuit measurement stalls all qubits for the readout
duration; an all-qubit delay of that duration is inserted after each MCM,
with a thermal-relaxation error of exactly that duration attached to it.
Sweeping the duration (250/500/750/1000 ns) isolates how much fidelity a
faster discriminator (shorter readout window) buys.

Output: results/readout_duration_teleportation_fidelity.csv schema
    Benchmark,N,Method,Fidelity     (Method = readout duration, e.g. "250ns")

Example
-------
    python -m evaluation.evaluate_readout_duration --output results/readout_duration_teleportation_fidelity.csv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

from applications.quantum_teleportation import (
    create_repeated_teleportation_circuit,
    create_ladder_teleportation_circuit,
    get_perfect_distribution_teleportation,
)
from evaluation.fidelity import fidelity
from evaluation.distribution_processing import process_distribution_teleportation

T1_S = 170e-6
T2_S = 130e-6


def add_readout_delays(circuit: QuantumCircuit, delay_ns: float) -> QuantumCircuit:
    """Insert an all-qubit delay after every mid-circuit measurement.
    A measurement is mid-circuit if any non-measure/barrier op follows it."""
    last_active = -1
    for i, instr in enumerate(circuit.data):
        if instr.operation.name not in ("measure", "barrier", "delay"):
            last_active = i

    new_qc = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)
    for i, instr in enumerate(circuit.data):
        new_qc.append(instr)
        if instr.operation.name == "measure" and i < last_active:
            new_qc.delay(delay_ns, new_qc.qubits, unit="ns")
    return new_qc


def idle_only_noise_model(delay_ns: float) -> NoiseModel:
    nm = NoiseModel()
    err = thermal_relaxation_error(T1_S, T2_S, delay_ns * 1e-9)
    nm.add_all_qubit_quantum_error(err, ["delay"])
    return nm


def evaluate(circuit: QuantumCircuit, duration_ns: float, shots: int) -> float:
    delayed = add_readout_delays(circuit, duration_ns)
    sim = AerSimulator(noise_model=idle_only_noise_model(duration_ns))
    tqc = transpile(delayed, sim, optimization_level=1)
    counts = sim.run(tqc, shots=shots).result().get_counts()
    perfect = get_perfect_distribution_teleportation(shots)
    return fidelity(perfect, process_distribution_teleportation(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-n", type=int, default=7, help="Max teleportation steps (paper: 7)")
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--durations", type=int, nargs="+", default=[250, 500, 750, 1000])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    benchmarks = [
        ("Repeated teleportation", create_repeated_teleportation_circuit),
        ("Ladder teleportation", create_ladder_teleportation_circuit),
    ]

    rows = []
    for bench_label, build in benchmarks:
        for n in range(2, args.max_n + 1):  # circuit builders require n_teleports > 1
            qc = build(n)
            for dur in args.durations:
                fid = evaluate(qc, dur, args.shots)
                rows.append({"Benchmark": bench_label, "N": n, "Method": f"{dur}ns", "Fidelity": fid})
                print(f"{bench_label},{n},{dur}ns,{fid:.4f}", flush=True)

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Benchmark", "N", "Method", "Fidelity"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
