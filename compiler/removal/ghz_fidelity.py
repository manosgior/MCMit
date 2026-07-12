"""GHZ state fidelity: WITH mid-circuit measurements (the original constant-depth
dynamic circuit) vs WITHOUT (an MCM-free unitary preparation of the same GHZ state).

Noiseless: both should be 1.0 (the optimized circuit reproduces the exact state).
Under noise: the MCM version is constant-depth but pays measurement/reset/feed-forward
error; the MCM-free version is a deeper unitary (CNOT ladder). We compute the exact
state fidelity F = <GHZ|rho|GHZ> via a density-matrix simulation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "applications"))

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from constant_depth_GHZ import create_constant_depth_ghz


def ideal_ghz(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return Statevector(qc)


def mcm_free_ghz_circuit(n):
    """Textbook MCM-free GHZ prep: H + CNOT ladder (linear depth)."""
    qr = QuantumRegister(n, "q")
    qc = QuantumCircuit(qr)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def with_mcm_ghz_circuit(n):
    """Original constant-depth dynamic circuit, terminal read-out stripped."""
    qc = create_constant_depth_ghz(n)
    return qc.remove_final_measurements(inplace=False)


def make_noise(p1, p2, pm):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["u", "rx", "ry", "rz", "h", "x", "z", "sx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx", "cz"])
    if pm > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(pm, 1), ["reset"])
        nm.add_all_qubit_quantum_error(depolarizing_error(pm, 1), ["measure"])
    return nm


def density_matrix(qc, noise):
    sim = AerSimulator(method="density_matrix", noise_model=noise)
    tqc = transpile(qc, sim, basis_gates=["u", "cx", "reset", "measure", "if_else"])
    tqc.save_density_matrix()
    rho = sim.run(tqc).result().data()["density_matrix"]
    return DensityMatrix(rho)


def main():
    # (1q depol, 2q depol, meas/reset depol)
    NOISE = {
        "noiseless": (0.0, 0.0, 0.0),
        "light  (1q .05%, 2q .5%, M 1%)": (5e-4, 5e-3, 1e-2),
        "heavy  (1q .2%, 2q 2%, M 3%)": (2e-3, 2e-2, 3e-2),
    }
    for label, (p1, p2, pm) in NOISE.items():
        nm = make_noise(p1, p2, pm) if (p1 or p2 or pm) else None
        print(f"\n=== noise: {label} ===")
        print(f"{'n':>3} | {'F  WITH MCMs':>13} | {'F  WITHOUT MCMs':>16}")
        print("-" * 42)
        for n in (5, 7, 9, 11):
            ghz = ideal_ghz(n)
            try:
                f_with = state_fidelity(ghz, density_matrix(with_mcm_ghz_circuit(n), nm))
            except Exception as e:
                f_with = f"err:{type(e).__name__}"
            f_without = state_fidelity(ghz, density_matrix(mcm_free_ghz_circuit(n), nm))
            fw = f"{f_with:.4f}" if isinstance(f_with, float) else f_with
            print(f"{n:>3} | {fw:>13} | {f_without:>16.4f}")


if __name__ == "__main__":
    main()
