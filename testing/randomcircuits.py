from backends.backend import loadBackend, getRealEagleBackend
from backends.simulator import simulatorFromBackend, getNoiselessSimulator

from analysis.properties import countNonLocalGates, countMeasurements
from analysis.fidelity import fidelity

from error_mitigation.repeated_measurements import add_redundant_measurements, majority_vote_counts

from qiskit.circuit.random import random_circuit
from qiskit.circuit import QuantumCircuit
from qiskit import transpile


def getMCMBinaryOPratio(circuit: QuantumCircuit, offset: int = 10) -> float:
    binary_ops = countNonLocalGates(circuit)
    measurements = countMeasurements(circuit) - offset

    if binary_ops == 0:
        return float('inf')
    return binary_ops, measurements, measurements / binary_ops

def randomCircuits(n: int, num_qubits: int, depth: int, conditional: bool = False, reset: bool = False) -> list[QuantumCircuit]:
    to_return = []

    for i in range(n):
        qc = random_circuit(num_qubits = num_qubits, max_operands = 2, measure = True, depth = depth, conditional = conditional, reset = reset, seed=1)
        to_return.append(qc)

    return to_return

circs = randomCircuits(n=1, num_qubits=5, depth=8, conditional=True, reset=True) 

c = circs[0]
print(c)

tqc = transpile(c, getNoiselessSimulator())
perfect_counts = getNoiselessSimulator().run(tqc, shots=10000).result().get_counts()

fixed = add_redundant_measurements(c)[0]
print(fixed)

backend = getRealEagleBackend()
simulator = simulatorFromBackend(backend)
tqc = transpile(c, backend)
tqc_fixed = transpile(fixed, backend)

noisy_counts = simulator.run(tqc, shots=10000).result().get_counts()
noisy_counts_fixed = simulator.run(tqc_fixed, shots=10000).result().get_counts()
noisy_counts_fixed = majority_vote_counts(noisy_counts_fixed)

print(perfect_counts)
print(noisy_counts)
print(noisy_counts_fixed)

print(fidelity(perfect_counts, noisy_counts))
print(fidelity(perfect_counts, noisy_counts_fixed))

exit()


