from backends.backend import loadBackend, getRealEagleBackend
from backends.simulator import simulatorFromBackend

from analysis.properties import countNonLocalGates, countMeasurements

from qiskit.circuit.random import random_circuit
from qiskit.circuit import QuantumCircuit
from qiskit import transpile


def getMCMBinaryOPratio(circuit: QuantumCircuit, offset: int = 8) -> float:
    binary_ops = countNonLocalGates(circuit)
    measurements = countMeasurements(circuit) - offset

    if binary_ops == 0:
        return float('inf')
    return measurements / binary_ops

def randomCircuits(n: int, num_qubits: int, depth: int, conditional: bool = False, reset: bool = False) -> list[QuantumCircuit]:
    to_return = []

    for i in range(n):
        qc = random_circuit(num_qubits = num_qubits, max_operands = 2, measure = True, depth = depth, conditional = conditional, reset = reset)
        to_return.append(qc)

    return to_return


circs = randomCircuits(n=30, num_qubits=10, depth=5, conditional=True, reset=True)   

backend = getRealEagleBackend()
simulator = simulatorFromBackend(backend)

tqcs = [transpile(c, backend=backend, optimization_level=3) for c in circs]
sorted_tqcs = sorted(tqcs, key=getMCMBinaryOPratio)

for c in sorted_tqcs:
    print(getMCMBinaryOPratio(c))

