from applications.long_range_CNOT import get_dynamic_CNOT_circuit
from applications.constant_depth_GHZ import create_constant_depth_ghz
from applications.quantum_teleportation import *

from qiskit import QuantumCircuit

def test_long_range_CNOT(limit: int) -> list[QuantumCircuit]:
    """
    Test the long-range CNOT circuit generation for a range of qubit counts.
    
    Args:
        limit (int): The maximum number of qubits to test.
    
    Returns:
        list[QuantumCircuits]: A list of generated quantum circuits.
    """
    circuits = []
    for num_qubit in range(7, limit + 1, 2):
        circuit = get_dynamic_CNOT_circuit(num_qubit)
        circuits.append(circuit)
    return circuits


def test_constant_depth_GHZ(limit: int) -> list[QuantumCircuit]:
    """
    Test the constant depth GHZ circuit generation for a range of qubit counts.
    
    Args:
        limit (int): The maximum number of qubits to test.
    
    Returns:
        list[QuantumCircuits]: A list of generated quantum circuits.
    """
    circuits = []
    for num_qubit in range(7, limit + 1, 2):
        circuit = create_constant_depth_ghz(num_qubit)
        circuits.append(circuit)
    return circuits

def test_quantum_teleportation(reps: int) -> QuantumCircuit:
    """
    Test the quantum teleportation circuit generation.
    
    Returns:
        QuantumCircuit: The generated quantum teleportation circuit.
    """
    circuits = []
    for i in range(1, reps + 1):
        circuits.append(create_repeated_teleportation_circuit(i))

    return circuits

qcs = test_quantum_teleportation(5)
for qc in qcs:
    print(qc)