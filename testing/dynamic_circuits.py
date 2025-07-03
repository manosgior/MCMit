from applications.long_range_CNOT import get_dynamic_CNOT_circuit
from applications.constant_depth_GHZ import create_constant_depth_ghz
from applications.quantum_teleportation import *

from backends.backend import getRealEagleBackend

from analysis.dag import *
from analysis.properties import *

from compiler.decoding.adaptive_soft_decoding import *

from qiskit import QuantumCircuit, transpile

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


def test_constant_depth_GHZ(min: int, max: int) -> list[QuantumCircuit]:
    """
    Test the constant depth GHZ circuit generation for a range of qubit counts.
    
    Args:
        limit (int): The maximum number of qubits to test.
    
    Returns:
        list[QuantumCircuits]: A list of generated quantum circuits.
    """
    circuits = []
    for num_qubit in range(min, max + 1, 2):
        circuit = create_constant_depth_ghz(num_qubit)
        circuits.append(circuit)
    return circuits

def test_quantum_teleportation(min: int, max: int) -> QuantumCircuit:
    """
    Test the quantum teleportation circuit generation.
    
    Returns:
        QuantumCircuit: The generated quantum teleportation circuit.
    """
    circuits = []
    for i in range(min, max + 1):
        circuits.append(create_repeated_teleportation_circuit(i))

    return circuits

qcs = test_constant_depth_GHZ(19, 21)
#qc = qcs[0]
#print(qc)

backend = getRealEagleBackend()
#tqc = transpile(qc, backend)
#print(tqc)
#circuit = add_parity_checks(tqc, backend.coupling_map)

circuits = [add_parity_checks_greedy(qc, backend) for qc in qcs]
for c in circuits:
    print(c)
