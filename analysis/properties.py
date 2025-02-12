from qiskit.circuit import QuantumCircuit

def countGates(circuit: QuantumCircuit, gates: list[str]):
    pass

def countNonLocalGates(circuit: QuantumCircuit, gates: list[str] = ["cx", "ecr", "cz", "rzz"]):
    pass

def countRemoteGates(circuit: QuantumCircuit, remote_couplings: list[tuple]):
    pass

def getDepth(circuit: QuantumCircuit):
    pass

def getSize(circuit: QuantumCircuit):
    return circuit.num_qubits
    