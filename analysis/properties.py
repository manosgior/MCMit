from qiskit.circuit import QuantumCircuit

def countGates(circuit: QuantumCircuit, gates: list[str]):
    pass

def countNonLocalGates(circuit: QuantumCircuit):
    return sum(1 for gate, qargs, _ in circuit.data if len(qargs) == 2)

def countRemoteGates(circuit: QuantumCircuit, remote_couplings: list[tuple]):
    pass

def countMeasurements(circuit: QuantumCircuit):
    measurements = [(op, qargs) for op, qargs, _ in circuit.data if op.name == "measure"]
    
    return len(measurements)

def getDepth(circuit: QuantumCircuit):
    pass

def getSize(circuit: QuantumCircuit):
    return circuit.num_qubits
    