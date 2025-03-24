from qiskit.circuit import QuantumCircuit

def countGates(circuit: QuantumCircuit, gates: list[str]):
    pass

def countNonLocalGates(circuit: QuantumCircuit):
    return sum(1 for gate, qargs, _ in circuit.data if len(qargs) == 2)

def countRemoteGates(circuit: QuantumCircuit, remote_couplings: list[tuple]):
    pass

def getMeasurements(circuit: QuantumCircuit, include_final: bool = True) -> list:
    circ_copy = circuit.copy()
  
    if not include_final:
        circ_copy.remove_final_measurements()
   
    measurements = [(op, qargs) for op, qargs, _ in circ_copy.data if op.name == "measure"]    
    
    return measurements

def getDepth(circuit: QuantumCircuit):
    pass

def getSize(circuit: QuantumCircuit):
    return circuit.num_qubits
    