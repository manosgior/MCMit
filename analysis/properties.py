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

def get_measurements(circuit: QuantumCircuit, include_final: bool = True):
    """Returns a list of measurement operations in the circuit.
    If include_final is False, excludes measurements that are the last operations on their respective qubits."""
    measurements = [(op, qargs) for op, qargs, _ in circuit.data if op.name == "measure"]
    
    if not include_final:
        # Identify last operations per qubit
        last_ops = {}
        for op, qargs, _ in reversed(circuit.data):
            for q in qargs:
                if q.index not in last_ops:
                    last_ops[q.index] = op
        
        # Exclude measurements if they are the last operations on their qubits
        measurements = [(op, qargs) for op, qargs in measurements if any(last_ops[q.index] != op for q in qargs)]
    
    return measurements

def getDepth(circuit: QuantumCircuit):
    pass

def getSize(circuit: QuantumCircuit):
    return circuit.num_qubits
    