from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.classical import expr

def create_teleportation_circuit() -> QuantumCircuit:
    """
    Creates a quantum circuit that implements quantum teleportation protocol.
    The circuit teleports the state of qubit 0 to qubit 2.
    
    Returns:
        QuantumCircuit: The teleportation circuit
    """
    # Create registers
    qr = QuantumRegister(3, 'q')  # 3 qubits: sender, auxiliary, receiver
    cr1 = ClassicalRegister(2, 'c')  # 2 classical bits for measurements
    cr2 = ClassicalRegister(1, 'final')  # Final measurement of the receiver qubit
    qc = QuantumCircuit(qr, cr1, cr2)
    
    # Create an arbitrary state to teleport (can be modified)
    qc.h(0)  # Creates |+⟩ state
    #qc.rz(0.5, 0)  # Add some phase
    
    # Create Bell pair between auxiliary and receiver qubits
    qc.h(1)
    qc.cx(1, 2)
    
    # Teleportation protocol
    qc.cx(0, 1)  # Apply CNOT between sender and auxiliary
    qc.h(0)      # Apply H gate to sender
    
    # Measure sender and auxiliary qubits
    qc.measure(0, cr1[0])  # First qubit measurement
    qc.measure(1, cr1[1])  # Second qubit measurement
    
    # Apply corrections based on measurements
    with qc.if_test((cr1[1], 1)):  # If second measurement is 1
        qc.x(2)
    with qc.if_test((cr1[0], 1)):  # If first measurement is 1
        qc.z(2)

    qc.measure(2, cr2)  # Measure the receiver qubit to see the teleported state
    
    return qc

def create_repeated_teleportation_circuit(n_teleports: int = 1) -> QuantumCircuit:
    """
    Creates a quantum circuit that implements multiple quantum teleportations.
    The state is teleported back and forth between qubits 0 and 2.
    
    Args:
        n_teleports: Number of times to teleport the state (default: 1)
    Returns:
        QuantumCircuit: The teleportation circuit
    """
    # Create registers
    qr = QuantumRegister(3, 'q')  # 3 qubits: sender/receiver1, auxiliary, sender/receiver2
    cr1 = ClassicalRegister(2*n_teleports, 'c')  # 2 classical bits per teleportation
    cr2 = ClassicalRegister(1, 'final')  # Final measurement of the receiver qubit
    qc = QuantumCircuit(qr, cr1, cr2)
    
    # Create initial state to teleport
    qc.h(0)  # Creates |+⟩ state
    #qc.rz(0.5, 0)  # Add some phase
    
    for i in range(n_teleports):
        # Determine direction of teleportation
        source = 0 if i % 2 == 0 else 2
        target = 2 if i % 2 == 0 else 0
        
        # Create Bell pair between auxiliary and target qubits
        qc.h(1)
        qc.cx(1, target)
        
        # Teleportation protocol
        qc.cx(source, 1)
        qc.h(source)
        
        # Measure source and auxiliary qubits
        qc.measure(source, cr1[2*i])
        qc.measure(1, cr1[2*i + 1])
        
        # Apply corrections based on measurements
        with qc.if_test((cr1[2*i + 1], 1)):
            qc.x(target)
        with qc.if_test((cr1[2*i], 1)):
            qc.z(target)
        
        # Reset source and auxiliary qubits for next teleportation
        qc.reset([source, 1])

        qc.measure(target, cr2)  # Measure the target qubit to see the teleported state
    
    return qc