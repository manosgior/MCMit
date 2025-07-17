from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.quantum_info import Statevector, Pauli

import numpy as np

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
    phi_x = np.pi / 8
    phi_z = 3 * np.pi / 8

    qc.rx(2 * phi_x, 0)
    qc.rz(2 * phi_z, 0)
    
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
    phi_x = np.pi / 8
    phi_z = 3 * np.pi / 8

    qc.rx(2 * phi_x, 0)
    qc.rz(2 * phi_z, 0)
    
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

def create_ladder_teleportation_circuit(n_teleports: int = 1) -> QuantumCircuit:
    """
    Creates a quantum circuit that teleports the |i+> state across n_steps
    in a ladder fashion: 0 -> 2 -> 4 -> ... -> 2n.

    Args:
        n_steps (int): Number of teleportation steps.

    Returns:
        QuantumCircuit: The complete teleportation ladder circuit.
    """
    num_qubits = 2 * n_teleports + 1
    qr = QuantumRegister(num_qubits, 'q')
    cr1 = ClassicalRegister(2 * n_teleports, 'c')  # 2 bits per teleportation step
    cr2 = ClassicalRegister(1, 'final')  # Final measurement of the receiver qubit
    qc = QuantumCircuit(qr, cr1, cr2)

    # Step 1: Prepare |i+> on qubit 0
    phi_x = np.pi / 8
    phi_z = 3 * np.pi / 8

    qc.rx(2 * phi_x, 0)
    qc.rz(2 * phi_z, 0)

    # Teleportation steps
    for step in range(n_teleports):
        sender = 2 * step
        ent_a = sender + 1
        receiver = sender + 2

        # Create Bell pair between ent_a and receiver
        qc.h(ent_a)
        qc.cx(ent_a, receiver)

        # Bell measurement between sender and ent_a
        qc.cx(sender, ent_a)
        qc.h(sender)

        # Measure sender and ent_a
        qc.measure(sender, 2 * step)
        qc.measure(ent_a, 2 * step + 1)

        # Conditional corrections on receiver
        # Apply corrections based on measurements
        with qc.if_test((cr1[2*step + 1], 1)):
            qc.x(receiver)
        with qc.if_test((cr1[2*step], 1)):
            qc.z(receiver)
        #qc.x(receiver).c_if(cr1, 1 << (2 * step + 1))  # X if ent_a bit == 1
        #qc.z(receiver).c_if(cr1, 1 << (2 * step))      # Z if sender bit == 1

    qc.measure(-1, cr2)  # Measure the receiver qubit to see the teleported state

    return qc

def generate_repeated_teleportations(max_reps: int, is_ladder: bool = False) -> list[QuantumCircuit]:
    """
    Generates a list of quantum circuits for repeated teleportation.
    
    Args:
        max_reps: Maximum number of teleportations to generate circuits for
    Returns:
        List[QuantumCircuit]: List of teleportation circuits
    """
    circuits = []
    for i in range(1, max_reps + 1):
        if is_ladder:
            circuits.append(create_ladder_teleportation_circuit(i))
        else:
            circuits.append(create_repeated_teleportation_circuit(i))
    return circuits

def get_perfect_expectation_value(circuit: QuantumCircuit) -> float:
    state = Statevector(circuit)
    Z_op = Pauli('Z') # Pauli Z operator
    Y_op = Pauli('Z') # Pauli Z operator
    X_op = Pauli('Z') # Pauli Z operator

    z_expectation_value = state.expectation_value(Z_op)
    y_expectation_value = state.expectation_value(Y_op)
    x_expectation_value = state.expectation_value(X_op)

    return (z_expectation_value, y_expectation_value, x_expectation_value)

def get_perfect_expectation_value_hardcoded(shots: int = 0) -> float:
    return 0.7071067811865475
