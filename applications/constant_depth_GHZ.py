from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.classical import expr

def create_constant_depth_ghz(n_qubits: int) -> QuantumCircuit:
    """
    Create a constant-depth circuit for GHZ state preparation using the pattern:
    - H gates on even qubits (0,2,...)
    - CNOTs from even qubits to their neighbors
    - Measurements on odd qubits
    - Conditional operations based on measurements
    
    Args:
        n_qubits: Number of qubits in the GHZ state
    Returns:
        QuantumCircuit: Circuit that prepares the GHZ state
    """
    qr = QuantumRegister(n_qubits, 'q')
    cr1 = ClassicalRegister((n_qubits-1)//2, 'cr1')  # We need classical bits for odd qubits
    qc = QuantumCircuit(qr, cr1)
    
    # First layer: Apply H gates to even-numbered qubits
    for i in range(0, n_qubits, 2):
        qc.h(i)
    
    # Second layer: Apply CNOTs from even qubits to neighbors
    for i in range(0, n_qubits-1, 2):
        # CNOT to next qubit
        qc.cx(i, i+1)
        # CNOT to previous qubit if it exists and isn't the first qubit
        if i > 0:
            qc.cx(i, i-1)
 
    qc.cx(-1, -2)  # Last CNOT to ensure the last qubit is entangled

    # Third layer: Measure odd-numbered qubits
    for i in range(1, n_qubits, 2):
        qc.measure(i, cr1[i//2])

    # Fourth layer: Conditional operations with XOR patterns
    # First condition is simple measurement check
    with qc.if_test((cr1[0], 1)):
        qc.x(2)

    # Subsequent conditions use XOR of increasing measurements
    for i in range(3, n_qubits-1, 2):
        xor_expr = expr.lift(cr1[0])
        # XOR with subsequent measurements
        for j in range(1, i//2 + 1):
            xor_expr = expr.bit_xor(cr1[j], xor_expr)
        with qc.if_test((xor_expr)):
            qc.x(i+1)

    # Fifth layer: Reset measured qubits
    for i in range(1, n_qubits, 2):
        qc.reset(i)
    
    # Sixth layer: CNOTs between reset qubits
    for i in range(0, n_qubits-1, 2):
        qc.cx(i, i+1)

    qc.measure_all()
 
    return qc

def get_ghz_states(min: int = 5, max: int = 155) -> list[QuantumCircuit]:
    """
    Generate a list of constant-depth GHZ state circuits for a range of qubit counts.
    
    Args:
        min (int): Minimum number of qubits.
        max (int): Maximum number of qubits.
        
    Returns:
        list[QuantumCircuit]: List of generated GHZ circuits.
    """
    assert min % 2 == 1 and min >=5, "Both min is odd and we ask for at least 5 qubits."

    circuits = []
    for n_qubits in range(min, max + 1, 2):
        circuits.append(create_constant_depth_ghz(n_qubits))
    
    return circuits

def get_perfect_ghz_distribution(ghz_state_size: int, shots: int) -> dict[str, int]:
    return {'0' * ghz_state_size: int(shots / 2), '1' * ghz_state_size: int(shots / 2)}
