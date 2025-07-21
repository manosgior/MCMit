import qiskit.qasm3 as qasm3

def export_qasm3(circuit, filename):
    """
    Exports a Qiskit circuit to a QASM 3 file.

    Args:
        circuit (QuantumCircuit): The quantum circuit to export.
        filename (str): The name of the file to save the QASM 3 code.
    """
    with open(filename, 'w') as file:
       qasm3.dump(circuit, file)