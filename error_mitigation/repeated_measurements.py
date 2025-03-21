from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import CircuitInstruction, Measure
from collections import Counter

from helpers.dag import *

import networkx as nx


def add_redundant_measurements(circuit: QuantumCircuit, N: int = 2) -> tuple[QuantumCircuit, list[ClassicalRegister]]:
    dag = DAG(circuit)
    new_cregs = []
    counter = 0

    dag.remove_cregs()

    for node in nx.topological_sort(dag):
        instr = dag.get_node_instr(node)
        qubits = instr.qubits
        if isinstance(instr.operation, Measure):
            print(qubits)
            creg = ClassicalRegister(N + 1, name=f"m_{qubits[0]._index}_{counter}")
            
            #op.clbits = creg[0]
            #new_inst = Instruction(op)

            counter += 1
            new_cregs.append(creg)

            dag.add_creg(creg)

            print(list(dag.successors(node)))

            print(new_inst)
            exit()
            


    # Identify all measurement nodes
    measurement_nodes = [node for node in dag.nodes() if isinstance(node, DAGOpNode) and node.op.name == "measure"]

    for i, node in enumerate(measurement_nodes):
        qubit = node.qargs[0]
        creg = ClassicalRegister(N + 1, name=f"m_{qubit._index}_{i}")
        new_cregs.append(creg)

        dag.add_creg(creg) 

        tmp_dag = DAGCircuit()
        reg = QuantumRegister(size=1, name=node.qargs[0]._register.name)
        tmp_dag.add_qreg(reg)
        tmp_dag.add_creg(creg)
        
        # Insert N new measurements immediately after the original one
        for j in range(N + 1):
            tmp_dag.apply_operation_back(node.op.copy(), reg, [creg[j]]) 

        print(tmp_dag.qubits, tmp_dag.clbits)

        dag.substitute_node_with_dag(node, tmp_dag, propagate_condition=True)
        print(dag_to_circuit(dag))
        exit()
        #dag.remove_op_node(node)
        

    return dag_to_circuit(dag), new_cregs


def majority_vote_counts(raw_counts: dict[str, int]) -> dict[str, int]:
    """Processes raw measurement counts where each qubit has three redundant measurements.
    Returns a new counts dictionary with corrected values using majority voting."""
    corrected_counts = {}

    for bitstring, count in raw_counts.items():
        bit_groups = bitstring.split()  # Split on whitespace to get 3-bit groups
        corrected_bits = [Counter(group).most_common(1)[0][0] for group in bit_groups]  # Majority vote

        corrected_bitstring = "".join(corrected_bits)  # Reconstruct bitstring with spaces
        corrected_counts[corrected_bitstring] = corrected_counts.get(corrected_bitstring, 0) + count

    return corrected_counts