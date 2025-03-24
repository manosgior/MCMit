from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import CircuitInstruction, Measure
from collections import Counter

from helpers.dag import *

import networkx as nx
import matplotlib.pyplot as plt

def convert_int_to_qreg(value: int) -> int:
    result = []
    bit_position = 0
    while value > 0:
        if value & 1:
            result.append(bit_position)
        value >>= 1
        bit_position += 1
    return result

def add_redundant_measurements(circuit: QuantumCircuit, N: int = 2) -> QuantumCircuit:
    dag = DAG(circuit)
    counter = 0
    to_fix_ops = []

    dag.remove_cregs()

    for node in nx.topological_sort(dag):
        instr = dag.get_node_instr(node)
        if instr.condition is not None:
            to_fix_ops.append(node)
            print(instr.condition)
            print(convert_int_to_qreg(instr.condition[1]))
 
        qubits = instr.qubits

        if isinstance(instr.operation, Measure):
            prev = list(dag.predecessors(node))
            if len(prev) > 0:
                assert(len(prev) == 1)
                prev = prev[0]
                dag.remove_edge(prev, node)
            next = list(dag.successors(node))
            if len(next) > 0:
                assert(len(next) == 1)
                next = next[0]
                dag.remove_edge(node, next)            

            creg = ClassicalRegister(N + 1, name=f"m_{qubits[0]._index}_{counter}")
            counter += 1
            dag.add_creg(creg)
            
            for i in range(N + 1):
                new_inst = CircuitInstruction(instr.operation, instr.qubits, [creg[i]])
                id = dag.add_instr_node(new_inst)
                dag.add_edge(prev, id)
                prev = id
            
            if isinstance(next, int):
                dag.add_edge(prev, next)

            dag.remove_node(node)

            #for op in to_fix_ops:


    return dag.to_circuit()


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