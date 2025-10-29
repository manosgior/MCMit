from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import CircuitInstruction, Measure
from qiskit.primitives import BitArray
from qiskit.circuit.classical import expr

from collections import Counter, defaultdict
from itertools import product

from analysis.dag import *

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
        if hasattr(instr, 'condition'):
            to_fix_ops.append(node)
 
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

def add_redundant_measurements_with_logic(circuit: QuantumCircuit, N: int = 2) -> QuantumCircuit:
    if N + 1 != 3:
        # The nested if_test logic below is hardcoded for d=3 (N=2)
        raise ValueError("This implementation currently only supports N=2 (d=3 majority vote).")
    
    d = N + 1 # Total measurements

    # 1. Find all cbits that are used as conditions (these are from MCMs)
    mcm_cbits = set()
    for instr in circuit.data:
        if hasattr(instr.operation, 'condition'):
            for bit in instr.clbits:
                mcm_cbits.add(bit) # Add the Clbit object
                print(bit)

    # 2. Build a new circuit and a map for re-wiring
    new_qc = QuantumCircuit(*circuit.qregs, *circuit.cregs, name="mcm_voted_circ")
    
    # Map from {old_mcm_cbit: new_majority_cbit}
    cbit_remap = {} 
    mcm_counter = 0

    # 3. Iterate through the original circuit's instructions
    for instr in circuit.data:
        op = instr.operation
        qubits = instr.qubits
        cbits = instr.clbits
        
        # --- Check if this is an MCM we need to replace ---
        is_mcm = isinstance(op, Measure) and cbits and cbits[0] in mcm_cbits
        
        if is_mcm:
            # --- REPLACE with repetition code and majority vote ---
            original_qbit = qubits[0]
            original_cbit = cbits[0]
            
            # If we've already created logic for this bit, skip (shouldn't happen)
            if original_cbit in cbit_remap:
                continue 

            # 1. Add new registers for this MCM
            creg_d = ClassicalRegister(d, name=f"m_{original_qbit.index}_{mcm_counter}")
            creg_maj = ClassicalRegister(1, name=f"maj_{original_qbit.index}_{mcm_counter}")
            new_qc.add_register(creg_d, creg_maj)
            mcm_counter += 1
            
            # 2. Add 'd' sequential measurements
            for i in range(d):
                new_qc.measure(original_qbit, creg_d[i])
            
            # 3. Add d=3 majority vote logic using nested if_test
            # This logic computes: maj_bit = (c0 & c1) | (c0 & c2) | (c1 & c2)
            # We initialize the majority bit to 0
            new_qc.store(creg_maj[0], 0)
            
            # if (c0 == 1):
            with new_qc.if_test(expr.equal(creg_d[0], 1)):
                # if (c1 == 1):
                with new_qc.if_test(expr.equal(creg_d[1], 1)):
                    # c0=1, c1=1 -> majority is 1 (done)
                    new_qc.store(creg_maj[0], 1)
                # else (c1 == 0):
                with new_qc.if_test(expr.equal(creg_d[1], 0)):
                    # if (c2 == 1):
                    with new_qc.if_test(expr.equal(creg_d[2], 1)):
                        # c0=1, c1=0, c2=1 -> majority is 1
                        new_qc.store(creg_maj[0], 1)
            # else (c0 == 0):
            with new_qc.if_test(expr.equal(creg_d[0], 0)):
                # if (c1 == 1):
                with new_qc.if_test(expr.equal(creg_d[1], 1)):
                    # if (c2 == 1):
                    with new_qc.if_test(expr.equal(creg_d[2], 1)):
                        # c0=0, c1=1, c2=1 -> majority is 1
                        new_qc.store(creg_maj[0], 1)
            
            # 4. Store the remapping for later ops
            cbit_remap[original_cbit] = creg_maj[0]
            
        elif op.condition:
            # --- This is a conditional op, we must update its condition ---
            cond_reg, val = op.condition
            
            # Check if the bit it depends on has been remapped
            if cond_reg in cbit_remap:
                new_cbit = cbit_remap[cond_reg]
                # Apply the op with the *new* condition
                new_qc.append(instr.replace(condition=(new_cbit, val)))
            else:
                # This op's condition was not from an MCM, append as is
                new_qc.append(instr)
        else:
            # --- Not an MCM, not conditional -> append as is ---
            new_qc.append(instr)
            
    return new_qc

def majority_vote(bitstring: str) -> str:
    return Counter(bitstring).most_common(1)[0][0] 

def cleanup_bitstrings_per_creg(bitstrings: list[str]) -> list[str]:
    return "".join([majority_vote(bitstring) for bitstring in bitstrings])

def majority_vote_counts(raw_counts: dict[str, int]) -> dict[str, int]:
    """Processes raw measurement counts where each qubit has three redundant measurements.
    Returns a new counts dictionary with corrected values using majority voting."""
    corrected_counts = {}

    for bitstring, count in raw_counts.items():
        bit_groups = bitstring.split()  # Split on whitespace to get 3-bit groups
        corrected_bits = [majority_vote(group) for group in bit_groups]  # Majority vote

        corrected_bitstring = "".join(corrected_bits)  # Reconstruct bitstring with spaces
        corrected_counts[corrected_bitstring] = corrected_counts.get(corrected_bitstring, 0) + count

    return corrected_counts

def majority_vote_counts_separate_cregs(list_bitstings: list[list[str]]) -> dict[str, int]:
    reduced = defaultdict(int)

    for i in range(len(list_bitstings[0])):
        correct_bitstring = ''
        for j in range(len(list_bitstings)):
            correct_bitstring += list_bitstings[j][i]
        reduced[correct_bitstring] += 1
    return dict(reduced)

