from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.classical import expr, types
from qiskit.circuit import Instruction
from typing import List, Dict, Tuple
from collections import defaultdict

from analysis.dag import *

def add_measurement_redundancy(circuit: QuantumCircuit, N: int, M: int) -> QuantumCircuit:
    """
    Adds redundancy to mid-circuit measurements (MCMs) by:
    1. Finding MCMs in the circuit using DAG analysis
    2. Adding N ancilla qubits for each MCM (up to M MCMs)
    3. Entangling ancillas with measured qubits
    4. Adding majority voting (odd N) or AND (even N) to conditional operations
    
    Args:
        circuit: Input quantum circuit
        N: Number of ancilla qubits to add per MCM
        M: Maximum number of MCMs to modify (0 means no modification)
        
    Returns:
        QuantumCircuit: Modified circuit with redundant measurements
    """
    if M <= 0:
        return circuit
        
    # Convert to DAG and find MCMs
    dag = DAG(circuit)
    mcms: List[Tuple[int, List[int]]] = []
    measured_qubits = set()  # Track unique measured qubits
    
    # Find MCMs by checking for measurements with successors
    for node in dag.nodes():
        instr = dag.get_node_instr(node)
        if hasattr(instr.operation, 'name') and instr.operation.name == 'measure':
            successors = list(dag.successors(node))
            if successors:
                measured_qubits.add(instr.qubits[0])  # Add measured qubit to set
                mcms.append((node, successors))
    
    if not mcms or M <= 0:
        return circuit.copy()
    
    # Create new circuit with ancillas
    num_mcms = min(len(mcms), M)
    qr_anc = QuantumRegister(N * len(measured_qubits), 'anc')
    cr_ancs = [ClassicalRegister(N+1, f'cr_anc_{i}') for i in range(num_mcms)]

    regs = []
    # Add original quantum and classical registers
    for qreg in circuit.qregs:
        regs.append(qreg)
    for creg in circuit.cregs:
        regs.append(creg)
    # Add new registers
    regs.append(qr_anc)
    regs.extend(cr_ancs)
    
    new_circ = QuantumCircuit(*regs)
    
    # Track MCM modifications
    processed_mcms = 0
    #mcm_to_ancillas: Dict[int, List[int]] = {}
    qubit_to_ancillas = {}  # Maps measured qubit to its ancilla indices
    mcm_to_creg = {}  # Maps MCM node to its classical register
    
    # Process circuit instructions
    for idx, instr in enumerate(circuit.data):
        op, qargs, cargs = instr
        
        # Check if current instruction is an MCM to modify
        mcm_match = next((m for m in mcms if m[0] == idx and mcms.index(m) < M), None)

        if mcm_match:
            measured_qubit = qargs[0]

            # Get or assign ancilla qubits for this measured qubit
            if measured_qubit not in qubit_to_ancillas:
                anc_start = len(qubit_to_ancillas) * N
                qubit_to_ancillas[measured_qubit] = list(range(anc_start, anc_start + N))
            
            # Reset ancillas if they were used before
            for anc_idx in qubit_to_ancillas[measured_qubit]:
                new_circ.reset(qr_anc[anc_idx])
            
            # Store mapping of MCM to its classical register
            mcm_to_creg[idx] = cr_ancs[processed_mcms]
            
            # Add redundant measurements
            new_cargs = [cr_ancs[processed_mcms][0]]
            for i, anc_idx in enumerate(qubit_to_ancillas[measured_qubit]):
                new_circ.cx(measured_qubit, qr_anc[anc_idx])
                new_circ.measure(qr_anc[anc_idx], cr_ancs[processed_mcms][i+1])
            
            new_instr = CircuitInstruction(op, qargs, new_cargs)
            new_circ.append(new_instr)
            processed_mcms += 1
           
            
        # Modify conditional operations
        elif isinstance(op, Instruction) and op.name == 'if_else':

            condition_reg = op.condition[0]
            condition_val = op.condition[1]
            
            for m_idx, successors in mcms:
                 if m_idx in mcm_to_creg and condition_reg == circuit.data[m_idx].clbits[0]:
                    mcm_node = m_idx
                    break

            #new_circ.append(op, qargs, cargs)

            if mcm_node in mcm_to_creg:
                creg = mcm_to_creg[mcm_node]
                if N % 2 == 0: # Majority voting for even N
                    votes = [expr.lift(creg[i]) for i in range(N+1)]
                    majority = _create_majority_voting(votes)
                    new_condition = expr.bit_and(op.condition[0], majority)
                else:  # AND for odd N
                    votes = [expr.lift(creg[i]) for i in range(0, N+1)]                    
                    # Check if all votes are 1
                    all_ones = votes[0]
                    for vote in votes[1:]:
                        all_ones = expr.bit_and(all_ones, vote)
                    
                    # Check if all votes are 0
                    all_zeros = expr.bit_not(votes[0])
                    for vote in votes[1:]:
                        all_zeros = expr.bit_and(all_zeros, expr.bit_not(vote))
                    
                    # Final condition: (all ones OR all zeros) AND original condition
                    unanimous = expr.bit_or(all_ones, all_zeros)
                    new_condition = unanimous
                
                new_op = Instruction(name="if_else", num_qubits=op.num_qubits, num_clbits=creg.size, params=op.params)
                new_op.condition = new_condition
                new_circ.append(new_op, qargs, creg)
        else:
            new_circ.append(op, qargs, cargs)
    
    return new_circ

def _create_majority_voting(votes: List) -> expr:
    """Creates a majority voting expression from a list of vote expressions"""
    from itertools import combinations
    
    threshold = (len(votes) + 1) // 2
    majority = False
    
    for r in range(threshold, len(votes) + 1):
        for combo in combinations(votes, r):
            term = True
            for vote in combo:
                term = expr.bit_and(term, vote)
            majority = expr.bit_or(majority, term)
    
    return majority