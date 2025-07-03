from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.classical import expr, types
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import CountOps
from qiskit.circuit import Instruction
from qiskit import transpile

from qiskit.circuit.library import CXGate
from qiskit.circuit import Measure

from typing import List, Dict, Set, Tuple
from collections import defaultdict
import random

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

def add_parity_checks_already_mapped(circuit: QuantumCircuit, coupling_map: CouplingMap) -> Tuple[QuantumCircuit, bool]:
    """
    Adds optimal number of ancilla qubits to a circuit with MCMs based on physical connectivity.
    Ancillas are added only if they can be connected to two program qubits without requiring SWAP gates.
    
    Args:
        circuit: Input quantum circuit with MCMs
        coupling_map: Physical connectivity of the quantum device
        
    Returns:
        QuantumCircuit: Modified circuit with added ancilla qubits and their measurements
    """
    
    # 1. Convert coupling map to graph
    physical_graph = nx.Graph()
    physical_graph.add_edges_from(coupling_map.get_edges())
    
    # 2. Find initial mapping of program qubits to physical qubits
    # This assumes the circuit is already mapped to the device
    program_to_physical = {}
    physical_to_program = {}

    if hasattr(circuit, 'layout'):
        initial_layout = circuit.layout.initial_index_layout(filter_ancillas=True)

        for i in range(len(initial_layout)):
            program_to_physical[i] = initial_layout[i]
            physical_to_program[initial_layout[i]] = i
  
    # 3. Find available physical qubits adjacent to program qubits
    program_locations = set(program_to_physical.values())
    available_ancillas = _find_available_neighbors(physical_graph, program_locations)

    if not available_ancillas:
        return circuit.copy(), False
    
    # 4. Create new circuit with ancilla qubits
    num_ancillas = len(available_ancillas)
    qr_anc = QuantumRegister(num_ancillas, 'anc')
    cr_anc = ClassicalRegister(num_ancillas, 'cr_anc')
    
    regs = []
    # Add original quantum and classical registers
    for qreg in circuit.qregs:
        regs.append(qreg)
    for creg in circuit.cregs:
        regs.append(creg)

    # Add new registers
    regs.append(qr_anc)
    regs.extend(cr_anc)

    new_circ = QuantumCircuit(*regs)
    
    # 5. Copy original circuit
    new_circ.data.extend(circuit.data)
    
    # 6. Add ancilla operations
    for i, (_, program_neighbors) in enumerate(available_ancillas):
        # Get corresponding program qubits
        prog_qubit1 = physical_to_program[program_neighbors[0]]
        prog_qubit2 = physical_to_program[program_neighbors[1]]
        
        # Create GHZ-like state between program qubits and ancilla
        #new_circ.h(qr_anc[i])
        new_circ.cx(circuit.qubits[prog_qubit1], qr_anc[i])
        new_circ.cx(circuit.qubits[prog_qubit2], qr_anc[i])
        
        # Measure ancilla
        new_circ.measure(qr_anc[i], cr_anc[i])
        
        # Reset ancilla for potential reuse
        new_circ.reset(qr_anc[i])
    
    return new_circ, True

def add_parity_checks_greedy(circuit: QuantumCircuit, backend, max_attempts: int = 10):
    initial_circuit = transpile(circuit, backend, optimization_level=3)
    initial_cx_count = count_two_qubit_gates(initial_circuit)
    
    working_circuit = circuit.copy()
    successful_additions = 0

    while True:
        best_circuit = None
        best_cx_overhead = float('inf')
        
        # Try adding one ancilla with different qubit pairs
        for _ in range(max_attempts):
            # Create a copy for this attempt
            test_circuit = working_circuit.copy()
            
            # Add new ancilla register
            qr_anc = QuantumRegister(1, f'anc_{successful_additions}')
            cr_anc = ClassicalRegister(1, f'cr_anc_{successful_additions}')
            test_circuit.add_register(qr_anc, cr_anc)
            
            # Create DAG to identify final measurements
            dag = DAG(test_circuit)
            final_measure_pos = -1

            # Find first final measurement (measurement with no successors)
            for idx, instr in enumerate(test_circuit.data):
                if (instr.operation.name == 'measure' and 
                    not list(dag.successors(idx))):
                    final_measure_pos = idx
                    break

            # Select random consecutive program qubits
            n_qubits = len(working_circuit.qubits) - successful_additions
            start_idx = random.randint(0, n_qubits - 2)
            q1, q2 = start_idx, start_idx + 1
            
            new_data = []
            for i, instr in enumerate(test_circuit.data):
                if i == final_measure_pos:
                    # Insert ancilla operations before first final measurement
                    new_data.append(CircuitInstruction(CXGate(), [test_circuit.qubits[q1], qr_anc[0]], []))
                    new_data.append(CircuitInstruction(CXGate(), [test_circuit.qubits[q2], qr_anc[0]], []))
                    new_data.append(CircuitInstruction(Measure(), [qr_anc[0]], [cr_anc[0]]))
                new_data.append(instr)
            
            test_circuit.data = new_data
            
            # Transpile and count CX gates
            mapped_circuit = transpile(test_circuit, backend)
            cx_count = count_two_qubit_gates(mapped_circuit) - 2  # Subtract the two CNOTs used to entangle ancilla
            
            # Calculate overhead
            cx_overhead = cx_count - initial_cx_count
            
            # Update best if this attempt has less overhead
            if cx_overhead < best_cx_overhead:
                best_cx_overhead = cx_overhead
                best_circuit = test_circuit
        
        # If no acceptable solution found, stop
        if best_circuit is None or best_cx_overhead > initial_cx_count * 0.3:  # 30% threshold
            break
            
        # Accept this addition and continue
        working_circuit = best_circuit
        successful_additions += 1
    
    return working_circuit

def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    """
    Counts all two-qubit gates in a circuit.
    
    Args:
        circuit: Input quantum circuit
        
    Returns:
        int: Number of two-qubit gates
    """
    count = 0
    for instr in circuit.data:
        if len(instr.qubits) == 2:  # Any instruction with 2 qubits is a 2-qubit gate
            count += 1
    return count

def _find_available_neighbors(
    physical_graph: nx.Graph,
    program_locations: Set[int],
    min_connections: int = 2
) -> List[Tuple[int, List[int]]]:
    """
    Finds physical qubits that can serve as ancillas based on connectivity.
    
    Args:
        physical_graph: Device coupling graph
        program_locations: Physical locations of program qubits
        min_connections: Minimum required connections to program qubits
        
    Returns:
        List of tuples (ancilla_location, [connected_program_qubits])
    """
    available = []
    for node in physical_graph.nodes():
        if node not in program_locations:
            neighbors = list(physical_graph.neighbors(node))
            program_neighbors = [n for n in neighbors if n in program_locations]
            if len(program_neighbors) >= min_connections:
                available.append((node, program_neighbors))
    return available


    