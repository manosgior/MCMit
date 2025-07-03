import mthree
from qiskit import QuantumCircuit
import numpy as np
import random

from analysis.dag import DAG

def apply_stochastic_branching(circuit: QuantumCircuit, calibration_matrix: list[np.array]) -> QuantumCircuit:
    """
    Modifies conditions in if_test operations based on calibration-derived flip probabilities.
    
    Args:
        circuit: Input quantum circuit with conditional operations
        calibration_matrix: Calibration data for measurement errors
        
    Returns:
        QuantumCircuit: Modified circuit with stochastically updated conditions
    """
    new_circuit = circuit.copy()
    
    # Create DAG to find MCM dependencies
    dag = DAG(circuit)
    
    # Map conditional operations to their controlling MCMs
    mcm_to_conditionals = {}
    for node in dag.nodes():
        instr = dag.get_node_instr(node)
        if hasattr(instr.operation, 'name') and instr.operation.name == 'measure':
            for succ in dag.successors(node):
                succ_instr = dag.get_node_instr(succ)
                if hasattr(succ_instr.operation, 'condition'):
                    mcm_to_conditionals[succ] = instr.qargs[0].index
    
    # Process all instructions
    for idx, instr in enumerate(new_circuit.data):
        if hasattr(instr.operation, 'condition') and instr.operation.condition:
            # Get condition register and value
            cond_reg, cond_val = instr.operation.condition
            
            # Get the qubit index of the MCM this condition depends on
            mcm_qubit_idx = mcm_to_conditionals.get(idx)
            if mcm_qubit_idx is None:
                continue  # Skip if we can't find the controlling MCM
            
            # Get appropriate flip probability based on current condition value
            flip_probs = get_bitflip_probabilities(calibration_matrix, mcm_qubit_idx)
            flip_prob = flip_probs[cond_val]  # Use P(1|0) for 0, P(0|1) for 1
            
            # Apply stochastic flip
            new_val = stochastic_flip(cond_val, flip_prob)
            
            # Update condition with potentially flipped value
            instr.operation.condition = (cond_reg, new_val)
    
    return new_circuit


def get_bitflip_probabilities(calibration_matrix: list[np.array], index_qubit: int) -> float:   
    return (calibration_matrix[index_qubit][0][1], calibration_matrix[index_qubit][1][0])

def compute_bitflip_probabilities(qubits: list[int], mode: int, calibration_matrix: list[np.array]):
    probs = [get_bitflip_probabilities(calibration_matrix, q)[mode] for q in qubits]

    return np.prod(probs)

def stochastic_flip(value: int, flip_prob: float) -> int:
    """
    Flips a binary value (0->1 or 1->0) with given probability.
    
    Args:
        value: Binary value (0 or 1) to potentially flip
        flip_prob: Probability of flipping the value (0.0 to 1.0)
        
    Returns:
        int: Either the original value or its flip, based on probability
    """
    if not isinstance(value, int) or value not in [0, 1]:
        raise ValueError("Value must be binary (0 or 1)")
    if not 0 <= flip_prob <= 1:
        raise ValueError("Probability must be between 0 and 1")
        
    # Generate random number and compare with flip probability
    if random.random() < flip_prob:
        return 1 - value  # Flip the bit
    return value

def compute_calibrations_from_backend(circuit: QuantumCircuit, backend, filename: str = "backends/calibrations/calibrations.json"):
    mit = mthree.M3Mitigation(backend)
    mit.cals_from_system(mthree.utils.final_measurement_mapping(circuit))
    mit.cals_to_file(filename)

    return mit.single_qubit_cals

def fetch_calibrations_from_file(filename: str):
    mit = mthree.M3Mitigation()
    mit.cals_from_file(filename)
    return mit.single_qubit_cals