import mthree
from qiskit import QuantumCircuit
from qiskit.circuit import Clbit, ClassicalRegister
import numpy as np
import random

def apply_stochastic_branching(circuit: QuantumCircuit, calibration_matrix: list[np.array]) -> QuantumCircuit:
    """
    Modifies conditions in if_test operations based on calibration-derived flip probabilities.

    Each call independently redraws every flip (via `stochastic_flip`), so calling this once
    per shot yields the per-shot-resampled circuit population described in the paper (§7.3):
    across N calls, a given single-MCM condition is left as measured in a (1-p) fraction of
    the returned circuits and flipped in a p fraction, with no extra shots introduced.

    Only conditions on a single classical bit written by exactly one preceding `measure` are
    handled. Compound conditions (a multi-bit ClassicalRegister, or an `Expr` built from
    several measured bits, e.g. the XOR/majority-vote conditions produced by measurement
    hardening) are intentionally left untouched -- the paper's per-branch formula only covers
    independent single-MCM branches, and the right per-shot flip semantics for a *compound*
    condition (flip the whole boolean outcome? resample each contributing bit independently
    and re-evaluate the expression?) is an open design question, not yet decided here.

    Args:
        circuit: Input quantum circuit with conditional operations
        calibration_matrix: Calibration data for measurement errors

    Returns:
        QuantumCircuit: Modified circuit with stochastically updated conditions
    """
    new_circuit = circuit.copy()

    # Map each classical bit to the qubit index of the most recent `measure` that wrote it.
    # (Not a qubit-adjacency DAG lookup: the qubit a condition acts on is essentially always
    # different from the qubit that was measured -- that's the point of feed-forward -- so a
    # qubit-based dependency graph never finds this edge. The actual dependency runs through
    # the classical bit, which is what we track directly here.)
    clbit_to_qubit_idx: dict[Clbit, int] = {}

    for instr in new_circuit.data:
        op = instr.operation

        if op.name == 'measure':
            for qubit, clbit in zip(instr.qubits, instr.clbits):
                clbit_to_qubit_idx[clbit] = new_circuit.find_bit(qubit).index
            continue

        condition = getattr(op, 'condition', None)
        if not condition:
            continue

        if not (isinstance(condition, tuple) and len(condition) == 2):
            # An `Expr` condition (e.g. XOR/majority-vote) rather than a plain
            # (bit_or_register, value) tuple -- compound, see docstring. Skip.
            continue

        cond_target, cond_val = condition

        if isinstance(cond_target, Clbit):
            clbit = cond_target
        elif isinstance(cond_target, ClassicalRegister) and len(cond_target) == 1:
            clbit = cond_target[0]
        else:
            # Multi-bit register or `Expr` condition -- compound, see docstring. Skip.
            continue

        mcm_qubit_idx = clbit_to_qubit_idx.get(clbit)
        if mcm_qubit_idx is None:
            continue  # Skip if we can't find the controlling MCM

        # Get appropriate flip probability based on current condition value
        flip_probs = get_bitflip_probabilities(calibration_matrix, mcm_qubit_idx)
        flip_prob = flip_probs[cond_val]  # P(0|1) for cond_val=0, P(1|0) for cond_val=1

        # Apply stochastic flip
        new_val = stochastic_flip(cond_val, flip_prob)

        # Update condition with potentially flipped value
        op.condition = (cond_target, new_val)

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