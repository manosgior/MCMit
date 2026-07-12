import mthree
from qiskit import QuantumCircuit
from qiskit.circuit import Clbit, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit.circuit.classical.expr import Expr, Var, Unary, Binary
import numpy as np
import random

# Binary boolean combinators for which "resample a leaf independently" is well-defined:
# each operand contributes to the result without any implicit extra negation, so recursing
# into both sides at the *same* polarity and reassembling with the same operator is correct.
# (Comparison/arithmetic ops -- EQUAL, LESS, ADD, etc. -- are deliberately excluded: flipping
# a bit inside e.g. an arithmetic sum being compared to a threshold isn't a leaf-negation, so
# those subtrees are left untouched by the fallback at the bottom of _resample_condition_expr.)
_POLARITY_PRESERVING_BINARY_CTORS = {
    Binary.Op.BIT_XOR: expr.bit_xor,
    Binary.Op.BIT_AND: expr.bit_and,
    Binary.Op.BIT_OR: expr.bit_or,
    Binary.Op.LOGIC_AND: expr.logic_and,
    Binary.Op.LOGIC_OR: expr.logic_or,
}


def apply_stochastic_branching(circuit: QuantumCircuit, calibration_matrix: list[np.array]) -> QuantumCircuit:
    """
    Modifies conditions in if_test operations based on calibration-derived flip probabilities.

    Each call independently redraws every flip (via `stochastic_flip`), so calling this once
    per shot yields the per-shot-resampled circuit population described in the paper (§7.3):
    across N calls, a given single-MCM condition is left as measured in a (1-p) fraction of
    the returned circuits and flipped in a p fraction, with no extra shots introduced.

    Simple conditions -- a single classical bit written by exactly one preceding `measure` --
    are flipped directly. Compound boolean conditions (an `Expr` built from several measured
    bits via XOR/AND/OR, e.g. the parity/majority-vote conditions produced by measurement
    hardening) are handled by *resampling each contributing bit independently* (at that bit's
    own qubit-specific flip probability) and re-evaluating the same expression tree on the
    resampled bits -- see `_resample_condition_expr`. This is the physically-faithful
    generalisation of the simple case (which is just the one-leaf special case of the same
    procedure), is linear in the number of operands (no branch/subcircuit blow-up), and
    handles any boolean combinator rather than needing a separate closed-form formula for
    each one (e.g. the XOR-specific parity-flip formula does not generalise to majority-vote).
    Non-boolean conditions (a multi-bit ClassicalRegister compared to an integer, or exotic
    comparison/arithmetic Exprs) are still left untouched, since a "flip a leaf" operation
    isn't well-defined for those.

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

        if isinstance(condition, tuple) and len(condition) == 2:
            cond_target, cond_val = condition

            if isinstance(cond_target, Clbit):
                clbit = cond_target
            elif isinstance(cond_target, ClassicalRegister) and len(cond_target) == 1:
                clbit = cond_target[0]
            else:
                continue  # multi-bit register condition -- not a single leaf, skip

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

        elif isinstance(condition, Expr):
            op.condition = _resample_condition_expr(condition, clbit_to_qubit_idx, calibration_matrix)

    return new_circuit


def _resample_condition_expr(node: Expr, clbit_to_qubit_idx: dict, calibration_matrix: list[np.array],
                              negate: bool = False) -> Expr:
    """
    Recursively resamples a boolean classical-Expr condition, one measured leaf at a time.

    `negate` tracks the node's polarity in the tree (how many enclosing NOTs surround it, mod
    2). A bare `Var(clbit)` leaf reached at positive polarity is structurally "querying for 1"
    (same convention as a plain `(clbit, 1)` condition -- of which this is the k=1 special
    case), so its flip is drawn from P(1|0); at negative polarity it's "querying for 0", drawn
    from P(0|1). Flipping a leaf = toggling whether that specific occurrence is wrapped in one
    more NOT (equivalently: simulating that this qubit's measurement came out the other way),
    which composes correctly through XOR/AND/OR regardless of nesting depth.

    Leaves whose controlling qubit is unknown, and any non-boolean-combinator subtree (a
    `Value`, or a comparison/arithmetic `Binary`), are returned unchanged.
    """
    if isinstance(node, Var) and isinstance(node.var, Clbit):
        clbit = node.var
        mcm_qubit_idx = clbit_to_qubit_idx.get(clbit)
        if mcm_qubit_idx is None:
            return node

        flip_probs = get_bitflip_probabilities(calibration_matrix, mcm_qubit_idx)
        cond_val = 0 if negate else 1
        flip_prob = flip_probs[cond_val]

        if random.random() < flip_prob:
            return expr.logic_not(node)
        return node

    if isinstance(node, Unary) and node.op == Unary.Op.LOGIC_NOT:
        new_operand = _resample_condition_expr(node.operand, clbit_to_qubit_idx, calibration_matrix,
                                                negate=not negate)
        if isinstance(new_operand, Unary) and new_operand.op == Unary.Op.LOGIC_NOT:
            return new_operand.operand  # NOT(NOT(x)) -> x
        return expr.logic_not(new_operand)

    if isinstance(node, Binary) and node.op in _POLARITY_PRESERVING_BINARY_CTORS:
        ctor = _POLARITY_PRESERVING_BINARY_CTORS[node.op]
        new_left = _resample_condition_expr(node.left, clbit_to_qubit_idx, calibration_matrix, negate)
        new_right = _resample_condition_expr(node.right, clbit_to_qubit_idx, calibration_matrix, negate)
        return ctor(new_left, new_right)

    return node


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