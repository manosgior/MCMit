"""
Branching-state extension of the constant-propagation pass (PROTOTYPE).

The shipped `ConstantPropagation` tracks a *single* deterministic symbolic
statevector.  When it meets a non-deterministic mid-circuit measurement it must
abandon the whole entangled group (`UnionTable.set_top`), because the
post-measurement state is a *mixture* of two branches a single statevector
cannot hold.  In circuits built around one entangled "payload" thread
(GHZ, long-range CNOT, teleportation) that caps elimination at 1 MCM.

This module explores the other lever discussed: instead of topping the group,
it *forks* on every non-deterministic measurement, propagates the projected
state of each outcome, resolves the downstream classical corrections inside
each branch (the controlling bit is now known), and **merges** branches that
reconverge to the same physical state.  Merging is what keeps the structure
small: a teleportation step fans out to 4 outcomes but the Pauli corrections
drive all 4 back to the same payload state, so they collapse to one node again.

The result is a `BranchTree`; `generate_instance` samples a concrete circuit
from it.  Every eliminable mid-circuit measurement is gone — only terminal
read-out measurements remain.

This file is additive: it does not modify `ConstantPropagation`.  It reuses the
existing sparse statevector engine (`QuState.QubitState`).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ControlledGate, Qubit, Clbit, ClassicalRegister
from qiskit.quantum_info import Operator
from qiskit.circuit.classical import expr

if __package__:
    from .QuState import QubitState, EPS
else:
    from QuState import QubitState, EPS

__all__ = ["BranchingConstantPropagation", "BranchTree"]

MEASURE = "measure"
RESET = "reset"
IF_ELSE = "if_else"
IGNORED = {"barrier", "delay", "id"}


# --------------------------------------------------------------------------- #
# Output representation: a tree of emit-ops with measurement-induced splits.
# --------------------------------------------------------------------------- #
@dataclass
class EmitOp:
    """A concrete operation to place into a sampled instance."""
    gate: Instruction
    qubits: Tuple[int, ...]
    clbits: Tuple[int, ...] = ()


@dataclass
class BranchTree:
    """A node of the optimized circuit.

    Internal node: `split` is set; a biased coin (probability `p_one` of the
    '1' outcome) selects which child continues.  When `child_one is child_zero`
    the coin has no effect on the emitted circuit — the measurement is fully
    removed and only its (discarded) randomness remains.

    Leaf node: `split` is False.  `leaf_state` is the exact tracked statevector
    of the branch at the point the terminal read-out block begins; the instance
    re-prepares it (correct by construction — no replayed protocol gates) and
    then applies `terminal_measures`.
    """
    split: bool = False
    p_one: float = 0.0
    child_one: Optional["BranchTree"] = None
    child_zero: Optional["BranchTree"] = None
    # leaf payload
    leaf_state: Optional["QubitState"] = None
    terminal_measures: List[Tuple[int, int]] = field(default_factory=list)  # (qubit, clbit)


# --------------------------------------------------------------------------- #
# Classical-condition helpers (cover tuple- and expr-style conditions).
# --------------------------------------------------------------------------- #
def _clbits_of_target(target) -> List[Clbit]:
    if isinstance(target, Clbit):
        return [target]
    if isinstance(target, ClassicalRegister):
        return list(target)
    return []


def _collect_cond_clbits(cond) -> List[Clbit]:
    if cond is None:
        return []
    if isinstance(cond, tuple):
        return _clbits_of_target(cond[0])
    out: List[Clbit] = []

    def walk(node):
        if isinstance(node, expr.Var):
            out.extend(_clbits_of_target(node.var))
        elif isinstance(node, expr.Cast):
            walk(node.operand)
        elif isinstance(node, expr.Unary):
            walk(node.operand)
        elif isinstance(node, expr.Binary):
            walk(node.left)
            walk(node.right)
    walk(cond)
    return out


def _reg_value(target, bits: Dict[Clbit, int]) -> int:
    cbs = _clbits_of_target(target)
    if isinstance(target, Clbit):
        return bits.get(target, 0)
    val = 0
    for i, cb in enumerate(cbs):
        val |= (bits.get(cb, 0) & 1) << i
    return val


def _eval_expr(node, bits: Dict[Clbit, int]) -> int:
    if isinstance(node, expr.Var):
        return _reg_value(node.var, bits)
    if isinstance(node, expr.Value):
        return int(node.value)
    if isinstance(node, expr.Cast):
        return _eval_expr(node.operand, bits)
    if isinstance(node, expr.Unary):
        v = _eval_expr(node.operand, bits)
        name = node.op.name
        if name == "BIT_NOT":
            return ~v & 1
        if name == "LOGIC_NOT":
            return int(v == 0)
        raise NotImplementedError(f"unary {name}")
    if isinstance(node, expr.Binary):
        a = _eval_expr(node.left, bits)
        b = _eval_expr(node.right, bits)
        name = node.op.name
        return {
            "BIT_AND": lambda: a & b,
            "BIT_OR": lambda: a | b,
            "BIT_XOR": lambda: a ^ b,
            "LOGIC_AND": lambda: int(bool(a) and bool(b)),
            "LOGIC_OR": lambda: int(bool(a) or bool(b)),
            "EQUAL": lambda: int(a == b),
            "NOT_EQUAL": lambda: int(a != b),
            "LESS": lambda: int(a < b),
            "GREATER": lambda: int(a > b),
            "LESS_EQUAL": lambda: int(a <= b),
            "GREATER_EQUAL": lambda: int(a >= b),
        }[name]()
    raise NotImplementedError(f"expr node {type(node)}")


def _cond_true(cond, bits: Dict[Clbit, int]) -> bool:
    if isinstance(cond, tuple):
        target, val = cond
        return _reg_value(target, bits) == val
    return _eval_expr(cond, bits) != 0


# --------------------------------------------------------------------------- #
# Matrices
# --------------------------------------------------------------------------- #
def _mat1(instr: Instruction) -> List[complex]:
    base = instr.base_gate if isinstance(instr, ControlledGate) else instr
    m = Operator(base).data
    return [complex(m[0, 0]), complex(m[0, 1]), complex(m[1, 0]), complex(m[1, 1])]


def _mat2(instr: Instruction) -> List[List[complex]]:
    m = Operator(instr).data
    return [[complex(m[r, c]) for c in range(4)] for r in range(4)]


def _apply_unitary(sv: QubitState, op: Instruction, q_inds: List[int]):
    if isinstance(op, ControlledGate):
        nc = op.num_ctrl_qubits
        controls, targets = q_inds[:nc], q_inds[nc:]
        if len(targets) == 1:
            sv.apply_gate(targets[0], _mat1(op), controls)
        else:
            sv.apply_two_qubit_gate(targets[0], targets[1], _mat2(op), controls)
    elif len(q_inds) == 1:
        sv.apply_gate(q_inds[0], _mat1(op))
    elif len(q_inds) == 2:
        sv.apply_two_qubit_gate(q_inds[0], q_inds[1], _mat2(op))
    else:
        raise NotImplementedError(f"{op.name} on {len(q_inds)} qubits")


def _project_inplace(sv: QubitState, q: int, outcome: bool) -> None:
    sv.state = {k: v for k, v in sv.state.items() if k[q] == outcome}
    sv.normalize()


def _sv_canon(sv: QubitState):
    """Canonical, global-phase-invariant key of a sparse statevector."""
    items = [(k, v) for k, v in sv.state.items() if abs(v) > EPS]
    if not items:
        return ()
    items.sort(key=lambda kv: kv[0])
    phase = items[0][1] / abs(items[0][1])
    return tuple((k, round((v / phase).real, 7), round((v / phase).imag, 7))
                 for k, v in items)


def _born_over(sv: QubitState, qubits: List[int]):
    """Distribution over the given qubits' joint outcome (as a bitstring)."""
    from collections import Counter
    dist = Counter()
    for k, v in sv.state.items():
        key = "".join("1" if k[q] else "0" for q in qubits)
        dist[key] += abs(v) ** 2
    return dict(dist)


class BranchingConstantPropagation:
    DEFAULT_MAX_NODES = 200_000

    # ----------------------------------------------------------------- #
    @classmethod
    def optimize(cls, circuit: QuantumCircuit, max_nodes: int = DEFAULT_MAX_NODES) -> BranchTree:
        data = list(circuit.data)
        n = circuit.num_qubits

        # qubit / clbit -> index
        qidx = {q: i for i, q in enumerate(circuit.qubits)}
        cidx = {c: i for i, c in enumerate(circuit.clbits)}

        # ---- terminal read-out block: index of the last *non-measurement* op.
        # A measurement at index j is terminal (kept as read-out) iff no
        # non-measurement operation occurs after it. ----
        last_nonmeas_idx = -1
        for i, inst in enumerate(data):
            if inst.operation.name.lower() not in (MEASURE, "barrier", "delay"):
                last_nonmeas_idx = i

        # ---- per-qubit last touch (any op acting on the qubit, incl. measures
        # and if_else inner gates).  A qubit whose last touch is a measurement
        # is "dead" afterwards and can be zeroed so branches reconverge. ----
        def touched_qubits(inst) -> List[int]:
            op = inst.operation
            if op.name == IF_ELSE:
                qs = set()
                for body in op.params:
                    if body is None:
                        continue
                    for inner in body.data:
                        for q in inner.qubits:
                            qs.add(qidx[inst.qubits[body.qubits.index(q)]])
                return list(qs)
            return [qidx[q] for q in inst.qubits]

        last_touch = [-1] * n
        for i, inst in enumerate(data):
            for q in touched_qubits(inst):
                last_touch[q] = i

        # ---- liveness of clbits: last index a clbit is *read* by a condition ----
        last_read = {}
        for i, inst in enumerate(data):
            if inst.operation.name == IF_ELSE:
                for cb in _collect_cond_clbits(inst.operation.condition):
                    last_read[cb] = i

        memo: Dict[Tuple, BranchTree] = {}
        leaf_memo: Dict[Tuple, BranchTree] = {}   # dedup leaves by (state, readout)
        node_count = [0]

        # ---- statevector canonicalisation (global-phase invariant) for merging --
        def sv_key(sv: QubitState):
            items = []
            for k, v in sv.state.items():
                if abs(v) > EPS:
                    items.append((k, v))
            if not items:
                return ()
            items.sort(key=lambda kv: kv[0])
            # divide out global phase from first amplitude
            phase = items[0][1] / abs(items[0][1])
            out = []
            for k, v in items:
                w = v / phase
                out.append((k, round(w.real, 8), round(w.imag, 8)))
            return tuple(out)

        def live_key(bits: Dict[Clbit, int], i: int):
            return tuple(sorted((cidx[cb], val) for cb, val in bits.items()
                                if last_read.get(cb, -1) >= i))

        # ---- apply a unitary instruction to the statevector + emit it ----
        def apply_unitary(sv: QubitState, op: Instruction, q_inds: List[int]):
            if isinstance(op, ControlledGate):
                nc = op.num_ctrl_qubits
                controls = q_inds[:nc]
                targets = q_inds[nc:]
                if len(targets) == 1:
                    sv.apply_gate(targets[0], _mat1(op), controls)
                else:
                    sv.apply_two_qubit_gate(targets[0], targets[1], _mat2(op), controls)
            elif len(q_inds) == 1:
                sv.apply_gate(q_inds[0], _mat1(op))
            elif len(q_inds) == 2:
                sv.apply_two_qubit_gate(q_inds[0], q_inds[1], _mat2(op))
            else:
                raise NotImplementedError(f"{op.name} on {len(q_inds)} qubits")

        def project(sv: QubitState, q: int, outcome: bool) -> QubitState:
            ns = sv.clone()
            ns.state = {k: v for k, v in sv.state.items() if k[q] == outcome}
            ns.normalize()
            return ns

        def zero_if_dead(sv: QubitState, q: int, j: int) -> None:
            """If qubit q is never touched again, reset it to |0> in the tracker
            so branches that differ only in this dead qubit's value reconverge."""
            if last_touch[q] <= j and sv.probability_measure_one(q) > 1 - EPS:
                sv.apply_gate(q, [0, 1, 1, 0])

        # ---- recursive build ----
        def build(i: int, sv: QubitState, bits: Dict[Clbit, int]) -> BranchTree:
            key = (i, sv_key(sv), live_key(bits, i))
            if key in memo:
                return memo[key]
            node_count[0] += 1
            if node_count[0] > max_nodes:
                raise RuntimeError("branch tree exceeded max_nodes (no reconvergence?)")

            node = BranchTree()
            memo[key] = node

            j = i
            while j < len(data):
                # entered the terminal read-out block -> snapshot state as a leaf.
                # Dedup leaves by (final state, read-out): distinct outcome paths
                # that reconverge to the same state become ONE subcircuit.
                if j > last_nonmeas_idx:
                    tmeas = []
                    for inst in data[j:]:
                        if inst.operation.name.lower() == MEASURE:
                            tmeas.append((qidx[inst.qubits[0]], cidx[inst.clbits[0]]))
                    lkey = (sv_key(sv), tuple(tmeas))
                    if lkey in leaf_memo:
                        memo[key] = leaf_memo[lkey]
                        return leaf_memo[lkey]
                    node.leaf_state = sv.clone()
                    node.terminal_measures = tmeas
                    leaf_memo[lkey] = node
                    return node

                inst = data[j]
                op = inst.operation
                name = op.name.lower()
                q_inds = [qidx[q] for q in inst.qubits]

                if name in IGNORED:
                    j += 1
                    continue

                if name == IF_ELSE:
                    taken = inst.operation.condition
                    body = op.params[0] if _cond_true(taken, bits) else (
                        op.params[1] if len(op.params) > 1 else None)
                    if body is not None:
                        for inner in body.data:
                            outer_q = [qidx[inst.qubits[body.qubits.index(q)]] for q in inner.qubits]
                            apply_unitary(sv, inner.operation, outer_q)
                    j += 1
                    continue

                if name == MEASURE:
                    q = q_inds[0]
                    cb = inst.clbits[0]
                    p1 = sv.probability_measure_one(q)
                    if p1 < EPS:
                        bits = {**bits, cb: 0}
                    elif p1 > 1 - EPS:
                        bits = {**bits, cb: 1}
                        zero_if_dead(sv, q, j)
                    else:
                        # genuine fork: project each outcome, resolve corrections per
                        # branch, then zero the measured qubit if it is now dead so
                        # the branches can reconverge (merge).
                        node.split = True
                        node.p_one = p1
                        sv1 = project(sv, q, True); zero_if_dead(sv1, q, j)
                        sv0 = project(sv, q, False); zero_if_dead(sv0, q, j)
                        node.child_one = build(j + 1, sv1, {**bits, cb: 1})
                        node.child_zero = build(j + 1, sv0, {**bits, cb: 0})
                        return node
                    j += 1
                    continue

                if name == RESET:
                    q = q_inds[0]
                    p1 = sv.probability_measure_one(q)
                    if p1 > 1 - EPS:           # definitely |1> -> X to |0>
                        sv.apply_gate(q, [0, 1, 1, 0])
                    elif p1 > EPS:             # superposed: fork, discard outcome, zero it
                        node.split = True
                        node.p_one = p1
                        sv1 = project(sv, q, True); sv1.apply_gate(q, [0, 1, 1, 0])
                        sv0 = project(sv, q, False)
                        node.child_one = build(j + 1, sv1, bits)
                        node.child_zero = build(j + 1, sv0, bits)
                        return node
                    j += 1
                    continue

                # plain unitary
                apply_unitary(sv, op, q_inds)
                j += 1

            # no terminal block (no trailing measures) -> leaf with current state
            lkey = (sv_key(sv), ())
            if lkey in leaf_memo:
                memo[key] = leaf_memo[lkey]
                return leaf_memo[lkey]
            node.leaf_state = sv.clone()
            leaf_memo[lkey] = node
            return node

        sv0 = QubitState(n)
        return build(0, sv0, {})

    # ----------------------------------------------------------------- #
    @staticmethod
    def _prepare_state(out: QuantumCircuit, sv: QubitState):
        """Append gates that prepare `sv` from |0...0>.

        Fast path for product states (the common case after measurements have
        collapsed everything except a single payload qubit): prepare each qubit
        independently.  Falls back to a full StatePreparation otherwise.
        """
        from qiskit.circuit.library import StatePreparation
        n = sv.get_n_qubits()
        free = [q for q in range(n) if EPS < sv.probability_measure_one(q) < 1 - EPS]
        definite = [q for q in range(n) if sv.probability_measure_one(q) > 1 - EPS]

        # is the state a product of (definite bits) x (free qubits)?  Check by
        # counting amplitudes: a product with f free qubits has <= 2**f terms.
        if len(free) <= 1 and len(sv.state) <= (1 << max(len(free), 0)):
            for q in definite:
                out.x(out.qubits[q])
            if free:
                q = free[0]
                # factor out the single free qubit's amplitudes
                a = b = 0j
                for k, v in sv.state.items():
                    if k[q]:
                        b = v
                    else:
                        a = v
                norm = (abs(a) ** 2 + abs(b) ** 2) ** 0.5
                out.append(StatePreparation([a / norm, b / norm]), [out.qubits[q]])
            return
        # general fallback (only feasible for modest n)
        if n > 16:
            raise RuntimeError(f"non-product {n}-qubit leaf; full prep too large")
        vec = sv.to_state_vector()
        out.append(StatePreparation(vec), list(out.qubits))

    @classmethod
    def generate_instance(cls, tree: BranchTree, template: QuantumCircuit) -> QuantumCircuit:
        import random
        out = QuantumCircuit(*template.qregs, *template.cregs)
        node = tree
        while node.split:
            node = node.child_one if random.random() < node.p_one else node.child_zero
        # leaf: re-prepare the tracked state, then terminal read-out
        if node.leaf_state is not None:
            cls._prepare_state(out, node.leaf_state)
        for q, c in node.terminal_measures:
            out.measure(out.qubits[q], out.clbits[c])
        return out

    # ================================================================= #
    # Fork-merging engine (polynomial): exploit that feed-forward Pauli
    # byproducts are LINEAR in the measurement bits.  Instead of exploring
    # 2^k outcome branches, compute the reference (all-outcomes-0) branch
    # once, then verify reconvergence by flipping ONE measurement bit at a
    # time (k+1 linear passes).  If every single-bit-flip branch returns to
    # the reference state, linearity guarantees all 2^k do.
    # ================================================================= #
    @staticmethod
    def _forward_pass(circuit: QuantumCircuit, forced: Dict[int, int]):
        """One linear sweep.  Non-deterministic measurements are projected to
        `forced[ordinal]` (default 0), counting their ordinal as we go.
        Returns (final_state, terminal_measures, n_nondet_measures)."""
        data = list(circuit.data)
        n = circuit.num_qubits
        qidx = {q: i for i, q in enumerate(circuit.qubits)}
        cidx = {c: i for i, c in enumerate(circuit.clbits)}

        def touched(inst):
            op = inst.operation
            if op.name == IF_ELSE:
                out = set()
                for body in op.params:
                    if body is None:
                        continue
                    for inner in body.data:
                        for q in inner.qubits:
                            out.add(qidx[inst.qubits[body.qubits.index(q)]])
                return out
            return {qidx[q] for q in inst.qubits}

        last_touch = [-1] * n
        for i, inst in enumerate(data):
            for q in touched(inst):
                last_touch[q] = i
        last_nonmeas = -1
        for i, inst in enumerate(data):
            if inst.operation.name.lower() not in (MEASURE, "barrier", "delay"):
                last_nonmeas = i

        sv = QubitState(n)
        bits: Dict[Clbit, int] = {}
        mcount = 0
        for j, inst in enumerate(data):
            if j > last_nonmeas:
                tmeas = [(qidx[d.qubits[0]], cidx[d.clbits[0]])
                         for d in data[j:] if d.operation.name.lower() == MEASURE]
                return sv, tmeas, mcount

            op = inst.operation
            name = op.name.lower()
            if name in IGNORED:
                continue
            q_inds = [qidx[q] for q in inst.qubits]

            if name == IF_ELSE:
                body = op.params[0] if _cond_true(op.condition, bits) else (
                    op.params[1] if len(op.params) > 1 else None)
                if body is not None:
                    for inner in body.data:
                        oq = [qidx[inst.qubits[body.qubits.index(q)]] for q in inner.qubits]
                        _apply_unitary(sv, inner.operation, oq)
                continue

            if name == MEASURE:
                q, cb = q_inds[0], inst.clbits[0]
                p1 = sv.probability_measure_one(q)
                if p1 < EPS:
                    b = 0
                elif p1 > 1 - EPS:
                    b = 1
                else:
                    b = forced.get(mcount, 0)
                    mcount += 1
                _project_inplace(sv, q, bool(b))
                bits[cb] = b
                if last_touch[q] <= j and b == 1:   # dead -> zero it
                    sv.apply_gate(q, [0, 1, 1, 0])
                continue

            if name == RESET:
                q = q_inds[0]
                p1 = sv.probability_measure_one(q)
                if p1 > 1 - EPS:
                    sv.apply_gate(q, [0, 1, 1, 0])
                elif p1 > EPS:
                    _project_inplace(sv, q, False)   # reference: reset-measured 0
                continue

            _apply_unitary(sv, op, q_inds)

        return sv, [], mcount

    @classmethod
    def optimize_merged(cls, circuit: QuantumCircuit, full_state_check: bool = True) -> dict:
        """Polynomial fork-merging.  Returns a report dict with the single
        reference final state, #MCMs eliminated, whether all branches were
        verified to reconverge, and the resulting subcircuit count."""
        ref_state, tmeas, k = cls._forward_pass(circuit, {})
        ref_key = _sv_canon(ref_state)
        tqs = [q for q, _ in tmeas]
        ref_dist = _born_over(ref_state, tqs) if tqs else {}

        state_ok = dist_ok = True
        failed = []
        for i in range(k):
            si, _, _ = cls._forward_pass(circuit, {i: 1})
            same_state = (_sv_canon(si) == ref_key)
            same_dist = _born_over(si, tqs) == ref_dist if tqs else True
            if not same_state:
                state_ok = False
            if not same_dist:
                dist_ok = False
                failed.append(i)

        return {
            "mcm_eliminated": k,
            "verified_reconverge_state": state_ok,    # identical state up to global phase
            "verified_reconverge_dist": dist_ok,      # identical terminal read-out distribution
            "failed_bits": failed,
            "subcircuits": 1 if dist_ok else None,
            "passes": k + 1,                          # linear passes done (vs 2**k)
            "ref_state": ref_state,
            "terminal_measures": tmeas,
        }

    # ----------------------------------------------------------------- #
    @staticmethod
    def count_stats(tree: BranchTree) -> dict:
        """Static stats over the merged branch DAG."""
        seen = set()
        stats = {"nodes": 0, "splits": 0, "leaves": 0, "terminal_measures": 0}

        def walk(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            stats["nodes"] += 1
            if node.split:
                stats["splits"] += 1
                walk(node.child_one)
                walk(node.child_zero)
            else:
                stats["leaves"] += 1
                stats["terminal_measures"] = max(stats["terminal_measures"],
                                                 len(node.terminal_measures))
        walk(tree)
        return stats
