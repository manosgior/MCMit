# compiler/removal

Static MCM elimination for MCMit's software error-mitigation stage (§ 7.1,
"Dynamic Circuit Simplification"): identifies mid-circuit measurements whose
outcome is statically determined (or can be resolved without full quantum
simulation) and rewrites the circuit into an equivalent **probabilistic
circuit**, removing the MCM, its dependent classical feedback, and — where
applicable — 2-qubit gates that only existed to consume the measurement
result.

Synced from [pcm-ccop-dc-pass](https://github.com/i2-tum/pcm-ccop-dc-pass)
(i2-tum), the standalone implementation of this pass. That repo also depends
on MCMit's own circuit generators (`applications/constant_depth_GHZ.py`,
`applications/long_range_CNOT.py`, `applications/quantum_teleportation.py`)
for its benchmarks, so it's folded in here as a proper part of the MCMit
software rather than kept as a separate sibling checkout.

The pass is based on quantum constant propagation (Y. Chen and Y. Stade,
"Quantum Constant Propagation," Static Analysis (SAS 2023), Springer,
pp. 164–189), extended for mid-circuit measurement removal in Y. Chen,
I. Fulginiti, C. B. Mendl, "Reducing Mid-Circuit Measurements via
Probabilistic Circuits," QCE 2024
(https://ieeexplore.ieee.org/abstract/document/10821341) and "Optimization
Framework for Reducing Mid-Circuit Measurements and Resets," Computational
Science – ICCS 2025 Workshops, Springer
(https://link.springer.com/chapter/10.1007/978-3-031-97570-7_13).

## Quick start

```python
from ConstantPropagation import ConstantPropagation
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

qr = QuantumRegister(2, "q")
cr = ClassicalRegister(1, "c")
qc = QuantumCircuit(qr, cr)

qc.h(0)
qc.measure(0, 0)
with qc.if_test((cr[0], 1)):
    qc.x(1)
qc.reset(0)

# 1) Transform dynamic circuit -> probabilistic circuit
prob_circ = ConstantPropagation.optimize(qc)

# 2) Instantiate (sample) the probabilistic circuit
inst_1 = ConstantPropagation.generate_instance(prob_circ)
inst_2 = ConstantPropagation.generate_instance(prob_circ)
```

`inst_1` and `inst_2` may differ — sampling of the probabilistic constructs
(`util/ProbabilisticGate.py`) is independent at each call.

## Layout

| File | Role |
|---|---|
| `ConstantPropagation.py` | The shipped pass (§ 7.1's static MCM elimination, based on quantum constant propagation — see references above). Tracks a single deterministic symbolic statevector per entangled group (`UnionTable`); a non-deterministic MCM forces the whole group to be abandoned (capped at 1 eliminable MCM per entangled thread). |
| `BranchingPropagation.py` | Prototype extension: forks a `BranchTree` on every non-deterministic MCM instead of abandoning the group, then merges branches that reconverge to the same physical state. Eliminates *all* eliminable MCMs, at a cost that (for the merged engine, `optimize_merged`) stays polynomial rather than the naive 2^(#MCM). See `MCM_ELIMINATION_ANALYSIS.md`. |
| `SimplifyCondition.py` | Simplifies classical control-flow conditions (XOR/majority-vote collapsing, condition inversion) once the measurements feeding them are known or partially known — the multi-qubit classical-control-logic extension described in § 7.1. |
| `UnionTable.py` | Entanglement-aware union-find structure tracking which qubits are jointly represented by a single symbolic state, and the `N_max` cutoff that bounds pass runtime (§ 7.1). |
| `QuState.py` | Symbolic quantum state representation used by both engines. |
| `util/ProbabilisticGate.py` | The `ProbabilisticGate` construct — a gate applied with probability *p*, resolved by `generate_instance`. |
| `util/ActivationState.py`, `util/BitState.py` | Classical-bit / activation bookkeeping for conditional blocks. |
| `util/MyRandomCircuit.py` | Random dynamic-circuit generator, used by `test.ipynb` for fuzz-testing the pass. |
| `main.py`, `test.ipynb` | Minimal usage examples / scratch tests. |

## Benchmarking against MCMit's own circuits

These scripts run the pass (both engines) against MCMit's constant-depth
GHZ, long-range CNOT, and teleportation benchmarks
(`applications/*.py`) — the same circuits underlying Fig. 9 and Table 8:

| Script | What it measures |
|---|---|
| `mcm_count.py` | MCMs eliminated by the shipped `ConstantPropagation` pass alone, vs. `max_amplitudes`/`max_ent_group_size` cutoffs, on GHZ/CNOT. |
| `branching_validate.py` | Branching engine on teleportation circuits, cross-checked against a plain Aer simulation of the original circuit. |
| `branching_ghz_cnot.py` | Branching engine on GHZ/CNOT — demonstrates the 2^k branch blow-up the merged engine avoids. |
| `branching_merged.py` | Merged vs. branching engine: elimination count, subcircuit count, wall time, speedup — across all four benchmark families. |
| `ghz_fidelity.py` | State fidelity of the GHZ circuit with MCMs eliminated vs. an MCM-free unitary (CNOT-ladder) preparation, under no/light/heavy depolarizing noise. |
| `shots_experiment.py` | Whether MCM elimination changes the number of shots needed to estimate the same final-readout distribution to a given accuracy (it doesn't — shot-neutral). |

Run any of them directly, e.g. `python mcm_count.py`, from this directory —
they resolve `applications/` relative to their own file location, so no
`PYTHONPATH`/`sys.path` setup is needed beyond having Qiskit + Aer
installed. See `MCM_ELIMINATION_ANALYSIS.md` for the full write-up and
numbers behind these scripts (shipped pass vs. the two prototype engines,
including the "honest catch" on where the eliminated cost actually moves
to, and when eliminating MCMs helps vs. doesn't).

## Status

The shipped `ConstantPropagation` pass is what § 7.1 describes and is used
elsewhere in MCMit's evaluation. `BranchingPropagation.py`'s two engines are
prototypes explored on top of it (not claimed as a paper result) — see
`MCM_ELIMINATION_ANALYSIS.md`'s TL;DR table for the exact trade-offs before
relying on either for anything beyond benchmarking.
