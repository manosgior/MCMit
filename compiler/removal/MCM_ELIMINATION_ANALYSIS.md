# Eliminating mid-circuit measurements (MCMs) from dynamic circuits

This note documents an analysis of how this library removes mid-circuit measurements
from four families of dynamic circuits, why the shipped pass hits a hard ceiling on
them, two prototype engines that lift that ceiling, and what it costs — including the
practical question of **how many shots** you need afterwards.

All numbers below were produced by the scripts in this repo (see [Files](#files)) on
Qiskit 2.1.0 / Aer 0.17.1, single-threaded CPU.

---

## TL;DR

| | shipped `ConstantPropagation` | branching engine | merged engine |
|---|---|---|---|
| MCMs removed (these circuits) | **1** | **all** | **all** |
| output distribution | exact | exact | exact |
| resulting subcircuits | 1 | up to 2^(#MCM) | **1** (proven) |
| cost | linear | exp. in #MCM (unless forks reconverge early) | (#MCM+1) classical sims |
| in repo | yes | `BranchingPropagation.optimize` | `BranchingPropagation.optimize_merged` |

**Shots needed do *not* increase with MCMs eliminated.** The transforms are
distribution-preserving, so the variance of any estimator — and hence the shot count
to reach a target precision — is unchanged. See [Shots](#how-many-shots).

---

## The circuits

From `MCMit/applications`:

- **`constant_depth_GHZ`** — prepares an *n*-qubit GHZ state; measures odd qubits,
  applies XOR-parity `X` corrections, resets and re-entangles; `(n-1)/2` MCMs.
- **`long_range_CNOT`** — teleports a CNOT across a 1-D chain via staggered Bell pairs;
  parity-controlled `Z`/`X` corrections; `n-2` MCMs.
- **`quantum_teleportation`** — `repeated` (3 qubits, resets between steps) and
  `ladder` (`2n+1` qubits, fresh qubits per step); `2n` MCMs each.

All four share one structural feature: **a single entangled "payload" thread** whose
measurements are non-deterministic (Bell/GHZ correlations → p = 0.5) and which feeds
forward into every later operation.

---

## Why the shipped pass removes only 1 MCM

`ConstantPropagation` symbolically simulates the circuit with a **single deterministic
sparse statevector**, partitioned into entangled groups (`UnionTable`). It removes a
measurement two ways:

- **Deterministic measurement** (p = 0 or 1): deleted outright, the bit becomes a known
  constant, the group is preserved. These chain indefinitely.
- **Non-deterministic measurement** (0 < p < 1): replaced by a `ProbabilisticGate`, and
  then `UnionTable.set_top` is called — which tops the **entire entangled group**,
  because the post-measurement state is a *mixture* of two branches a single statevector
  cannot represent.

In all four circuit families every mid-circuit measurement is the non-deterministic kind
and they all live in **one** entangled group. So the first measurement tops the whole
register and every later measurement sees `TOP`. Result: exactly **1** MCM removed,
independent of *n*. Re-running the pass on its own output is a fixpoint (the inserted
`ProbabilisticGate` is *more* opaque than the measurement it replaced).

This is a property of the single-state model, **not** of the information available: the
measurements' randomness is reversible feed-forward, and the final state is in fact
deterministic.

> **Bug fix.** `optimize()` forwarded `max_ent_group_size=None`, crashing whenever a
> group stayed tracked. Fixed to fall back to `DEFAULT_MAX_ENT_GROUP_SIZE`. This is the
> only change to the original code; both prototypes are in a separate module.

---

## Engine 1 — branching (`optimize`, in `BranchingPropagation.py`)

Instead of topping the group on a non-deterministic measurement, **fork** into the two
outcomes, project + renormalise each, resolve the feed-forward corrections per branch
(the controlling bit is now known), and **re-prepare the tracked state at the leaves**
(you cannot replay the protocol gates without the physical collapse — you must prepare
the projected state, mirroring the library's own `_synthesize_rotation`).

Two tricks bound the tree:
- **Statevector merging** (global-phase invariant): branches reconverging to the same
  state share a node.
- **Qubit liveness**: a measured qubit never used again is zeroed, so branches differing
  only in a dead qubit's value reconverge.

Validated correct (output distribution matches the original / the ideal):

| circuit | n | MCMs removed | subcircuits (distinct) | branch nodes | output |
|---|---|---|---|---|---|
| repeated teleport | 5–10 | all (10–20) | 4 | O(n) | exact |
| ladder teleport | 5–10 | all (10–20) | 4 | O(n) | exact |
| GHZ | 5–25 | all ((n-1)/2) | **1** | 2^((n-1)/2) | exact |
| long-range CNOT | 5–17 | all (n-2) | **1** | 2^(n-2) | exact |

**Teleportation stays cheap** (O(n) nodes): resets / dead-qubit zeroing re-bound the
state between measurements, so the *fork points themselves* merge.
**GHZ and CNOT explode** (2^(#MCM) nodes): all measurements happen up front, before any
correction, so the forks nest into a full binary tree and only reconverge at the leaf.
The output is a single circuit, but the *analysis* enumerated every outcome.

---

## Engine 2 — merged (`optimize_merged`, polynomial branches)

The fix for the GHZ/CNOT explosion exploits that the feed-forward byproducts are **Pauli
operators linear in the measurement bits**. So we needn't explore 2^k branches:

1. Compute the **reference branch** once — project every measurement to 0, so all
   bit-controlled corrections drop out. For these protocols the all-zeros outcome needs
   no correction, so this *is* the common final state. (1 linear pass.)
2. **Verify reconvergence** by flipping **one bit at a time** (k passes). Because the
   byproducts compose independently over GF(2), if every single-bit-flip branch returns
   to the reference state, *all* 2^k do.

So `k+1` classical simulations replace `2^k` branches, and the single-subcircuit result
is *proven*, not assumed. Verified `True` (state and distribution) for every circuit.

Merged vs branching wall-clock (single core):

| circuit | n | MCMs | passes (vs branches) | merged time | branching time |
|---|---|---|---|---|---|
| GHZ | 17 | 8 | 9 (vs 256) | 0.15 s | 0.26 s |
| GHZ | 25 | 12 | 13 (vs 4096) | 6.8 s | infeasible |
| CNOT | 17 | 15 | 16 (vs 32768) | 4.9 s | 26 s (5×) |
| CNOT | 25 | 23 | 24 (vs 8.4M) | 2272 s | infeasible |
| teleport | 10 | 20 | 21 | 0.04 s | 0.02 s |

### The honest catch: the cost moves, it doesn't vanish

- Branching is exponential in the **number of measurements** (2^k). Merging **kills that
  factor** (k+1 passes).
- But each pass is a **full statevector simulation**, which is itself exponential when
  the circuit's *intermediate* state is dense. GHZ stays sparse (n=25 in 7 s); long-range
  CNOT stages ~2^(n/2) amplitudes, so n=25 takes ~38 min. **No MCM trick removes that —
  it is the underlying quantum-simulation hardness of the circuit, not the measurements.**
- For **teleportation** the branching engine is actually *better*: it merges forks early,
  while the merged engine wastefully re-simulates k+1 times. The two engines are
  **complementary** (use branching when measurements interleave with resets; use merging
  when they're up front).

A cheap improvement not yet done: share the common prefix across the k+1 passes (each
single-bit-flip pass only diverges from the reference at its own measurement).

---

## How many shots?

**Short answer: the number of shots does not increase with the number of MCMs you
eliminate.** All three transforms are *distribution-preserving* (verified throughout).
The shots needed to estimate an observable *O* to precision ε is `≈ Var(O)/ε²`, and
`Var(O)` is a property of the **output distribution**, which is unchanged. The MCM count
never enters this formula.

What actually happens to the randomness:

- A non-deterministic MCM's randomness is **relocated**, not removed — from a quantum
  measurement to a classical coin flip in a `ProbabilisticGate` — *or it cancels
  entirely* (when the feed-forward corrections undo the byproduct, as in all four
  circuits here). Either way the final distribution and its variance are identical.
- **Best case (reconvergence — GHZ, CNOT, teleportation):** the eliminated MCMs carried
  **zero net output randomness**, so the merged engine yields a *single deterministic
  subcircuit*. You run **one** circuit; no per-shot resampling is needed, and the shot
  count is whatever the final read-out alone demands.
- **General case (probabilistic output):** the optimized circuit has classical coins. To
  estimate the *marginal* distribution correctly you must draw a **fresh instance per
  shot** (re-flip the coins) — otherwise all shots share one branch and the estimate is
  biased. The shot *count* is the same; the number of (cheap, classical) circuit
  generations equals the number of shots.
- The **number of subcircuits / branches does not set the shot count** — branches are
  *sampled* according to their probabilities, never enumerated. A tree with 2^k branches
  still needs only as many shots as the output variance demands.

### Empirical confirmation

Estimating the readout distribution at matched total shots, original (real MCMs) vs
merged (MCMs eliminated):

```
GHZ n=5   TVD-to-ideal           REPEATED teleport n=4   P(final=0)
 shots | orig    | optimized      shots | orig   | optimized
   128 | 0.0297  | 0.0297          128  | 1.0000 | 1.0000
   512 | 0.0270  | 0.0199          512  | 1.0000 | 1.0000
  2048 | 0.0135  | 0.0143         2048  | 1.0000 | 1.0000
  8192 | 0.0025  | 0.0026         8192  | 1.0000 | 1.0000
 32768 | 0.0007  | 0.0014
```

The error falls as ~1/√S identically for both — eliminating 2 (GHZ) or 8 (teleport) MCMs
does not change the shots-vs-accuracy curve. (Teleportation's output is deterministic, so
its estimate is exact at any shot count, with or without the MCMs.)

### Practical upside on hardware

On real devices MCMs and feed-forward are slow and error-prone. Removing them removes
those error sources, so for a *target accuracy on noisy hardware* you typically need
**fewer** shots, plus lower per-shot latency. The fundamental (noiseless) shot count is
unchanged; the practical one usually improves.

---

## When this helps (and when it doesn't)

The leverage of MCM elimination scales with:
- the number of **independent** entangled groups, and
- the number of **deterministic** measurements,

and inversely with how monolithic / dense the entangled state is. The four benchmark
circuits are worst-case for the shipped pass (one group, all non-deterministic) — which
is exactly why they need the branching/merging engines. The remaining wall is classical
simulation cost (dense states), which hardware cannot meaningfully push back (≈ +1 qubit
per 2× single-core speedup; many cores help only across independent runs, not within one).

---

## Files

| file | what |
|---|---|
| `BranchingPropagation.py` | both prototype engines (additive; original pass untouched) |
| `mcm_count.py` | MCMs removed by the shipped pass (GHZ, CNOT) |
| `branching_validate.py` | branching engine on teleportation, validated vs Aer |
| `branching_ghz_cnot.py` | branching engine on GHZ/CNOT (shows 2^k blow-up) |
| `branching_merged.py` | merged vs branching: time, MCMs, subcircuits |
| `shots_experiment.py` | shots-vs-accuracy, original vs optimized |

Run with the `MCMit` venv, e.g.:

```bash
/path/to/MCMit/.venv/bin/python branching_merged.py
```
