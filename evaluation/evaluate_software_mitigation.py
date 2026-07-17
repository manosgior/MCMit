"""
evaluate_software_mitigation.py
================================
Fig. 11 driver (§8.4): Raw vs. MCMit vs. Qiskit M3 fidelity across the four
dynamic-circuit benchmarks (constant-depth GHZ, long-range CNOT, repeated
teleportation, ladder teleportation).

Methods
-------
- Raw       : circuit as-is.
- Qiskit M3 : circuit as-is, mthree readout correction applied to the final
              payload-register counts.
- MCMit     : the §7 software pipeline -- (1) static MCM elimination
              (compiler/removal's ConstantPropagation: dynamic circuit ->
              probabilistic circuit, one concrete instance sampled per shot
              group), (2) optional measurement hardening (parity checks +
              shot discarding, --hardening), (3) stochastic branching of the
              remaining conditionals using the M3 confusion matrices.

The paper's published Fig. 11 numbers were produced with this methodology on
ibm_fez hardware (10,000 shots per point, one calibration cycle); this script
reproduces the pipeline on a noisy Aer simulator by default so it runs
anywhere. Expect the same qualitative ordering, not bit-identical values.

Output: results/software_mitigation_fidelity.csv schema
    Benchmark,N,Method,Fidelity

Examples
--------
    python -m evaluation.evaluate_software_mitigation --benchmark ghz --max-n 9
    python -m evaluation.evaluate_software_mitigation --benchmark all --output results/software_mitigation_fidelity.csv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mthree
import mthree.utils
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from applications.constant_depth_GHZ import create_constant_depth_ghz, get_perfect_ghz_distribution
from applications.long_range_CNOT import create_dynamic_CNOT_circuit, get_perfect_distribution_long_range_cnot
from applications.quantum_teleportation import (
    create_repeated_teleportation_circuit,
    create_ladder_teleportation_circuit,
    get_perfect_distribution_teleportation,
)
from compiler.removal.ConstantPropagation import ConstantPropagation
from compiler.branching.stochastic_branching import apply_stochastic_branching, fetch_calibrations_from_file
from compiler.decoding.adaptive_soft_decoding import add_parity_checks_greedy, discard_parity_violations
from evaluation.fidelity import fidelity

CALIBRATIONS_FILE = os.path.join(os.path.dirname(__file__), "..", "backends", "calibrations", "calibrations.json")

# Benchmark registry: label, N sweep, circuit builder, perfect distribution,
# and the name of the classical register holding the final payload readout.
BENCHMARKS = {
    "ghz": dict(
        label="Constant-depth GHZ",
        ns=lambda max_n: list(range(5, max_n + 1, 2)),
        build=create_constant_depth_ghz,
        perfect=lambda n, shots: get_perfect_ghz_distribution(n, shots),
        payload_reg="meas",
    ),
    "cnot": dict(
        label="Long-range CNOT",
        ns=lambda max_n: list(range(5, max_n + 1, 2)),
        build=create_dynamic_CNOT_circuit,
        perfect=lambda n, shots: get_perfect_distribution_long_range_cnot(shots),
        payload_reg="cr3",
    ),
    # removal_exact=False: ConstantPropagation.generate_instance does not yet
    # reproduce the exact output distribution on the teleportation circuits
    # (verified noiselessly: instance fidelity ~0.3 vs 1.0 raw; upstream
    # pcm-ccop-dc-pass's own main.py notes an instance-generation issue with
    # BigProbabilisticGate). Under --removal auto (default), stage 1 is
    # skipped for these and MCMit = hardening + stochastic branching only.
    "teleport-repeated": dict(
        label="Repeated teleportation",
        ns=lambda max_n: list(range(3, max_n + 1, 2)),  # builder requires n_teleports > 1
        build=create_repeated_teleportation_circuit,
        perfect=lambda n, shots: get_perfect_distribution_teleportation(shots),
        payload_reg="final",
        removal_exact=False,
    ),
    "teleport-ladder": dict(
        label="Ladder teleportation",
        ns=lambda max_n: list(range(2, max_n + 1)),  # builder requires n_teleports > 1
        build=create_ladder_teleportation_circuit,
        perfect=lambda n, shots: get_perfect_distribution_teleportation(shots),
        payload_reg="final",
        removal_exact=False,
    ),
}


def payload_indices(circuit: QuantumCircuit, reg_name: str) -> list:
    """Global clbit indices of the payload register, taken from the ORIGINAL
    circuit. ConstantPropagation.generate_instance flattens registers into
    loose clbits but preserves clbit indices, so the original layout stays
    valid for every circuit variant this driver executes."""
    reg = next(r for r in circuit.cregs if r.name == reg_name)
    return [circuit.find_bit(b).index for b in reg]


def marginalize(counts: dict, idxs: list) -> dict:
    """Marginal counts over the given global clbit indices."""
    out = {}
    for key, c in counts.items():
        flat = key.replace(" ", "")[::-1]  # position 0 = clbit 0
        sub = "".join(flat[i] for i in idxs)[::-1]
        out[sub] = out.get(sub, 0) + c
    return out


# Composite instruction blocks emitted by ConstantPropagation.generate_instance
# segfault qiskit-aer 0.17's assembler if left undecomposed, so every circuit is
# first lowered to standard gates before the backend transpile.
_LOWERING_BASIS = ["u", "cx", "measure", "reset", "if_else"]


def run_counts(circuit: QuantumCircuit, backend, shots: int) -> dict:
    lowered = transpile(circuit, basis_gates=_LOWERING_BASIS, optimization_level=1)
    tqc = transpile(lowered, backend, optimization_level=1)
    return backend.run(tqc, shots=shots).result().get_counts(), tqc


def eval_raw(qc, spec, backend, shots):
    counts, _ = run_counts(qc, backend, shots)
    return marginalize(counts, payload_indices(qc, spec["payload_reg"]))


def eval_m3(qc, spec, backend, shots, mit_cache={}):
    counts, tqc = run_counts(qc, backend, shots)
    mapping = mthree.utils.final_measurement_mapping(tqc)

    payload_idxs = payload_indices(qc, spec["payload_reg"])
    qubits = [mapping[i] for i in payload_idxs if i in mapping]
    marginal = marginalize(counts, payload_idxs)
    if len(qubits) != len(payload_idxs):
        return marginal  # payload bits not all from terminal measurements; skip correction

    key = id(backend)
    if key not in mit_cache:
        mit_cache[key] = mthree.M3Mitigation(backend)
    mit = mit_cache[key]
    mit.cals_from_system(qubits, shots=min(shots, 10000))
    quasi = mit.apply_correction(marginal, qubits)
    probs = quasi.nearest_probability_distribution()
    return {k: v * shots for k, v in probs.items()}


def eval_mcmit(qc, spec, backend, shots, branch_samples, hardening, cal_matrix, removal=True):
    # Stage 1 (§7.1): static MCM elimination -> probabilistic circuit.
    prob_circ = None
    if removal:
        try:
            prob_circ = ConstantPropagation.optimize(qc, max_amplitudes=2**20, max_ent_group_size=64)
        except Exception as e:
            print(f"    [mcmit] MCM removal skipped ({type(e).__name__}: {e})")

    merged = {}
    shots_per_sample = max(1, shots // branch_samples)
    for _ in range(branch_samples):
        inst = ConstantPropagation.generate_instance(prob_circ) if prob_circ is not None else qc.copy()

        # Stage 2 (§7.2): measurement hardening (parity checks), optional.
        hardened = False
        if hardening:
            try:
                inst = add_parity_checks_greedy(inst, backend, max_attempts=5, threshold=0.3)
                hardened = any(r.name.startswith("cr_anc") for r in inst.cregs)
            except Exception as e:
                print(f"    [mcmit] hardening skipped ({type(e).__name__}: {e})")

        # Stage 3 (§7.3): stochastic branching on remaining conditionals.
        inst = apply_stochastic_branching(inst, cal_matrix)

        counts, _ = run_counts(inst, backend, shots_per_sample)
        if hardened:
            counts, _ = discard_parity_violations(counts, inst)
        for k, c in marginalize(counts, payload_indices(qc, spec["payload_reg"])).items():
            merged[k] = merged.get(k, 0) + c
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=[*BENCHMARKS, "all"], default="all")
    parser.add_argument("--max-n", type=int, default=9,
                        help="Upper end of the size sweep (paper: 25 for GHZ/CNOT/repeated, 12 for ladder)")
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--branch-samples", type=int, default=16,
                        help="Stochastic-branching / instance samples per data point (shots are split across them)")
    parser.add_argument("--hardening", action="store_true",
                        help="Apply parity-check measurement hardening (transpile-heavy; paper applies it to GHZ)")
    parser.add_argument("--removal", choices=["auto", "on", "off"], default="auto",
                        help="Stage-1 MCM elimination: 'auto' skips benchmarks where instance "
                             "generation is known not to be distribution-preserving (teleportation)")
    parser.add_argument("--noiseless", action="store_true", help="Noiseless Aer (sanity check: all fidelities ~1)")
    parser.add_argument("--calibrations", default=CALIBRATIONS_FILE,
                        help="M3 calibration file for stochastic branching (ideally from the same "
                             "backend the circuits run on; see compute_calibrations_from_backend)")
    parser.add_argument("--output", default=None, help="CSV to append rows to (default: print only)")
    args = parser.parse_args()

    if args.noiseless:
        backend = AerSimulator()
        # Stochastic branching mirrors the backend's measurement confusion
        # matrix; a noiseless backend has an identity confusion matrix, so no
        # conditions should ever be flipped in the sanity-check mode.
        import numpy as np
        cal_matrix = [np.eye(2)] * 256
    else:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        fake = FakeSherbrooke()
        backend = AerSimulator(noise_model=NoiseModel.from_backend(fake))
        cal_matrix = fetch_calibrations_from_file(args.calibrations)
    names = list(BENCHMARKS) if args.benchmark == "all" else [args.benchmark]

    rows = []
    for name in names:
        spec = BENCHMARKS[name]
        for n in spec["ns"](args.max_n):
            qc = spec["build"](n)
            perfect = spec["perfect"](n, args.shots)
            use_removal = {"on": True, "off": False,
                           "auto": spec.get("removal_exact", True)}[args.removal]
            for method, dist in [
                ("Raw", eval_raw(qc, spec, backend, args.shots)),
                ("MCMit", eval_mcmit(qc, spec, backend, args.shots, args.branch_samples,
                                     args.hardening and name == "ghz", cal_matrix,
                                     removal=use_removal)),
                ("Qiskit M3", eval_m3(qc, spec, backend, args.shots)),
            ]:
                fid = fidelity(perfect, dist)
                rows.append({"Benchmark": spec["label"], "N": n, "Method": method, "Fidelity": fid})
                print(f"{spec['label']},{n},{method},{fid:.4f}", flush=True)

    if args.output:
        exists = os.path.isfile(args.output)
        with open(args.output, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Benchmark", "N", "Method", "Fidelity"])
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        print(f"\nAppended {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()