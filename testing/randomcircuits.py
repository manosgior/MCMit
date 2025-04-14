from backends.backend import loadBackend, getRealEagleBackend
from backends.simulator import simulatorFromBackend, getNoiselessSimulator

from analysis.properties import countNonLocalGates, getMeasurements, getSize
from analysis.fidelity import fidelity

from benchmarks.load_benchmarks import load_qasm_files

from error_mitigation.repeated_measurements import *

from applications.qubit_reuse.qubit_reuse import QubitReuser

from qiskit.circuit.random import random_circuit
from qiskit.circuit import QuantumCircuit
from qiskit import transpile
from qiskit.primitives import BitArray
from qiskit.converters import circuit_to_dag
from qiskit.qasm3 import dumps

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

import numpy as np

import csv
import json
import re

def getMCMBinaryOPratio(circuit: QuantumCircuit) -> float:
    binary_ops = countNonLocalGates(circuit)
    measurements = getMeasurements(circuit, include_final=False) 

    if binary_ops == 0:
        return float('inf')
    return binary_ops, measurements, measurements / binary_ops

def randomCircuits(n: int, num_qubits: int, depth: int, conditional: bool = False, reset: bool = False) -> list[QuantumCircuit]:
    to_return = []

    for i in range(n):
        qc = random_circuit(num_qubits = num_qubits, max_operands = 2, measure = True, depth = depth, conditional = conditional, reset = reset, seed=1)
        to_return.append(qc)

    return to_return

def processResults(jobID: str, reps = 3, csv_file = "testing/final_measurement_voting.csv"):
    service = QiskitRuntimeService()
    job = service.job(jobID)
    result = job.result()

    results = []

    for r in result:
        if len(r.data.keys()) == 1:
            results.append(list(r.data.values())[0].get_counts())
        else:
            bitstrings = [v.get_bitstrings() for v in list(r.data.values())]
            cleaned_up_bitstrings = [cleanup_bitstrings_per_creg(bts) for bts in bitstrings]
            clean_counts = majority_vote_counts_separate_cregs(cleaned_up_bitstrings)
            print(clean_counts)
            results.append(clean_counts)
    
    tqc_full_counts = [results[i] for i in range(reps)]
    tqc_fixed_counts = [results[i + reps] for i in range(reps)]

    perfect_counts = {"0" * 6: 4096, "1" * 6: 4096}

    tqc_full_fids = [fidelity(perfect_counts, tqc_full_counts[i]) for i in range(reps)]
    tqc_full_fid = np.mean(tqc_full_fids)

    tqc_fixed_fids = [fidelity(perfect_counts, tqc_fixed_counts[i]) for i in range(reps)]
    tqc_fixed_fid = np.mean(tqc_fixed_fids)
    print(f"Fidelity of original circuit: {tqc_full_fid}, fids: {tqc_full_fids}")
    print(f"Fidelity of mitigated circuit: {tqc_fixed_fid}, fids: {tqc_fixed_fids}")  

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "ghz",
            6,
            json.dumps(tqc_full_fids),        # serialize list
            tqc_full_fid,
            json.dumps(tqc_fixed_fids),    # serialize list
            tqc_fixed_fid
    ])

#processResults("czygqqed8drg008hwwjg")
#exit()

benchmarks = load_qasm_files(benchname="ghz", nqbits=(8, 10), benchmark_suites=["QOSLib"], optional_args=[])
filtered = []

for path in benchmarks:
    match = re.search(r'n(\d+)\.qasm', path)
    if match:
        number = int(match.group(1))
        if number == 8:
            filtered.append(path)

circuits = [QuantumCircuit.from_qasm_file(b) for b in filtered]
circuits = sorted(circuits, key=getSize)
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.least_busy(operational=True, simulator=False)
csv_file = "testing/final_measurement_voting.csv"
 #backend = getRealEagleBackend()
#simulator = simulatorFromBackend(backend)
reps = 3

for example_circ in circuits:
    fixed = add_redundant_measurements(example_circ)
    tqc_full = transpile(example_circ, backend)
    tqc_fixed = transpile(fixed, backend)
    perfect_counts = {"0" * example_circ.num_qubits: 4096, "1" * example_circ.num_qubits: 4096}
    print("transpilations done")

    to_run = []
    for i in range(reps):
        to_run.append(tqc_full)
    for i in range(reps):
        to_run.append(tqc_fixed)

    print("running")
    sampler = Sampler(backend)
    job = sampler.run(to_run, shots=8192)    
    result = job.result()
    print("done running")

    results = []

    for r in result:
        if len(r.data.keys()) == 1:
            results.append(list(r.data.values())[0].get_counts())
        else:
            bitstrings = [v.get_bitstrings() for v in list(r.data.values())]
            cleaned_up_bitstrings = [cleanup_bitstrings_per_creg(bts) for bts in bitstrings]
            clean_counts = majority_vote_counts_separate_cregs(cleaned_up_bitstrings)
            results.append(clean_counts)
    
    tqc_full_counts = [results[i] for i in range(reps)]
    tqc_fixed_counts = [results[i + reps] for i in range(reps)]

    tqc_full_fids = [fidelity(perfect_counts, tqc_full_counts[i]) for i in range(reps)]
    tqc_full_fid = np.mean(tqc_full_fids)

    tqc_fixed_fids = [fidelity(perfect_counts, tqc_fixed_counts[i]) for i in range(reps)]
    tqc_fixed_fid = np.mean(tqc_fixed_fids)
    print(f"Fidelity of original circuit: {tqc_full_fid}, fids: {tqc_full_fids}")
    print(f"Fidelity of mitigated circuit: {tqc_fixed_fid}, fids: {tqc_fixed_fids}")  

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "ghz",
            example_circ.num_qubits,
            json.dumps(tqc_full_fids),        # serialize list
            tqc_full_fid,
            json.dumps(tqc_fixed_fids),    # serialize list
            tqc_fixed_fid
    ])
exit()



