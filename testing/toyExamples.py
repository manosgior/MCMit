from backends.backend import loadBackend, getRealEagleBackend
from backends.simulator import simulatorFromBackend
from benchmarks.load_benchmarks import load_qasm_files
from analysis.properties import getSize
from analysis.fidelity import fidelity

from collections import Counter

import mthree

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister

def majorityVote(bits: list[int]) -> int:
    count = Counter(bits)

    return '1' if count['1'] > count['0'] else '0'

def inflateCircuit(qc: QuantumCircuit, M: int) -> QuantumCircuit:
    qr = QuantumRegister(qc.num_qubits)
    cr = ClassicalRegister(qc.num_clbits * M)
    new_qc = QuantumCircuit(qr, cr)

    qc_copy = qc.copy()
    qc_copy.remove_final_measurements()
    new_qc.compose(qc_copy, inplace=True)

    counter = 0
    for i in range(qc.num_qubits):
        for j in range(M):
            new_qc.measure(i, counter)
            counter += 1

    return new_qc

def processCounts(counts: dict[str, int], N: int, M: int) -> dict[str, int]:
    processed_counts = {}
    
    for bitstring, count in counts.items():
        bits = [bitstring[i * M: (i + 1) * M] for i in range(N)]
        merged_bits = ''.join(majorityVote(b) for b in bits)

        processed_counts[merged_bits] = processed_counts.get(merged_bits, 0) + count
    
    return processed_counts


backend = getRealEagleBackend()
simulator = simulatorFromBackend(backend)
mit = mthree.M3Mitigation(simulator)
mit.cals_from_system()

benchmarks = load_qasm_files(benchname="ghz", nqbits=(8, 16), benchmark_suites=["QOSLib"], optional_args=[])
circuits = [QuantumCircuit.from_qasm_file(b) for b in benchmarks]
circuits = sorted(circuits, key=getSize)

for c in circuits:
    print("!" * 10, str(c.num_qubits), "!" * 10)
    for M in [1, 3, 5, 7]:
        print("_" * 8 + str(M) + "_" * 8)
        perfect_counts = {'0' * c.num_qubits : 5000, '1' * c.num_qubits : 5000}
        inflated_circuit = inflateCircuit(c, M)

        tqc = transpile(c, backend)
        counts = simulator.run(tqc, shots=10000).result().get_counts()

        tqc_infl = transpile(inflated_circuit, backend)
        counts_infl = simulator.run(tqc_infl, shots=10000).result().get_counts()
        processed_counts = processCounts(counts_infl, c.num_qubits, M)

        origin_fid = fidelity(perfect_counts, counts)
        print(origin_fid)
        mitiq_fid = fidelity(perfect_counts, processed_counts)
        print(mitiq_fid)        

        mitigated_counts = mit.apply_correction(processed_counts, qubits=list(range(c.num_qubits)))
        mitigated_counts = {bitstring: count * 10000 for bitstring, count in mitigated_counts.items() if count > 0}
        #print(mitigated_counts)
        doublemitiq_fid = fidelity(perfect_counts, mitigated_counts)
        print(doublemitiq_fid)
        
        print(mitiq_fid / origin_fid)
        print(doublemitiq_fid / origin_fid)

        


