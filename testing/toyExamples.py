from backends.backend import loadBackend, getRealEagleBackend
from backends.simulator import simulatorFromBackend
from benchmarks.load_benchmarks import load_qasm_files
from analysis.properties import getSize
from analysis.fidelity import fidelity

from collections import Counter

import mthree

from applications.qubit_reuse.qubit_reuse import QubitReuser

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
from qiskit_ibm_runtime import EstimatorV2 as Estimator

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


#backend = getRealEagleBackend()
#simulator = simulatorFromBackend(backend)
#mit = mthree.M3Mitigation(simulator)
#mit.cals_from_system()
service = QiskitRuntimeService(channel="ibm_quantum")
benchmarks = load_qasm_files(benchname="qaoa", nqbits=(30, 50), benchmark_suites=["QOSLib"], optional_args=["MaxCut", "regular", "qaoa_r4"])
circuits = [QuantumCircuit.from_qasm_file(b) for b in benchmarks]
circuits = sorted(circuits, key=getSize)

resilience_levels = [0, 1, 2]
num_executions = 5  # Runs per circuit per resilience level
backend = service.least_busy(operational=True, simulator=False)


for resilience in resilience_levels:
    print(f"Running Resilience: {resilience}")

    # Run circuit using Qiskit Runtime
    with Estimator(backend, options={"resilience_level": resilience}) as estimator:
        job = estimator.run(circuits)
        print(job.usage_estimation())
        result = job.result()
        print(job.usage())
        print(job.metrics())
    
    #print(f"Result: {result.quasi_dists}\n")
            
    
    #print("*" * 3, str(c.num_qubits), "*" * 3)
    #print(c)
    #print("-" * 50)
    #reuser = QubitReuser(c.num_qubits - 1, dynamic=False)
    #c_qr = reuser.run(c)
    #print(c_qr)
    continue
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

        mapping = mthree.utils.final_measurement_mapping(tqc_infl)        
        mitigated_counts = mit.apply_correction(processed_counts, qubits=mapping)
        mitigated_counts = {bitstring: count * 10000 for bitstring, count in mitigated_counts.items() if count > 0}
        #print(mitigated_counts)
        doublemitiq_fid = fidelity(perfect_counts, mitigated_counts)
        print(doublemitiq_fid)

        print(mitiq_fid / origin_fid)
        print(doublemitiq_fid / origin_fid)

        


