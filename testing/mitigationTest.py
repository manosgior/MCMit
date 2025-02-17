from benchmarks.load_benchmarks import load_qasm_files
from analysis.fidelity import calculateExpectedSuccessProbability, fidelity, calculateExpectationValue
from analysis.properties import getSize
from backends.backend import loadBackend, getRealEagleBackend
from backends.simulator import simulatorFromBackend

from mitiq import MeasurementResult
from mitiq import ddd
from mitiq import zne
from mitiq.zne.scaling import fold_gates_at_random 
from mitiq.zne.inference import RichardsonFactory
from mitiq.zne import combine_results, scaled_circuits

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator


def executor(circuit: QuantumCircuit, expVaue: bool = True):
    backend = loadBackend("backends/QPUs/GuadalupeDQC_0.015")
    #backend = getRealEagleBackend()
    simulator = simulatorFromBackend(backend)
    tqc = transpile(circuit, backend, scheduling_method="alap")
    counts = simulator.run(tqc, shots=10000).result().get_counts()

    if expVaue:
        return calculateExpectationValue(counts, 10000)
    else:
        return counts


def testDDD():
    benchmarks = load_qasm_files(benchname="ghz", nqbits=(8, 16), benchmark_suites=["QOSLib"], optional_args=[]) # "MaxCut", "regural", "qaoa_r4"

    assert(len(benchmarks))

    circuits = [QuantumCircuit.from_qasm_file(b) for b in benchmarks]
    circuits = sorted(circuits, key=getSize)

    for c in circuits:
        perfect_results = {'0' * c.num_qubits : 5000, '1' * c.num_qubits : 5000}
        perfect_expectation_value = calculateExpectationValue(perfect_results)
        fids = []
        evs = []

        for i in range(5):
            noisy_results = executor(c, False)
            noisy_expectation_value = calculateExpectationValue(noisy_results)
            fids.append(fidelity(noisy_results, perfect_results))
            evs.append(abs(noisy_expectation_value - perfect_expectation_value))
    
        print(fids)
        print(evs)
        #rule = ddd.rules.yy
        #dd_result = ddd.execute_with_ddd(circuit=circuits[0], executor=executor, rule=rule)
    
        #mitigated_result = zne.execute_with_zne(circuit=c, executor=executor)
        #print(abs(mitigated_result -  perfect_expectation_value))

    exit()
    #scale_factors = [1.0, 2.0, 3.0]
    #folded_circuits = scaled_circuits(
        #circuit=circuits[0],
        #scale_factors=scale_factors,
        #scale_method=fold_gates_at_random,
    #)
    #results = [executor(circuit) for circuit in folded_circuits]

    #extrapolation_method = RichardsonFactory(scale_factors=scale_factors).extrapolate
    #two_stage_zne_result = combine_results(
        #scale_factors, results, extrapolation_method
    #)
    #print(two_stage_zne_result)
   #print(abs(two_stage_zne_result -  perfect_expectation_value))

   


testDDD()

