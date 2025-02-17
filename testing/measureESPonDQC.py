from benchmarks.load_benchmarks import load_qasm_files
from analysis.fidelity import calculateExpectedSuccessProbability, fidelity, calculateExpectationValue
from analysis.properties import getSize
from backends.backend import loadBackend, getRealEagleBackend
from backends.simulator import simulatorFromBackend

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

from mitiq import Calibrator
from mitiq import MeasurementResult

def executor(circuit: QuantumCircuit):
    #backend = loadBackend("backends/QPUs/GuadalupeDQC_0.015")
    backend = getRealEagleBackend()
    simulator = simulatorFromBackend(backend)
    tqc = transpile(circuit, backend, scheduling_method="alap")
    counts = simulator.run(tqc, shots=10000).result().get_counts()

    return MeasurementResult.from_counts(counts)


def findMitigationStrategy():
    cal = Calibrator(executor, frontend="qiskit")
    print(cal.get_cost())
    cal.run()
    print(cal.results.log_results_cartesian())

def testing():
    benchmarks = load_qasm_files(benchname="ghz", nqbits=(15, 30), benchmark_suites=["QOSLib"], optional_args=[]) # "MaxCut", "regural", "qaoa_r4"

    assert(len(benchmarks))
    backend = loadBackend("backends/QPUs/GuadalupeDQC_0.015")
    simulator = simulatorFromBackend(backend)

    circuits = [QuantumCircuit.from_qasm_file(b) for b in benchmarks]
    circuits = sorted(circuits, key=getSize)
    non_local_gates_before = [c.num_nonlocal_gates() for c in circuits]

    transpiled_circuits = [transpile(c, backend, scheduling_method="alap") for c in circuits]
    non_local_gates_after = [c.num_nonlocal_gates() for c in transpiled_circuits]   
    
    for i, tc in enumerate(transpiled_circuits):
        #if circuits[i].num_qubits < 25:
            #perfect_results = {'0' * circuits[i].num_qubits : 5000, '1' * circuits[i].num_qubits : 5000}
            #results = simulator.run(transpiled_circuits[i]).result().get_counts()
            #print(fidelity(perfect_results, results))
        esp_idle = calculateExpectedSuccessProbability(circuit=tc, backend=backend, onlyIdling=True)
        esp_normal = calculateExpectedSuccessProbability(circuit=tc, backend=backend, onlyIdling=False)       
        print(esp_idle, esp_normal)

#def testMitiq():
    #benchmarks = load_qasm_files(benchname="ghz", nqbits=(15, 30), benchmark_suites=["QOSLib"], optional_args=[])


#testing()
findMitigationStrategy()