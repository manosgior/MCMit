from benchmarks.load_benchmarks import load_qasm_files
from analysis.fidelity import calculateExpectedSuccessProbability, fidelity, calculateExpectationValue, getEVFidelity
from analysis.properties import getSize
from backends.backend import loadBackend, getRealEagleBackend
from backends.simulator import simulatorFromBackend

from error_mitigation.apply_error_mitigation import *

from mitiq import MeasurementResult
from mitiq import ddd
from mitiq import zne
from mitiq.zne.scaling import fold_gates_at_random 
from mitiq.zne.inference import RichardsonFactory
from mitiq.zne import combine_results, scaled_circuits
from mitiq.rem import generate_inverse_confusion_matrix, generate_tensored_inverse_confusion_matrix
from mitiq import rem
from mitiq.raw import execute
from mitiq.lre import execute_with_lre

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator


import numpy as np

def executor(circuit: QuantumCircuit):
    backend = loadBackend("backends/QPUs/GuadalupeDQC_0.015")
    #backend = getRealEagleBackend()
    simulator = simulatorFromBackend(backend)
    tqc = transpile(circuit, backend)
    counts = simulator.run(tqc, shots=10000).result().get_counts()


    return calculateExpectationValue(counts, 10000, mode = "parity")


def getAllExpectationValues(counts: dict[str, int]):
    to_return = {}

    for mode in ["sum", "product", "parity"]:
        to_return[mode] = calculateExpectationValue(counts=counts, mode=mode)

    return to_return

def testDDD():
    benchmarks = load_qasm_files(benchname="ghz", nqbits=(8, 15), benchmark_suites=["QOSLib"], optional_args=[]) # "MaxCut", "regural", "qaoa_r4"
    backend = loadBackend("backends/QPUs/Guadalupe")
    assert(len(benchmarks))

    circuits = [QuantumCircuit.from_qasm_file(b) for b in benchmarks]
    circuits = sorted(circuits, key=getSize)

    for c in circuits:
        perfect_results = {'0' * c.num_qubits : 5000, '1' * c.num_qubits : 5000}
        perfect_expectation_value = calculateExpectationValue(perfect_results, mode="parity")
        fids = []
        evs = []
        miti_evs = []
        degree = [2, 3]
        fold_multiplier = [2, 3]

        for i in range(5):
            noisy_results = execute(circuit=c, executor=executor)
            #noisy_expectation_value = calculateExpectationValue(noisy_results, mode="parity")
            #fids.append(fidelity(noisy_results, perfect_results))
            evs.append(getEVFidelity(noisy_results, perfect_expectation_value))

            mitigated_result = applyReadoutErrorMitigation(c, backend)
            miti_evs.append(getEVFidelity(mitigated_result, perfect_expectation_value))


        print(evs, np.mean(evs))
        print(miti_evs, np.mean(miti_evs))
        #confusion_matrices = np.tile(backend.confusion_matrix, (32, 1, 1))
  
        #inverse_confusion_matrix = generate_tensored_inverse_confusion_matrix(backend.num_qubits, confusion_matrices=confusion_matrices)
        #mitigated_result = rem.execute_with_rem(
            #c,
            #executor=executor,
            #observable,
           # inverse_confusion_matrix=inverse_confusion_matrix,
        #)

        

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

def cleanTest():
    benchmarks = load_qasm_files(benchname="ghz", nqbits=(8, 15), benchmark_suites=["QOSLib"], optional_args=[]) # "MaxCut", "regural", "qaoa_r4"

    circuits = [QuantumCircuit.from_qasm_file(b) for b in benchmarks]
    circuits = sorted(circuits, key=getSize)

    for c in circuits:
        perfect_results = {'0' * c.num_qubits : 5000, '1' * c.num_qubits : 5000}
        perfect_expectation_value = calculateExpectationValue(perfect_results, mode="parity")
        evs = []
        miti_evs = []

        for i in range(5):
            noisy_results = execute(circuit=c, executor=executor)
            evs.append(getEVFidelity(noisy_results, perfect_expectation_value))

            mitigated_result = applyErrorMitigationQiskit(circuit=c, em_techniques=["DD"])
            miti_evs.append(getEVFidelity(mitigated_result, perfect_expectation_value))
        
        print(evs, np.mean(evs))
        print(miti_evs, np.mean(miti_evs))

testDDD()
#cleanTest()

