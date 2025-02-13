from benchmarks.load_benchmarks import load_qasm_files
from analysis.fidelity import calculateExpectedSuccessProbability
from analysis.properties import getSize
from backends.backend import loadBackend

from qiskit import transpile
from qiskit.circuit import QuantumCircuit

def testing():
    benchmarks = load_qasm_files(benchname="qaoa", nqbits=(15, 30), benchmark_suites=["QOSLib"], optional_args=["MaxCut", "regural", "qaoa_r4"])
    print(benchmarks)
    assert(len(benchmarks))
    backend = loadBackend("backends/QPUs/GuadalupeDQC_0.015")

    circuits = [QuantumCircuit.from_qasm_file(b) for b in benchmarks]
    circuits = sorted(circuits, key=getSize)
    non_local_gates_before = [c.num_nonlocal_gates() for c in circuits]

    transpiled_circuits = [transpile(c, backend, scheduling_method="alap") for c in circuits]
    non_local_gates_after = [c.num_nonlocal_gates() for c in transpiled_circuits]

    for tc in transpiled_circuits:
        esp_idle = calculateExpectedSuccessProbability(circuit=tc, backend=backend, onlyIdling=True)
        esp_normal = calculateExpectedSuccessProbability(circuit=tc, backend=backend, onlyIdling=False)
        print(esp_idle, esp_normal)

 
testing()