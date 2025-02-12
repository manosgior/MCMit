from benchmarks.load_benchmarks import load_qasm_files
from analysis.fidelity import calculateExpectedSuccessProbability
from analysis.properties import getSize
from backends.Backend import loadBackend

from qiskit import transpile
from qiskit.circuit import QuantumCircuit


def testing():
    benchmarks = load_qasm_files(benchname="ghz", nqbits=(15, 30), benchmark_suites=["Supermarq"])
    backend = loadBackend("backends/GuadalupeDQC_0.015")

    circuits = [QuantumCircuit.from_qasm_file(b) for b in benchmarks]
    circuits = sorted(circuits, key=getSize)
    non_local_gates_before = [c.num_nonlocal_gates() for c in circuits]

    transpiled_circuits = [transpile(c, backend) for c in circuits]
    non_local_gates_after = [c.num_nonlocal_gates() for c in transpiled_circuits]

    print(non_local_gates_before)
    print(non_local_gates_after)

    for tc in transpiled_circuits:
        esp_idle = calculateExpectedSuccessProbability(circuit=tc, backend=backend, onlyIdling=True)
        esp_normal = calculateExpectedSuccessProbability(circuit=tc, backend=backend, onlyIdling=False)
        print(esp_idle, esp_normal)

 
testing()