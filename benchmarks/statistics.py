import os
import re
import sys
from qiskit import QuantumCircuit

def get_unique_benchmarks(base_directory: str):
    unique_folders = set()
    benchmark_suites = os.listdir(base_directory)

    for benchmark_suite in benchmark_suites:
        if os.path.isdir(os.path.join(base_directory, benchmark_suite)):
            path = os.path.join(base_directory, benchmark_suite)
            for filename in os.listdir(path):
                if os.path.isdir(os.path.join(path, filename)):
                    unique_folders.add(filename)        

    return unique_folders

def count_all_benchmarks(base_directory: str, nqbits: tuple[int, int] = (2, 1000)):
    KV = {}

    for root, dirs, filenames in os.walk(base_directory):
        for filename in filenames:
            if filename.endswith(".qasm") or filename.endswith(".qasm3"):
                match = re.search(r'n(\d+)', filename)
                if match:
                    qubits = int(match.group(1))
                    if qubits >= nqbits[0] and qubits <= nqbits[1]:
                        if qubits in KV:
                            KV[qubits] = KV[qubits] + 1
                        else:
                            KV[qubits] = 1
    
    return sum(KV.values()), sorted(KV.items())

def count_ops(base_directory: str, nqbits: tuple[int, int] = (2, 1000), op: str = "cx"):
    KV = {}

    for root, dirs, filenames in os.walk(base_directory):
        for filename in filenames:
            if filename.endswith(".qasm") or filename.endswith(".qasm3"):
                match = re.search(r'n(\d+)', filename)
                if match:
                    qubits = int(match.group(1))
                    if qubits >= nqbits[0] and qubits <= nqbits[1]:                        
                        try:
                            circuit = QuantumCircuit.from_qasm_file(os.path.join(root, filename))
                            ops = circuit.count_ops()[op]

                            if ops in KV:
                                KV[ops] = KV[ops] + 1
                            else:
                                KV[ops] = 1
                        except:
                            print(os.path.join(root, filename))

    return sorted(KV.items())

def count_non_local_ops(base_directory: str, nqbits: tuple[int, int] = (2, 1000)):
    KV = {}

    for root, dirs, filenames in os.walk(base_directory):
        for filename in filenames:
            if filename.endswith(".qasm") or filename.endswith(".qasm3"):
                match = re.search(r'n(\d+)', filename)
                if match:
                    qubits = int(match.group(1))
                    if qubits >= nqbits[0] and qubits <= nqbits[1]:                        
                        try:
                            circuit = QuantumCircuit.from_qasm_file(os.path.join(root, filename))
                            ops = circuit.num_nonlocal_gates()

                            if ops in KV:
                                KV[ops] = KV[ops] + 1
                            else:
                                KV[ops] = 1
                        except:
                            print(os.path.join(root, filename))

    return sorted(KV.items())

print(get_unique_benchmarks("."))

nbenchmarks, distribution = count_all_benchmarks(".")
print(nbenchmarks, distribution)
print(count_non_local_ops("."))