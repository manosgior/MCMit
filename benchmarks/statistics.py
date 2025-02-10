import os
import re
import sys
from collections import defaultdict, OrderedDict

from qiskit import QuantumCircuit

def get_unique_benchmarks(dirs: list[str]):
    unique_folders = set()
    sizes = defaultdict(int)
    counter = 0

    for d in dirs:
        for folder in os.listdir(d):
            folder_path = os.path.join(d, folder)
            if os.path.isdir(folder_path):
                unique_folders.add(folder)

    for d in dirs:
         for root, dirs, files in os.walk(d):
            for f in files:
                for file in files:
                    match = re.match(r'n(\d+)\.qasm$', file)
                    if match:
                        number = int(match.group(1))
                        sizes[number] += 1  
                        counter += 1       

    return len(unique_folders), OrderedDict(sorted(sizes.items())), counter


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

print(get_unique_benchmarks(["benchmarks/MQTBench", "benchmarks/QASMBench", "benchmarks/QOSLib", "benchmarks/Supermarq"]))
#print(get_unique_benchmarks(["benchmarks/QOSLib"]))
#nbenchmarks, distribution = count_all_benchmarks(".")
#print(nbenchmarks, distribution)
#print(count_non_local_ops("."))