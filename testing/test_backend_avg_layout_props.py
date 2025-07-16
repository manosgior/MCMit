from backends.backend import *
from backends.simulator import *

def test_find_optimal_qubit_set(backend, properties: list[str], bools: list[bool]):
    for b in bools:
        for prop in properties:
            qubits, value = find_optimal_qubit_set(backend, 7, get_average_property, prop, b)
            print(prop, "Min" if b else "Max", qubits, value)
        
        qubits, value = find_optimal_qubit_set(backend, 7, average_two_qubit_error, prop, b)
        print("ECR", "Min" if b else "Max", qubits, value)


properties = ["T1", "T2", "readout_error", "readout_length"]
bools = [True, False]

backend = getRealEagleBackend()
test_find_optimal_qubit_set(backend, properties, bools)