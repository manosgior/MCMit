from applications.constant_depth_GHZ import *
from compiler.decoding.adaptive_soft_decoding import *

from backends.backend import getRealEagleBackend
from backends.simulator import *

def test_parity_checks(min_circuit_size: int, max_circuitsize: int, backend, num_reps: int = 5, shots: int = 8192):
    circuits = get_ghz_states(min_circuit_size, max_circuitsize)

    for circuit in circuits:
        encoded_circuit = add_parity_checks_greedy(circuit, backend)

        print(encoded_circuit)
        exit()

        for i in range(num_reps):
            result = backend.run(encoded_circuit, shots=shots).result()
            counts = result.get_counts()

            

backend = simulatorFromBackend(getRealEagleBackend())
test_parity_checks(5, 6, backend, num_reps=5, shots=8192)