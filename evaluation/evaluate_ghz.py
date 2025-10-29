from applications.constant_depth_GHZ import *
from backends.simulator import *
from analysis.fidelity import *
from analysis.distribution_processing import *

from compiler.branching import *
from compiler.decoding.repeated_measurements import *
from compiler.decoding.adaptive_soft_decoding import *

import argparse

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import transpile

parser = argparse.ArgumentParser()
parser.add_argument('n_qubits', help='Upper limit in number of qubits')
parser.add_argument('--nshots', help='Number of shots', default=8192)
args = parser.parse_args()

def test_on_ibm():
    circuits = get_ghz_states(5, int(args.n_qubits))

    service = QiskitRuntimeService()
        
    n_shots = int(args.nshots)

    simulator = getNoiselessSimulator()
    backend = service.backend("ibm_fez")

    #noisy_simulator = simulatorFromBackend(backend)

    # Transpile all circuits
    transpiled_circuits = [transpile(c, backend) for c in circuits]

    sampler = Sampler(mode=backend)
    # Run all circuits in one job
    job = sampler.run(transpiled_circuits, shots=n_shots)
    print(f"Job ID: {job.job_id}")
    results = job.result()

    job_id = 'd40t35cv6o9s73d07o10'

    # Process results
    for i, c in enumerate(circuits):
        perfect = get_perfect_ghz_distribution(c.num_qubits, n_shots)
        noisy = results.get_counts(i)
        
        #perfect = process_distribution_ghz(perfect)
        noisy = process_distribution_ghz(noisy)
        
        print(f"Circuit {i} ({c.num_qubits} qubits) Fidelity:", fidelity(perfect, noisy))
    
def test_with_parity_checks():
    circuits = get_ghz_states(5, int(args.n_qubits))

    for i,c in enumerate(circuits):
        cc = add_redundant_measurements_with_logic(c)
        cc.draw(output='mpl', filename=f'my_circuit_{cc.num_qubits}.png')

test_with_parity_checks()