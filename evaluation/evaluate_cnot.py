from applications.long_range_CNOT import *
from backends.simulator import *
from analysis.fidelity import *
from analysis.distribution_processing import *

import argparse

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import transpile

parser = argparse.ArgumentParser()
parser.add_argument('n_reps', help='Upper limit in number of repetitions')
parser.add_argument('--nshots', help='Number of shots', default=8192)
args = parser.parse_args()

def test_on_ibm():
    circuits = generate_long_range_cnots(int(args.n_reps))

    #for c in circuits:
        #print(c.num_qubits)

    service = QiskitRuntimeService()
        
    n_shots = int(args.nshots)

    #simulator = getNoiselessSimulator()
    #backend = service.backend("ibm_fez")

    #noisy_simulator = simulatorFromBackend(backend)

    # Transpile all circuits
    #transpiled_circuits = [transpile(c, backend) for c in circuits]

    #sampler = Sampler(mode=backend)
    # Run all circuits in one job
    #job = sampler.run(transpiled_circuits, shots=n_shots)
    
    job_id = 'd40uc9kv6o9s73d08ta0'
    job = service.job(job_id)
    results = job.result()
    


    # Process results
    for i, c in enumerate(circuits):
        perfect = get_perfect_distribution_long_range_cnot(n_shots)
        #noisy = results.get_counts(i)
        #print(results[i].data)
        noisy = results[i].data.cr3.get_counts()
        #noisy = noisy_simulator.run(c, n_shots=n_shots).result().get_counts()
        

        #noisy = process_distribution_long_range_cnot(noisy)
        
        print(f"Long-range CNOT,{c.num_qubits}, Raw,{fidelity(perfect, noisy)}")

test_on_ibm()