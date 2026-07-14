from applications.quantum_teleportation import *
from backends.simulator import *
from evaluation.fidelity import *
from evaluation.distribution_processing import *

import argparse

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import transpile


parser = argparse.ArgumentParser()
parser.add_argument('n_reps', help='Upper limit in number of repetitions')
parser.add_argument('--nshots', help='Number of shots', default=8192)
parser.add_argument('--is_ladder', help="Ladder this consecutive circuit", default=False)
args = parser.parse_args()

def test_on_ibm():
    #circuits = generate_repeated_teleportations(int(args.n_reps), 1, args.is_ladder)
    circuits = generate_repeated_teleportations(int(args.n_reps), 2,)
    print(circuits[0])
    service = QiskitRuntimeService()
        
    n_shots = int(args.nshots)

    #simulator = getNoiselessSimulator()
    #backend = service.backend("ibm_fez")

    #noisy_simulator = simulatorFromBackend(backend)

    # Transpile all circuits
    #transpiled_circuits = [transpile(c, backend) for c in circuits]

    #sampler = Sampler(mode=backend)
    # Run all circuits in one job
    
    job_id_tp = 'd40u29sv6o9s73d08kbg'
    job_id_tp_ladder = 'd40u5tcv6o9s73d08nhg'

    job = service.job(job_id_tp)
    results = job.result()

    # Process results
    for i, c in enumerate(circuits):
        perfect = get_perfect_distribution_teleportation(n_shots)
        #noisy = results.get_counts(i)
        #noisy = noisy_simulator.run(c, n_shots=n_shots).result().get_counts()
        noisy = results[i].data.final.get_counts() 
        print(noisy)

        #noisy = process_distribution_teleportation(noisy)
        
        print(f"Repeated teleportation,{i*2 + 1},Raw,{fidelity(perfect, noisy)}")

test_on_ibm()