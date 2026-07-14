from applications.constant_depth_GHZ import *
from backends.simulator import *
from evaluation.fidelity import *
from evaluation.distribution_processing import *

import mthree

import pickle

import argparse

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import transpile

from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

parser = argparse.ArgumentParser()
parser.add_argument('n_qubits', help='Upper limit in number of qubits')
parser.add_argument('--nshots', help='Number of shots', default=8192)
args = parser.parse_args()

LATENCY_MCMIT = 152.0  # Your fast, constant-time branch (e.g., 16ns + 34ns overhead)
LATENCY_QUBIC = 152.0 # Baseline (e.g., 130ns prep + 48ns for N=3 branch)

def get_decoherence_noise_model(real_backend, latency):
    """
    Creates a Qiskit Aer NoiseModel containing ONLY thermal relaxation
    (T1/T2) errors, based on a real backend's properties.
    """
    #print(f"Loading backend properties for '{backend_name}' to get T1/T2...")
    try:
        # TODO: Replace with your IBMProvider initialization
        # from qiskit_ibm_provider import IBMProvider
        # provider = IBMProvider()
        # backend = provider.get_backend(backend_name)
        #properties = real_backend.properties()

        noise_model = NoiseModel()
        
        # Build noise model from T1/T2 for all qubits
        for q in range(real_backend.num_qubits):
            t1 = real_backend.qubit_properties(q).t1
            t2 = real_backend.qubit_properties(q).t2
            # We must specify a time for the error. The 'delay' op will
            # be automatically scaled by the simulator. We add it for 1ns.
            error = thermal_relaxation_error(t1, t2, latency*1e-9)
            noise_model.add_quantum_error(error, ['delay'], [q])

        #print("Decoherence noise model created.")
        return noise_model
    
    except Exception as e:
        print(f"Error loading backend noise model: {e}")
        print("Using a generic, non-backend-based noise model.")
        noise_model = NoiseModel()
        error = thermal_relaxation_error(100e3, 70e3, 1, unit='ns') # 100us T1, 70us T2
        noise_model.add_quantum_error(error, ['delay'], list(range(20))) # Apply to 20 qubits
        return noise_model, None

def add_feedback_delays(circuit: QuantumCircuit, latency_ns: float) -> QuantumCircuit:
    """
    Inserts a delay after every conditional operation (MCM feedback).
    
    This function rebuilds the circuit, adding a 'delay' for all qubits
    to simulate the idle time during classical feedback processing.
    """
    new_qc = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)
    all_qubits = new_qc.qubits
    
    for instr in circuit.data:
        # Append the original instruction first
        new_qc.append(instr)
        
        # Check if the instruction was conditional (modern way)
        if hasattr(instr.operation, 'condition') and instr.operation.condition is not None:
            # This is a conditional operation, add the feedback delay
            new_qc.delay(latency_ns, all_qubits, unit='ns')
            
    return new_qc

def test_on_ibm():
    circuits = get_ghz_states(5, int(args.n_qubits))

    service = QiskitRuntimeService()
        
    n_shots = int(args.nshots)

    #simulator = getNoiselessSimulator()
    #backend = service.backend("ibm_fez")
   
    
    # Transpile all circuits
    #transpiled_circuits = [transpile(c, backend) for c in circuits]

    #simulator = simulatorFromBackend(backend)
    #mit = mthree.M3Mitigation(backend)
    #print("SENT CALIBRATION JOB")
    #mit.cals_from_file('calibrations.json')
    

    #sampler = Sampler(mode=backend)
    # Run all circuits in one job
    #job = sampler.run(transpiled_circuits, shots=n_shots)
    #print(f"Job ID: {job.job_id}")
    
    job_id = 'd40t35cv6o9s73d07o10'
    job = service.job(job_id)
    results = job.result()

    pubs = job.inputs['pubs'] 
    exec_circuits = [pub[0] for pub in pubs]

    new_job_id = 'd41qcn4h4j8s73egrp50'

    for i, c in enumerate(circuits):
        perfect = get_perfect_ghz_distribution(c.num_qubits, n_shots)
        #qubits = mthree.utils.final_measurement_mapping(exec_circuits[i])

        noisy = results[i].data.meas.get_counts() 

        print(f"Circuit {i} ({c.num_qubits} qubits) Fidelity:", fidelity(perfect, noisy))
    


test_on_ibm()