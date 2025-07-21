from applications.constant_depth_GHZ import *
from applications.quantum_teleportation import *
from applications.long_range_CNOT import *
from applications.qasm3_exporter import export_qasm3

from backends.simulator import *
from backends.backend import *

from analysis.fidelity import *
from analysis.distribution_processing import *

from evaluation.util import construct_results_data

import numpy as np
import pandas as pd
import os

from qiskit.transpiler.passes import VF2Layout

distribution_post_processing_mapping = {
    "constant_depth_GHZ": process_distribution_ghz,
    "quantum_teleportation_ladder": process_distribution_teleportation,
    "long_range_CNOT": process_distribution_long_range_cnot
}

optimal_result_mapping = {
    "constant_depth_GHZ": get_perfect_ghz_distribution,
    "quantum_teleportation_ladder": get_perfect_distribution_teleportation,
    "long_range_CNOT": get_perfect_distribution_long_range_cnot
}

def evaluate_layout_effect_on_fidelity(circuits: list[QuantumCircuit], type: str, backend, num_reps: int = 5, shots: int = 8192, filename: str = 'evaluation/motivation/results/scalability_analysis.csv'):
    for j, c in enumerate(circuits):     

        layout_pass = VF2Layout(target = backend.target)
        dag = circuit_to_dag(c)
        layout_pass.run(dag)
        layout = list(layout_pass.property_set['layout'].get_physical_bits().keys())

        small_backend = create_backend_from_layout(backend, layout)
        simulator = simulatorFromBackend(small_backend)  

        fidelities = []
        for i in range(num_reps):
            tqc = transpile(c, backend=small_backend, optimization_level=3)
            results = simulator.run(tqc, shots=shots).result()
            counts = results.get_counts()

            processed_counts = distribution_post_processing_mapping[type](counts)
            optimal_result = optimal_result_mapping[type](shots)

            fid = fidelity(processed_counts, optimal_result)
            
            fidelities.append(fid)
        
        avg_fidelity = np.mean(fidelities)
        fidelity_std = np.std(fidelities)

        data = construct_results_data(
            application=type + "_" + str(j + 1),
            num_qubits=c.num_qubits,
            backend=backend.name,
            num_reps=num_reps,
            num_shots=shots,
            layout_mode="qiskit_layout",
            fidelity=avg_fidelity,
            fidelity_std=fidelity_std
        )
        df = pd.DataFrame([data])
        df.to_csv(filename, mode='a', header=not os.path.isfile(filename), index=False)

ghz_circuit = create_constant_depth_ghz(5)
teleportation_circuit = create_teleportation_circuit()
long_range_cnot = create_dynamic_CNOT_circuit(5)

export_qasm3(ghz_circuit, 'applications/qasm_files/constant_depth_ghz.qasm3')
export_qasm3(teleportation_circuit, 'applications/qasm_files/quantum_teleportation.qasm3')
export_qasm3(long_range_cnot, 'applications/qasm_files/long_range_CNOT.qasm3')

exit()

constant_depth_ghz = get_ghz_states(5, 11)
teleportation_circuits_ladder = generate_repeated_teleportations(5, is_ladder=True)
long_range_cnot_circuits = generate_long_range_cnots(13)

backend = getRealEagleBackend()

evaluate_layout_effect_on_fidelity(constant_depth_ghz, "constant_depth_GHZ", backend, num_reps=5, shots=8192)
evaluate_layout_effect_on_fidelity(teleportation_circuits_ladder, "quantum_teleportation_ladder", backend, num_reps=5, shots=8192) 
evaluate_layout_effect_on_fidelity(long_range_cnot_circuits, "long_range_CNOT", backend, num_reps=5, shots=8192)