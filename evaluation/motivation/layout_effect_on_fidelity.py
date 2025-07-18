from backends.backend import *
from backends.simulator import *

from applications.constant_depth_GHZ import *
from applications.quantum_teleportation import *
from applications.long_range_CNOT import *

from analysis.fidelity import *
from analysis.distribution_processing import *

from evaluation.util import construct_results_data

import numpy as np
import pandas as pd
import os


distribution_post_processing_mapping = {
    "constant_depth_GHZ": process_distribution_ghz,
    "quantum_teleportation": process_distribution_teleportation,
    "quantum_teleportation_ladder": process_distribution_teleportation,
    "long_range_CNOT": process_distribution_long_range_cnot
}

optimal_result_mapping = {
    "constant_depth_GHZ": get_perfect_ghz_distribution,
    "quantum_teleportation": get_perfect_expectation_value_hardcoded,
    "quantum_teleportation_ladder": get_perfect_expectation_value_hardcoded,
    "long_range_CNOT": get_perfect_distribution_long_range_cnot
}

def evaluate_layout_effect_on_fidelity_teleportation(ensemble_circuits: list[QuantumCircuit], type: str, backend, num_reps: int = 5, shots: int = 8192, filename: str = 'evaluation/motivation/results/layout_impact.csv'):
    properties = ["T1", "T2", "readout_error", "2q-error"]
    bools = [True, False]
    layouts = get_optimal_layouts(backend, ensemble_circuits[0].num_qubits, properties, bools)  

    perfect_expectation_values = get_perfect_expectation_value_hardcoded()

    for layout_name, layout in layouts.items():
        small_backend = create_backend_from_layout(backend, layout[0])
        simulator = simulatorFromBackend(small_backend)

        expectation_value_axis = []
        for c in ensemble_circuits:
            tqc = transpile(c, backend=small_backend, optimization_level=3)

            expectation_values = []
            for i in range(num_reps):                
                results = simulator.run(tqc, shots=shots).result()
                counts = results.get_counts()

                processed_counts = process_distribution_teleportation(counts)
                expectation_values.append(calculateExpectationValue(processed_counts, shots, "sum"))
            
            average_expectation_value = np.mean(expectation_values)
            expectation_value_axis.append(average_expectation_value)
    
        fidelity = getEVFidelityWithBasis(expectation_value_axis[0], perfect_expectation_values[0], expectation_value_axis[1], perfect_expectation_values[1], expectation_value_axis[2], perfect_expectation_values[2])

        data = construct_results_data(
            application= type,
            num_qubits=c.num_qubits,
            backend=backend.name,
            num_reps=num_reps,
            num_shots=shots * 3,
            layout_mode=layout_name,
            fidelity=fidelity,
            fidelity_std=0
        )
        df = pd.DataFrame([data])
        df.to_csv(filename, mode='a', header=not os.path.isfile(filename), index=False)

def evaluate_layout_effect_on_fidelity(circuits: list[QuantumCircuit], type: str, backend, num_reps: int = 5, shots: int = 8192, filename: str = 'evaluation/motivation/results/layout_impact.csv'):
    for j, c in enumerate(circuits):
        properties = ["T1", "T2", "readout_error", "2q-error"]
        bools = [True, False]
        layouts = get_optimal_layouts(backend, c.num_qubits, properties, bools)  

        for layout_name, layout in layouts.items():
            small_backend = create_backend_from_layout(backend, layout[0])
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
                layout_mode=layout_name,
                fidelity=avg_fidelity,
                fidelity_std=fidelity_std
            )
            df = pd.DataFrame([data])
            df.to_csv(filename, mode='a', header=not os.path.isfile(filename), index=False)


backend = getRealEagleBackend()
#teleportation_circuits = generate_repeated_teleportations(10, is_ladder=False)
#teleportation_circuits_ladder = generate_repeated_teleportations(10, is_ladder=True)
#long_range_cnot_circuits = generate_long_range_cnots(13)

#for i in range(2, 11):
 #   circuits = [create_repeated_teleportation_circuit(i, measurement_basis='Z'), create_repeated_teleportation_circuit(i, measurement_basis='X'), create_repeated_teleportation_circuit(i, measurement_basis='Y')]
  #  evaluate_layout_effect_on_fidelity_teleportation(circuits, "quantum_teleportation_" + str(i), backend, num_reps=7, shots=2730)

for i in range(2, 5):
    circuits = [create_ladder_teleportation_circuit(i, measurement_basis='Z'), create_repeated_teleportation_circuit(i, measurement_basis='X'), create_repeated_teleportation_circuit(i, measurement_basis='Y')]
    evaluate_layout_effect_on_fidelity_teleportation(circuits, "quantum_teleportation_ladder_" + str(i), backend, num_reps=7, shots=2730)


#evaluate_layout_effect_on_fidelity(teleportation_circuits, "quantum_teleportation", backend, num_reps=7, shots=8192)
#evaluate_layout_effect_on_fidelity(teleportation_circuits_ladder, "quantum_teleportation_ladder", backend, num_reps=7, shots=8192)
#evaluate_layout_effect_on_fidelity(long_range_cnot_circuits, "long_range_CNOT", backend, num_reps=7, shots=8192)

