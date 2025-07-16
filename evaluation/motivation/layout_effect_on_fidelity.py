from backends.backend import *
from backends.simulator import *

from applications.constant_depth_GHZ import *
from applications.quantum_teleportation import *

from analysis.fidelity import *
from analysis.distribution_processing import *

import numpy as np
import pandas as pd
import os

from qiskit.transpiler import Layout
from qiskit.transpiler.passes import FullAncillaAllocation

def construct_data(application: str, num_qubits: int, backend: str, num_reps: int, num_shots: int, layout_mode: str, fidelity: float, fidelity_std: float) -> list[dict]:
    data = {
            'application': application,
            'num_qubits': num_qubits,
            'backend': backend,
            'num_reps': num_reps,
            'num_shots': num_shots,
            'layout_mode': layout_mode,
            'fidelity': fidelity,
            'fidelity_std': fidelity_std
    }
    return data

def evaluate_layout_effect_on_fidelity_ghz(min_circuit_size: int, max_circuitsize: int, backend, num_reps: int = 7, shots: int = 8192, filename: str = 'evaluation/motivation/results/results.csv'):
    circuits = get_ghz_states(min_circuit_size, max_circuitsize)

    for c in circuits:
        print(c)
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
                processed_counts = process_distribution(counts)
                print(counts)
                print(processed_counts)
                fid = fidelity(processed_counts, get_perfect_ghz_distribution(c.num_qubits, shots))
                fidelities.append(fid)
            
            avg_fidelity = np.mean(fidelities)
            fidelity_std = np.std(fidelities)
            print(avg_fidelity)
            exit()

            data = construct_data(
                application="constant_depth_GHZ",
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
circuit_sizes = (7, 24)
evaluate_layout_effect_on_fidelity_ghz(circuit_sizes[0], circuit_sizes[1], backend, num_reps=7, shots=8192)
