def construct_results_data(application: str, num_qubits: int, backend: str, num_reps: int, num_shots: int, layout_mode: str, fidelity: float, fidelity_std: float) -> list[dict]:
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