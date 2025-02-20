from backends.backend import loadBackend, getRealEagleBackend, customBackend
from backends.simulator import simulatorFromBackend
from analysis.fidelity import calculateExpectationValue

from mitiq import MeasurementResult
from mitiq import ddd
from mitiq import zne
from mitiq.zne.scaling import fold_gates_at_random, fold_all, fold_global
from mitiq.zne import inference
from mitiq.zne import combine_results, scaled_circuits
from mitiq.rem import generate_inverse_confusion_matrix, generate_tensored_inverse_confusion_matrix
from mitiq import rem
#from mitiq.raw import execute
from mitiq.lre import execute_with_lre
from mitiq.pt import generate_pauli_twirl_variants

from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.options import estimator_options

import numpy as np
from typing import Union

def executor(circuit: QuantumCircuit, mode: str = "parity") -> float:
    backend = loadBackend("backends/QPUs/GuadalupeDQC_0.015")
    #backend = getRealEagleBackend()
    simulator = simulatorFromBackend(backend)
    tqc = transpile(circuit, backend)
    counts = simulator.run(tqc, shots=10000).result().get_counts()

    return calculateExpectationValue(counts, 10000, mode)

def getEstimatorFromBackend(backend: customBackend = loadBackend("backends/QPUs/GuadalupeDQC_0.015")):
    return Estimator(mode=backend, options={"default_shots": 10000})

def estimatorExecutor(circuit: QuantumCircuit, estimator: Estimator) -> float:
    backend = estimator.backend()
    tqc = transpile(circuit, backend)
    estimator = estimator

    global_parity_observable = SparsePauliOp("Z" * circuit.num_qubits)
    result = estimator.run([(tqc, global_parity_observable)]).result()[0]

    return result

def getDDDrule(type: str = "xx"):
    assert type in ["xx", "yy", "xyxy"]

    return getattr(ddd, type, None)

def applyDDD(circuit: QuantumCircuit, rule: ddd.rules, mode: str = "parity") -> float:
    mitigated_result = ddd.execute_with_ddd(
        circuit=circuit,
        executor=executor,
        rule=rule,
    )
    return mitigated_result

def getDDQiskitSequences():
    return ["XX", "XpXm", "XY4"]

def applyDDQiskit(estimator: Estimator, sequence: str = "XX"):
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = sequence
    estimator.options.dynamical_decoupling.scheduling_method = "alap"

def getZNEScaleFactors(low: int = 1, high: int = 9):
    return list(range(low, high, 2))

def validZNEFactories():
    return ["LinearFactory", "RichardsonFactory", "PolyFactory", "ExpFactory", "PolyExpFactory", "AdaExpFactory"]

def getZNEFactories(factories: list[str] = validZNEFactories()):
    to_return = []
    valid_zne_factories = validZNEFactories()

    for f in factories:
        assert(f in valid_zne_factories)
        to_return.append(getattr(inference, f, None))

    return to_return

def getZNEFoldingMethods():
    return [fold_gates_at_random, fold_global, fold_all]


def applyZNE(circuit: QuantumCircuit, scale_factors: list[int] = getZNEScaleFactors(), scale_method = fold_gates_at_random, factory = inference.RichardsonFactory, mode: str = "parity"):
    folded_circuits = scaled_circuits(
        circuit=circuit,
        scale_factors=scale_factors,
        scale_method=scale_method,
    )

    results = [executor(circuit) for circuit in folded_circuits]

    extrapolation_method = factory(scale_factors=scale_factors).extrapolate
    two_stage_zne_result = combine_results(
        scale_factors, results, extrapolation_method
    )

    return two_stage_zne_result
    #zne.execute_with_zne(circuit=circuit, executor=executor)

def getZNEQiskitExtrapolationMethods():
    return ['linear', 'exponential', 'double_exponential', 'polynomial_degree_1', 'polynomial_degree_2', 'polynomial_degree_3', 'polynomial_degree_4', 'polynomial_degree_5', 'polynomial_degree_6', 'polynomial_degree_7', 'fallback']

def applyZNEQiskit(estimator: Estimator, noise_factors: list[float] = getZNEScaleFactors(), factory: str = "linear"):
    estimator.options.resilience.zne_mitigation = True
    estimator.options.resilience.zne.noise_factors = noise_factors
    estimator.options.resilience.zne.extrapolator = factory

def getLREdegrees(low: int = 1, high: int = 4):
    return list(range(low, high))

def getLREfoldMultipliers(low: int = 1, high: int = 4):
    return list(range(low, high))

def applyLRE(circuit: QuantumCircuit, degree: int = 2, fold_multiplier: int = 2, mode: str = "parity"):
    mitigated_result = execute_with_lre(
        circuit,
        executor,
        degree=degree,
        fold_multiplier=fold_multiplier,
    )

    return mitigated_result

def applyPauliTwirling(circuit: QuantumCircuit, num_variants: int = 10, mode: str = "parity"):
    twirled_circuits = generate_pauli_twirl_variants(circuit, num_circuits=num_variants)
    results = [executor(circuit) for circuit in twirled_circuits]

    mitigated_result = np.average(results)

    return mitigated_result


def applyPauliTwirlingQiskit(estimator: Estimator, num_randomizations: int = 32, shots_per_randomization: int = 100):
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = num_randomizations
    estimator.options.twirling.shots_per_randomization = shots_per_randomization

def applyMeasureMitigationQiskit(estimator: Estimator, num_random: int = 32, shots_per_random: int = 100):
    estimator.options.resilience.measure_mitigation = True
    estimator.options.resilience.measure_noise_learning.num_randomizations = num_random
    estimator.options.resilience.measure_noise_learning.shots_per_randomization = shots_per_random

def applyPECQiskit(estimator: Estimator, max_overhead: int = 100):
    estimator.options.resilience.pec_mitigation = True
    estimator.options.resilience.pec.max_overhead = max_overhead

def validEMOptions():
    return {
        "DD" : applyDDQiskit,
        "ZNE" : applyZNEQiskit,
        "pauli_twirling": applyPauliTwirlingQiskit,
        "measure_mitigation": applyMeasureMitigationQiskit,
        "PEC": applyPECQiskit,
    }

def applyErrorMitigationQiskit(circuit: QuantumCircuit, em_techniques: list[str] = ["DD", "ZNE", "pauli_twirling", "measure_mitigation"], **kwargs):
    estimator = getEstimatorFromBackend()
    valid_em_techniques = validEMOptions()

    for em in em_techniques:
        if em not in valid_em_techniques:
            raise ValueError(f"Unknown technique: {em}")

        if em == "DD":
            if kwargs.get("sequence") == None:
                valid_em_techniques[em](estimator)
            else:
                valid_em_techniques[em](estimator, kwargs.get("sequence"))
        elif em == "ZNE":
            if kwargs.get("noise_factors") == None or kwargs.get("factory") == None:
                valid_em_techniques[em](estimator)
            else:
                valid_em_techniques[em](estimator, kwargs.get("noise_factors"), kwargs.get("factory"))
        elif em == "pauli_twirling":
            if kwargs.get("num_randomizations") == None or kwargs.get("shots_per_randomization") == None:
                valid_em_techniques[em](estimator)
            else:
                valid_em_techniques[em](estimator, kwargs.get("num_randomizations"), kwargs.get("shots_per_randomization"))
        elif em == "measure_mitigation":
            if kwargs.get("num_randomi") == None or kwargs.get("shots_per_random") == None:
                 valid_em_techniques[em](estimator)
            else:
                if kwargs.get("num_random") == None or kwargs.get("shots_per_random") == None:
                    valid_em_techniques[em](estimator)
                else:
                    valid_em_techniques[em](estimator, kwargs.get("num_random"), kwargs.get("shots_per_random"))
        elif em == "PEC":
            if kwargs.get("max_overhead") == None:
                valid_em_techniques[em](estimator)
            else:
                valid_em_techniques[em](estimator, kwargs.get("max_overhead"))            

    return estimatorExecutor(circuit, estimator)