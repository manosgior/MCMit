from backends.backend import loadBackend, getRealEagleBackend
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
from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime import EstimatorV2 as Estimator

import numpy as np

def executor(circuit: QuantumCircuit, mode: str = "parity") -> float:
    backend = loadBackend("backends/QPUs/GuadalupeDQC_0.015")
    #backend = getRealEagleBackend()
    simulator = simulatorFromBackend(backend)
    tqc = transpile(circuit, backend)
    counts = simulator.run(tqc, shots=10000).result().get_counts()

    return calculateExpectationValue(counts, 10000, mode)

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
    return ["XX", "XpXm", "XY$"]

def applyDDQiskit(circuit: QuantumCircuit, sequence: str = "XX"):
    backend = loadBackend("backends/QPUs/GuadalupeDQC_0.015")
    tqc = transpile(circuit, backend)
    estimator = Estimator(mode=backend)

    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = sequence
    estimator.options.dynamical_decoupling.scheduling_method = "alap"

    counts = estimator.run(tqc, shots = 10000).result().get_counts()

    return calculateExpectationValue(counts, shots=10000)


def getZNEScaleFactors(low: float = 1.0, high: float = 9.0):
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


def applyZNE(circuit: QuantumCircuit, scale_factors: list[float], scale_method, factory, mode: str = "parity"):
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

def getLREdegrees(low: int = 1, high: int = 4):
    return list(range(low, high))

def getLREfoldMultipliers(low: int = 1, high: int = 4):
    return list(range(low, high))

def applyLRE(circuit: QuantumCircuit, degree: int, fold_multiplier: int, mode: str = "parity"):
    mitigated_result = execute_with_lre(
        circuit,
        executor,
        degree=degree,
        fold_multiplier=fold_multiplier,
    )

    return mitigated_result

def applyPauliTwirling(circuit: QuantumCircuit, num_variants: int, mode: str = "parity"):
    twirled_circuits = generate_pauli_twirl_variants(circuit, num_circuits=num_variants)
    results = [executor(circuit) for circuit in twirled_circuits]

    mitigated_result = np.average(results)

    return mitigated_result


def applyPauliTwirlingQiskit(circuit: QuantumCircuit, num_randomizations: int, shots_per_randomization: int):
    backend = loadBackend("backends/QPUs/GuadalupeDQC_0.015")
    tqc = transpile(circuit, backend)
    estimator = Estimator(mode=backend)

    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = num_randomizations
    estimator.options.twirling.shots_per_randomization = shots_per_randomization

    counts = estimator.run(tqc, shots = 10000).result().get_counts()

    return calculateExpectationValue(counts, shots=10000)

def applyMeasureMitigationQiskit(circuit: QuantumCircuit, num_randomizations: int, shots_per_randomization: int):
    backend = loadBackend("backends/QPUs/GuadalupeDQC_0.015")
    tqc = transpile(circuit, backend)
    estimator = Estimator(mode=backend)

    estimator.options.resilience.measure_mitigation = True
    estimator.options.resilience.measure_noise_learning.num_randomizations = num_randomizations
    estimator.options.resilience.measure_noise_learning.shots_per_randomization = shots_per_randomization

    counts = estimator.run(tqc, shots = 10000).result().get_counts()

    return calculateExpectationValue(counts, shots=10000)