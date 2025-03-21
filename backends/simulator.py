from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer.noise import NoiseModel

from backends.backend import loadBackend, getRealEagleBackend

def simulatorFromBackend(backend: GenericBackendV2):
    coupling_map = backend.coupling_map

    if hasattr(backend, "_basis_gates"):
        basis_gates = backend._basis_gates
    elif hasattr(backend, "configuration"):
        basis_gates = backend.configuration().supported_instructions
    noise_model = NoiseModel.from_backend(backend)

    simulator = AerSimulator(noise_model=noise_model, coupling_map=coupling_map, basis_gates=basis_gates)
    
    return simulator

def getNoiselessSimulator():
    return AerSimulator()