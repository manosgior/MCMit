from qiskit.transpiler import Target, InstructionProperties, CouplingMap
from qiskit.circuit import QuantumCircuit
from qiskit import transpile
#import qiskit.circuit.library as Gates
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.providers import QubitProperties
from qiskit_ibm_runtime import QiskitRuntimeService

#from itertools import islice
from collections import defaultdict

import matplotlib.pyplot as plt
#import seaborn as sns
import networkx as nx
#from rustworkx import EdgeList
#from networkx.algorithms import bipartite
import numpy as np
import pickle
#import sys

from plotting import utils


def defaultGateSet():
    return ["id", "sx", "x", "rz", "rzz", "cz", "rx"]

def defaultResolutionTime():
    return 2.2e-10

def GuadalupeCouplingMap():
    return CouplingMap([[0, 1], [1, 4], [4, 7], [6, 7], [7, 10], [10, 12], [12, 15], [1, 2], [2, 3], [3, 5], [5, 8], [8, 11], [11, 14], [12, 13], [13, 14], [8,9]])

def getRealNoiseModelsFromEagle():
    service = QiskitRuntimeService(instance="ibm-q/open/main")
    backend = service.backend("ibm_kyiv")
    qubit_properties = [backend.qubit_properties(i) for i in range(backend.num_qubits)]

    return qubit_properties, backend.target

def getRealResolutionTimeFromEagle():
    service = QiskitRuntimeService(instance="ibm-q/open/main")
    backend = service.backend("ibm_kyiv")

    return backend.dt

def heavyHexEagleCouplingMap():
    service = QiskitRuntimeService(instance="ibm-q/open/main")
    backend = service.backend("ibm_kyiv")
    #backend = service.backend("ibm_sherbrooke")

    return backend.coupling_map

def heavySquareHeronCouplingMap():
    service = QiskitRuntimeService(instance="ibm-q/open/main")
    backend = service.backend("ibm_kyiv")

    base_coupling_map = backend.coupling_map

    base_coupling_map.add_edge(13, 127)
    base_coupling_map.add_edge(113, 128) 
    base_coupling_map.add_edge(117, 129)
    base_coupling_map.add_edge(121, 130)
    base_coupling_map.add_edge(124, 131)
    base_coupling_map.add_edge(128, 132)

    return base_coupling_map

def DQCCouplingMap(coupling_map1: CouplingMap, coupling_map2: CouplingMap, endpoints: list):
    edges_1 = coupling_map1.get_edges()
    last_qubit = coupling_map1.size()
    converted_edges_1 = [[i, j] for i,j in edges_1]
    new_endpoints = [[i, last_qubit + j] for i,j in endpoints]
    new_coupling_map2 = [[last_qubit + i, last_qubit + j] for i,j in coupling_map2]
    
    new_coupling_map = CouplingMap(converted_edges_1 + new_endpoints + new_coupling_map2)
    new_endpoints_tuple = [(i, j) for i,j in new_endpoints]

    return new_endpoints_tuple, new_coupling_map

def heavyHexSmallLayers():
    return {"a": [6, 22], "b": [0, 1, 4, 7, 10, 12, 15, 16, 17, 20, 23, 26, 28, 31], "c": [2, 13, 18, 29], "d": [3, 5, 8, 11, 14, 19, 21, 24, 27, 30], "e": [9, 25]}

def heavyHexLargeLayers(coupling_map: CouplingMap):
    layers = {}
    counter = 0
    edges = coupling_map.get_edges()

    chains = []
    edges = sorted(edges)
    
    while edges:
        chain = [edges.pop(0)]
        while edges:
            for i, (a, b) in enumerate(edges):
                if chain[-1][1] == a:
                    chain.append((a, b))
                    edges.pop(i)
                    break
            else:
                break
        chains.append(chain)

    print(chains)
        
    return chains

def printCouplingMap(coupling_map: CouplingMap, layers: dict):
    edges = coupling_map.get_edges()
    G = nx.Graph()
    G.add_edges_from(edges)

    pos = nx.multipartite_layout(G, subset_key=layers, align='horizontal')

    nx.draw(G, with_labels=True, node_color="skyblue", edge_color="gray", pos=pos)
    plt.savefig("coupling_map.png", dpi=300, bbox_inches="tight")


class customBackend(GenericBackendV2):    
    
    def __init__(self, name: str ="customDQCBackend", num_qubits: int = 16, coupling_map: CouplingMap = GuadalupeCouplingMap(), dt: float = defaultResolutionTime(),basis_gates: list[str] = defaultGateSet(), noise_model: Target = None, qubit_properties: list[QubitProperties] = None, remote_gates: list[tuple] = []):
        assert num_qubits == coupling_map.size()
        super().__init__(num_qubits, basis_gates=basis_gates, coupling_map=coupling_map, control_flow=True, seed=1)
        self.name = name
        self.remote_gates = remote_gates
        self.dt = dt

        if noise_model == None:
            self.addStateOfTheArtNoise()
        else:
            for instruction, properties_dict in noise_model.items():
                if properties_dict is not None:
                    for p in properties_dict.items():
                        self.target.update_instruction_properties(instruction, p[0], p[1])

        if qubit_properties == None:
            self.addStateOfTheArtQubits()
        else: 
            self.target.qubit_properties = qubit_properties

    def updateGateProps(self, duration: int = 70, error_med: float = 0.002, error_min: float = 0.0001, error_max: float = 0.06, gate: str = "cz"):
        gates = self.target[gate].items()
        target_ratio = (error_med - error_min) / (error_max - error_min)
        alpha = 0.3
        beta = alpha * (1 - target_ratio) / target_ratio
        beta = 3
        values = []

        for g in gates:
            duration_rand = np.random.randint(duration - 10, duration + 10)
           
            error_rand = np.random.beta(alpha, beta)
            error_rand = error_min + (error_max - error_min) * error_rand
            values.append(error_rand)

            self.target.update_instruction_properties(gate, g[0], InstructionProperties(duration_rand, error_rand))

        #print("--------------------")
        #print(np.median(values))

    def addStateOfTheArtNoise(self): 

        for g in ["id", "sx", "x", "rz", "rx"]:
            self.updateGateProps(gate=g, duration=50, error_med=0.00025, error_min=0.0001, error_max=0.015)

        for g in ["cz", "rzz"]:
            self.updateGateProps(gate=g, duration=70, error_med=0.002, error_min=0.0009, error_max=0.06)

        self.updateGateProps(gate="measure", duration=70, error_med=0.01, error_min=0.002, error_max=0.5)

    def addNoiseDelayToRemoteGates(self, endpoints: list[tuple], duration: int = 300, error: float = 0.03, gates: list[str] = ["cz", "rzz"]):
        gate_props = InstructionProperties(duration, error)
        
        for g in gates:
            for e in endpoints:
                self.target.update_instruction_properties(g, e, gate_props)

    def addStateOfTheArtQubits(self):
        qubits = []
        
        for i in range(self.num_qubits):
            t1 = np.random.normal(190, 100, 1)
            t1 = np.clip(t1, 10, 500)

            t2 = np.random.normal(130, 100, 1)
            t2 = np.clip(t2, 10, 650)

            qubits.append(QubitProperties(t1=t1, t2=t2, frequency=5.0e9))

        self.target.qubit_properties = qubits

    def distance(self, q0: int, q1: int):
        return self.coupling_map.distance(q0, q1)

    def plotGateProbDistribution(self):
        gate_error_dict = defaultdict(list)

        group_1 = {"rx", "ry", "rz", "h", "x", "y", "z", "s", "sdg", "t", "sx", "id"}  # Single-qubit gates
        group_2 = {"cx", "ecr", "cz", "rzz"}  # Two-qubit gates
        group_3 = {"measure"}  # measurement

        group_titles = {
            1: "Single-Qubit Gates",
            2: "Two-Qubit Gates",
            3: "Measurements"
        }

        error_groups = {1: defaultdict(list), 2: defaultdict(list), 3: defaultdict(list)}

        for gate, inst_map in self.target.items():
            for qubits, properties in inst_map.items():
                if properties is not None and properties.error is not None:
                    if gate in group_1:
                        error_groups[1][gate].append(properties.error)
                    elif gate in group_2:
                        error_groups[2][gate].append(properties.error)
                    else:
                        error_groups[3][gate].append(properties.error) 
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        colors = utils.colors_deep
        markers = utils.line_markers

        for i, (group, gate_error_dict) in enumerate(error_groups.items(), start=1):
            ax = axes[i - 1] 
            for j, (gate, errors) in enumerate(gate_error_dict.items(), start=1):
                if errors: 
                    sorted_errors = np.sort(errors)
                    median = np.median(sorted_errors)
                    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                    ax.plot(sorted_errors, cdf, marker=markers[j], linestyle="-", label=str(gate), color=colors[j])
                    ax.axvline(median, marker=markers[j], color=colors[j])

            ax.set_xlabel("Gate Error Probability")
            ax.set_title(group_titles[i])
            ax.set_xscale("log")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(title="Gate Type")

        axes[0].set_ylabel("Cumulative Probability")
        plt.tight_layout()
        plt.savefig("backends/" + self.name + "_gate_probs.png")

def saveBackend(backend: customBackend, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(backend, f)

def loadBackend(filename: str):
    with open(filename, "rb") as f:
        backend = pickle.load(f)

    return backend

def constructDQCSmall(noise: float = 0.03): 
    endpoints, map_small = DQCCouplingMap(GuadalupeCouplingMap(), GuadalupeCouplingMap(), [[13, 2], [15, 0]])
    backend_small = customBackend(name="GuadalupeDQC_", num_qubits=32, coupling_map=map_small, remote_gates=endpoints)
    backend_small.addNoiseDelayToRemoteGates(endpoints, error=noise)

    return backend_small

def constructDQCMedium(noise: float = 0.03):
    endpoints, map_medium = DQCCouplingMap(heavyHexEagleCouplingMap(), heavyHexEagleCouplingMap(), [[32, 18], [51, 37], [70, 56], [89, 75]])
    #props_medium, noise_model_medium = getRealNoiseModelsFromEagle()
    #backend_medium = customBackend(name="KyivDQC_",num_qubits=254, coupling_map=map_medium, noise_model=noise_model_medium, qubit_properties=props_medium, basis_gates=["ecr", "id", "rz", "sx", "x"])
    backend_medium = customBackend(name="KyivDQC_",num_qubits=254, coupling_map=map_medium, remote_gates=endpoints)
    backend_medium.addNoiseDelayToRemoteGates(endpoints, error=noise)

    return backend_medium

def constructDQCLarge(noise: float = 0.03):
    endpoints, map_interm = DQCCouplingMap(heavySquareHeronCouplingMap(), heavySquareHeronCouplingMap(), [[32, 18], [51, 37], [70, 56], [89, 75]])
    endpoints_new, map_large = DQCCouplingMap(map_interm, heavySquareHeronCouplingMap(), [[165, 18], [184, 37], [203, 56], [222, 75]])
    all_endpoints = endpoints + endpoints_new
    backend_large = customBackend(name="FezDQC_",num_qubits=399, coupling_map=map_large, remote_gates=all_endpoints)
    backend_large.addNoiseDelayToRemoteGates(all_endpoints, error=noise)

    return backend_large

def generateBackends(backend_generator, noise: list[float] = [0.015, 0.03, 0.05]):
    for n in noise:
        backend = backend_generator(n)
        saveBackend(backend, "backends/" + backend.name + str(n))


def test(backend: str = "FezDQC"):
    #bs = loadBackend("guadalupeDQC")
    #bm = loadBackend("KyivDQC")
    backend = loadBackend(backend)

    qc = QuantumCircuit(5)
    qc.h(0)
    for i in range(1, 5):
        qc.cx(i - 1, i)

    qc.measure_all()

    qc_t = transpile(qc, backend)

    print(qc_t)

#generateBackends(constructDQCSmall)
#generateBackends(constructDQCMedium)
#generateBackends(constructDQCLarge)

#b = loadBackend("backends/GuadalupeDQC_0.015")
#b2 = loadBackend("backends/KyivDQC_0.015")
#b3 = loadBackend("backends/FezDQC_0.015")
#b.plotGateProbDistribution()
#b2.plotGateProbDistribution()
#b3.plotGateProbDistribution()
#print(b2.target)
#b2.plotGateProbDistribution()
#getRealNoiseModelsFromEagle()


