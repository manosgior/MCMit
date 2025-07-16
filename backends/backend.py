from qiskit.transpiler import Target, InstructionProperties, CouplingMap
from qiskit.circuit import QuantumCircuit
from qiskit import transpile
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

def cliffordGateSet():
    return ["h", "cx", "s"]

def defaultResolutionTime():
    return 2.2222222222222221e-10 * 1e9

def defaultConfusionMatrix():
    p0 = 0.00391
    p1 = 0.01221

    return  np.array([
        1-p0, p1,
        p0, 1-p1
    ]).reshape(2,2)

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

def getRealEagleBackend():
    service = QiskitRuntimeService()
    backend = service.backend("ibm_brisbane")

    return backend

def heavyHexEagleCouplingMap():
    service = QiskitRuntimeService()
    backend = service.backend("ibm_brisbane")
    #backend = service.backend("ibm_sherbrooke")

    return backend.coupling_map

def heavySquareHeronCouplingMap():
    service = QiskitRuntimeService()
    backend = service.backend("ibm_brisbane")

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
    plt.savefig("backends/visualization/coupling_map.png", dpi=300, bbox_inches="tight")

class customBackend(GenericBackendV2):    
    
    def __init__(self, name: str ="customDQCBackend", num_qubits: int = 16, coupling_map: CouplingMap = GuadalupeCouplingMap(), dt: float = defaultResolutionTime(),basis_gates: list[str] = defaultGateSet(), noise_model: Target = None, qubit_properties: list[QubitProperties] = None, remote_gates: list[tuple] = []):
        assert num_qubits == coupling_map.size()

        super().__init__(num_qubits, basis_gates=basis_gates, coupling_map=coupling_map, control_flow=True, dtm=dt, seed=1)
        self.name = name
        self.remote_gates = remote_gates
        self.confusion_matrix = defaultConfusionMatrix()

        if noise_model == None:
            self.addStateOfTheArtNoise()
        else:
            for instruction, properties_dict in noise_model.items():
                if properties_dict is not None:
                    for p in properties_dict.items():
                        self.target.update_instruction_properties(instruction, p[0], p[1])

        if qubit_properties == None:
            #self.addStateOfTheArtQubits()
            self.sampleFromRealEagleQubits()
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
            duration_rand = np.random.randint(duration - 10, duration + 10) * 1e-9
            duration_rand = round(duration_rand / self.dt) * self.dt
           
            error_rand = np.random.beta(alpha, beta)
            error_rand = error_min + (error_max - error_min) * error_rand
            values.append(error_rand)

            self.target.update_instruction_properties(gate, g[0], InstructionProperties(duration_rand, error_rand))

        #print("--------------------")
        #print(np.median(values))

    def addStateOfTheArtNoise(self):
        gate_set = self._basis_gates

        for g in gate_set:
            if g in ["id", "sx", "x", "rz", "rx", "h"]:
                self.updateGateProps(gate=g, duration=50, error_med=0.00025, error_min=0.0001, error_max=0.015)
            elif g in ["cz", "rzz", "cx"]:
                self.updateGateProps(gate=g, duration=70, error_med=0.002, error_min=0.0009, error_max=0.06)
            elif g == "measure":
                self.updateGateProps(gate=g, duration=70, error_med=0.01, error_min=0.002, error_max=0.5)

    def addNoiseDelayToRemoteGates(self, endpoints: list[tuple], duration: int = 300, error: float = 0.03, gates: list[str] = ["cz", "rzz", "cx"]):
        roundedDuration = round((duration * 1e-9) / self.dt) * self.dt
        gate_props = InstructionProperties(roundedDuration, error)
        basis_gates = self._basis_gates
        
        for g in gates:
            if g in basis_gates:
                for e in endpoints:
                    self.target.update_instruction_properties(g, e, gate_props)

    def addStateOfTheArtQubits(self):
        qubit_props = []
        
        for i in range(self.num_qubits):
            t1 = np.random.normal(190, 120, 1)
            t1 = np.clip(t1, 50, 500)
            t1 = t1 * 1e-6

            t2 = np.random.normal(130, 120, 1)
            t2 = np.clip(t2, 50, 650)
            t2 = t2 * 1e-6

            qubit_props.append(QubitProperties(t1=t1, t2=t2, frequency=5.0e9))

        self.target.qubit_properties = qubit_props

    def sampleFromRealEagleQubits(self):
        qubit_props, noise_model = getRealNoiseModelsFromEagle()
        own_qubit_props = []
        if self.num_qubits <= len(qubit_props):
            for i in range(self.num_qubits):
                own_qubit_props.append(qubit_props[i])
        else:
            for i in range(self.num_qubits):
                own_qubit_props.append(qubit_props[np.random.randint(0, len(qubit_props))])

        self.target.qubit_properties = own_qubit_props

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
        plt.savefig("backends/visualization/" + self.name + "_gate_probs.png")

def saveBackend(backend: customBackend, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(backend, f)

def loadBackend(filename: str):
    with open(filename, "rb") as f:
        backend = pickle.load(f)

    return backend

def constructBackendSmall():
    coupling_map = GuadalupeCouplingMap()
    backend_small = customBackend(name="GuadalupeDQC_", num_qubits=16, coupling_map=coupling_map)

    return backend_small

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
        saveBackend(backend, "backends/QPUs/" + backend.name + str(n))

def connected_subgraphs_of_size(G, k):
    seen = set()

    def extend(subset, frontier):
        if len(subset) == k:
            key = tuple(sorted(subset))
            if key not in seen:
                seen.add(key)
                yield G.subgraph(key)
            return

        # only consider frontier nodes greater than the minimum of subset
        min_seed = min(subset)
        for v in sorted(frontier):
            if v <= min_seed:
                continue
            new_subset = subset | {v}
            # expand frontier to include neighbors of v, then remove any in subset
            new_frontier = (frontier | set(G.neighbors(v))) - new_subset
            yield from extend(new_subset, new_frontier)

    for start in G.nodes():
        # initialize with single-node subset and its neighbors as frontier
        yield from extend({start}, set(G.neighbors(start)))

def find_optimal_qubit_set(backend, n_qubits: int, metric_func: callable, property: str = "", minimize: bool = True) -> tuple[list[int], float]:
    """
    Finds a set of n connected qubits that optimize (minimize/maximize) a given metric.
    
    Args:
        backend: Quantum backend
        n_qubits: Number of qubits to find
        metric_func: Function that takes a list of qubit indices and returns a float
        minimize: If True, minimize the metric; if False, maximize
        
    Returns:
        tuple[list[int], float]: Optimal qubit indices and corresponding metric value
    """
    coupling_map = backend.coupling_map
    G = nx.Graph()
    G.add_edges_from(coupling_map.get_edges())

    subgraphs = connected_subgraphs_of_size(G, n_qubits)
    
    best_score = float('inf') if minimize else float('-inf')
    best_qubits = None
    
    # Find all connected subgraphs of size n_qubits
    for subgraph in subgraphs:
        #qubits = subgraph.nodes()  # Convert set to list
        score = metric_func(backend, subgraph, property)
        
        if (minimize and score < best_score) or \
           (not minimize and score > best_score):
            best_score = score
            best_qubits = subgraph.nodes()
    
    return best_qubits, best_score

def get_average_property(backend, subgraph, property_name: str) -> float:
    qubits = subgraph.nodes()
    total_property = 0
    for q in qubits:        
        total_property += backend.properties().qubit_property(q).get(property_name, (None,))[0]
    return total_property / len(qubits)

def average_two_qubit_error(backend, subgraph, property_name: str) -> float:
    """Returns average two-qubit gate error for given qubits"""
    total_error = 0
    count = 0
    edges = subgraph.edges()
    
    # Check all pairs of qubits
    for e in edges:
        q1, q2 = e[0], e[1]
        for gate in ['cz', 'ecr']:
            if gate in backend.target:
                props = backend.target[gate].get((q1, q2))
                if props is None:
                    props = backend.target[gate].get((q2, q1))
                if props and props.error is not None:
                    total_error += props.error
                    count += 1
    
    return total_error / count if count > 0 else float('inf')


