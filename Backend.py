from abc import ABC, abstractmethod
from qiskit.providers.backend import BackendV2
from qiskit.transpiler import Target, InstructionProperties, CouplingMap
import qiskit.circuit.library as Gates
from qiskit.providers.fake_provider import GenericBackendV2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def defaultGateSet():
    return ["id", "sx", "x", "rz", "rzz", "cz", "rx"]

def defaultCouplingMap():
    return CouplingMap([[0, 1], [1, 4], [4, 7], [6, 7], [7, 10], [10, 12], [12, 15], [1, 2], [2, 3], [3, 5], [5, 8], [8, 11], [11, 14], [12, 13], [13, 14], [8,9]])


def DQCCouplingMap(coupling_map1: CouplingMap, coupling_map2: CouplingMap, endpoints: list):
    edges_1 = coupling_map1.get_edges()
    last_qubit = coupling_map1.size()
    converted_edges_1 = [[i, j] for i,j in edges_1]
    new_endpoints = [[i, last_qubit + j] for i,j in endpoints]
    new_coupling_map2 = [[last_qubit + i, last_qubit + j] for i,j in coupling_map2]
    
    new_coupling_map = CouplingMap(converted_edges_1 + new_endpoints + new_coupling_map2)
    new_endpoints_tuple = [(i, j) for i,j in new_endpoints]

    return new_endpoints_tuple, new_coupling_map

def printCouplingMap(coupling_map: CouplingMap):
    edges = coupling_map.get_edges()
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)

    nx.draw(G, with_labels=True, node_color="skyblue", edge_color="gray", pos=pos)
    plt.savefig("coupling_map.png", dpi=300, bbox_inches="tight")


class customBackend(GenericBackendV2):
    
    def __init__(self, name: str ="customDQCBackend", num_qubits: int = 16, coupling_map: CouplingMap = defaultCouplingMap(), basis_gates: list[str] = defaultGateSet()):
        self.name = name
        assert num_qubits == coupling_map.size()
        super().__init__(num_qubits, basis_gates=basis_gates, coupling_map=coupling_map, control_flow=True, seed=1)
        self.addStateOfTheArtNoise()

    def updateGateProps(self, duration: int = 70, error_med: float = 0.002, error_min: float = 0.0001, error_max: float = 0.06, gate: str = "cz"):
        gates = self.target[gate].items()

        for g in gates:
            duration_rand = np.random.randint(duration - 10, duration + 10)
            lambda_ = np.log(2) / error_med            
            error_rand = np.random.exponential(scale=1/lambda_)
            error_rand = np.clip(error_rand, error_min, error_max)
            self.target.update_instruction_properties(gate, g[0], InstructionProperties(duration_rand, error_rand))

    def addStateOfTheArtNoise(self): 

        for g in ["id", "sx", "x", "rz", "rx"]:
            self.updateGateProps(gate=g, duration=50, error_med=0.00025, error_min=0.0001, error_max=0.015)

        for g in ["cz", "rzz"]:
            self.updateGateProps(gate=g, duration=70, error_med=0.002, error_min=0.0009, error_max=0.06)

    def addNoiseDelayToGates(self, endpoints: list[tuple], duration: int = 300, error: float = 0.03, gate: str = "cx"):
        gate_props = InstructionProperties(duration, error)
        
        for e in endpoints:
            self.target.update_instruction_properties(gate, e, gate_props)

        

m1 = defaultCouplingMap()
m2 = defaultCouplingMap()
endpoints, m3 = DQCCouplingMap(m1, m2, [[13, 2], [15, 0]])
printCouplingMap(m3)
backend = customBackend(num_qubits=32, coupling_map=m3)
backend.addNoiseDelayToGates(endpoints, gate="cz")
print(backend.target["cz"])