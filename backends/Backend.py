from qiskit.transpiler import Target, InstructionProperties, CouplingMap
from qiskit.circuit import QuantumCircuit
from qiskit import transpile
import qiskit.circuit.library as Gates
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.providers import QubitProperties
from qiskit_ibm_runtime import QiskitRuntimeService

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle


def defaultGateSet():
    return ["id", "sx", "x", "rz", "rzz", "cz", "rx"]

def defaultCouplingMap():
    return CouplingMap([[0, 1], [1, 4], [4, 7], [6, 7], [7, 10], [10, 12], [12, 15], [1, 2], [2, 3], [3, 5], [5, 8], [8, 11], [11, 14], [12, 13], [13, 14], [8,9]])


def getRealNoiseModelsFromEagle():
    service = QiskitRuntimeService(instance="ibm-q/open/main")
    backend = service.backend("ibm_kyiv")
    qubit_properties = [backend.qubit_properties(i) for i in range(backend.num_qubits)]

    return qubit_properties, backend.target

def heavyHexEagleCouplingMap():
    service = QiskitRuntimeService(instance="ibm-q/open/main")
    backend = service.backend("ibm_kyiv")

    return backend.coupling_map


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
    
    def __init__(self, name: str ="customDQCBackend", num_qubits: int = 16, coupling_map: CouplingMap = defaultCouplingMap(), basis_gates: list[str] = defaultGateSet(), noise_model: Target = None, qubit_properties: list[QubitProperties] = None):
        self.name = name
        assert num_qubits == coupling_map.size()
        super().__init__(num_qubits, basis_gates=basis_gates, coupling_map=coupling_map, control_flow=True, seed=1)

        if noise_model == None:
            self.addStateOfTheArtNoise()
        else:
            for instruction, properties_dict in noise_model.items():
                #print(instruction, properties_dict)
                if properties_dict is not None:
                    for p in properties_dict.items():
                        #print(p)
                        #exit()
                        self.target.update_instruction_properties(instruction, p[0], p[1])

        if qubit_properties == None:
            self.addStateOfTheArtQubits()
        else: 
            self.target.qubit_properties = qubit_properties

    def updateGateProps(self, duration: int = 70, error_med: float = 0.002, error_min: float = 0.0001, error_max: float = 0.06, gate: str = "cz"):
        gates = self.target[gate].items()
        target_ratio = (error_med - error_min) / (error_max - error_min)
        alpha = 0.5
        beta = alpha * (1 - target_ratio) / target_ratio
        beta = 2
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

def saveBackend(backend: customBackend, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(backend, f)

def loadBackend(filename: str):
    with open(filename, "rb") as f:
        backend = pickle.load(f)

    return backend

def constructDQCSmall(): 
    endpoints, map_small = DQCCouplingMap(defaultCouplingMap(), defaultCouplingMap(), [[13, 2], [15, 0]])
    backend_small = customBackend(num_qubits=32, coupling_map=map_small)
    backend_small.addNoiseDelayToRemoteGates(endpoints)

    return backend_small

def constructDQCMedium():
    endpoints, map_medium = DQCCouplingMap(heavyHexEagleCouplingMap(), heavyHexEagleCouplingMap(), [[32, 18], [51, 37], [70, 56], [89, 75]])
    props_medium, noise_model_medium = getRealNoiseModelsFromEagle()
    backend_medium = customBackend(num_qubits=254, coupling_map=map_medium, noise_model=noise_model_medium, qubit_properties=props_medium, basis_gates=["ecr", "id", "rz", "sx", "x"])
    backend_medium.addNoiseDelayToRemoteGates(endpoints, gates=["ecr"])

#bs = constructDQCSmall()
#bm = constructDQCMedium()

#saveBackend(bs, "guadalupeDQC")
#saveBackend(bm, "KyivDQC")

bs = loadBackend("guadalupeDQC")
bm = loadBackend("KyivDQC")

qc = QuantumCircuit(5)
qc.h(0)
for i in range(1, 5):
    qc.cx(i - 1, i)

qc.measure_all()

print(qc)

qc_s = transpile(qc, bs)
qc_m = transpile(qc, bm)

print(qc_s)
print(qc_m)


