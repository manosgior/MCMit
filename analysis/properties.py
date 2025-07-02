from qiskit.circuit import QuantumCircuit

import networkx as nx
from networkx.classes.graph import Graph

from typing import Union, List

from analysis.dag import circuit_to_dependency_graph

def countGates(circuit: QuantumCircuit, gates: list[str]):
    pass

def countNonLocalGates(circuit: QuantumCircuit):
    return sum(1 for gate, qargs, _ in circuit.data if len(qargs) == 2)

def countRemoteGates(circuit: QuantumCircuit, remote_couplings: list[tuple]):
    pass

def getMeasurements(circuit: QuantumCircuit, include_final: bool = True) -> list:
    circ_copy = circuit.copy()
  
    if not include_final:
        circ_copy.remove_final_measurements()
   
    measurements = [(op, qargs) for op, qargs, _ in circ_copy.data if op.name == "measure"]    
    
    return measurements

def getFirstLayerOfMCMs(circuit_or_dag: Union[QuantumCircuit, nx.DiGraph]) -> list:
    dag = (circuit_to_dependency_graph(circuit_or_dag) 
        if isinstance(circuit_or_dag, QuantumCircuit) 
        else circuit_or_dag)
    
    # Find all measurement nodes
    measurement_nodes = []
    for node in dag.nodes():
        op = dag.nodes[node]['operation']
        if hasattr(op, 'name') and op.name == 'measure':
            measurement_nodes.append(node)
    
    if not measurement_nodes:
        return []
    
     # Find hotspot measurements (those with no measurement predecessors)
    hotspots = []
    for mnode in measurement_nodes:
        # Get all predecessor nodes
        predecessors = nx.ancestors(dag, mnode)
        # Check if any predecessor is a measurement
        has_measurement_predecessor = any(
            pred in measurement_nodes 
            for pred in predecessors
        )
        
        if not has_measurement_predecessor:
            hotspots.append((mnode, dag.nodes[mnode]['operation']))
    
    return hotspots



def getDepth(circuit: QuantumCircuit):
    pass

def getSize(circuit: QuantumCircuit):
    return circuit.num_qubits
    