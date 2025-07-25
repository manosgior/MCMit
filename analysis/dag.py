from typing import Iterator
import itertools
from collections import Counter
import matplotlib.pyplot as plt

import networkx as nx
from networkx.classes.graph import Graph

from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    CircuitInstruction,
    Instruction,
    Qubit,
    Barrier,
    Measure,
)

name_map = {
    "cx": "cx",
    "h": "h",
    "rz": "rz",
    "x": "x",
    "y": "y",
    "z": "z",
    "id": "id",
    "measure": "M",
    "reset": "R",
    "if_else": "if",
}

class DAG(nx.DiGraph):
    def __init__(self, circuit: QuantumCircuit):
        circuit = circuit.copy()

        def _next_op_on_qubit(qubit: int, from_idx: int) -> int:
            for i, instr in enumerate(circuit[from_idx + 1 :]):
                if qubit in instr.qubits:
                    return i + from_idx + 1
            return -1

        super().__init__()

        for i, instr in enumerate(circuit):
            self.add_node(i, instr=instr)

            for qubit in instr.qubits:
                next_op = _next_op_on_qubit(qubit, i)
                if next_op > -1:
                    self.add_edge(i, next_op)

        self._qregs = circuit.qregs
        self._cregs = circuit.cregs

    def copy(self) -> "DAG":
        return DAG(self.to_circuit())

    @property
    def qubits(self) -> list[Qubit]:
        return list(itertools.chain(*self._qregs))

    @property
    def clbits(self) -> list[Qubit]:
        return list(itertools.chain(*self._cregs))

    @property
    def depth(self) -> int:
        return nx.dag_longest_path_length(self)

    def add_qreg(self, qreg: QuantumRegister) -> None:
        if qreg in self._qregs:
            raise ValueError(f"Quantum register {qreg} already exists")
        self._qregs.append(qreg)

    def add_creg(self, creg: ClassicalRegister) -> None:
        if creg in self._cregs:
            raise ValueError(f"Classical register {creg} already exists")
        self._cregs.append(creg)

    def remove_1q_gates(self) -> None:
        nodes = list(self.nodes)
        nodes_to_remove = []
        for node in nodes:
            instr = self.get_node_instr(node)
            if len(instr.qubits) == 1:
                pred = next(self.predecessors(node), None)
                succ = next(self.successors(node), None)
                if pred is not None and succ is not None:
                    self.add_edge(pred, succ)
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            self.remove_node(node)

    def remove_cregs(self) -> None:
        self._cregs = []

    def to_circuit(self) -> QuantumCircuit:
        order = list(nx.topological_sort(self))
        circuit = QuantumCircuit(*self._qregs, *self._cregs)

        for i in order:
            instr = self.nodes[i]["instr"]
            circuit.append(instr)

        return circuit

    def qubit_dependencies(self) -> dict[Qubit, set[Qubit]]:
        depends_on: dict[Qubit, set[Qubit]] = {qubit: set() for qubit in self.qubits}
        for node in nx.topological_sort(self):
            instr = self.get_node_instr(node)
            qubits = instr.qubits
            if len(qubits) == 1 or isinstance(instr.operation, Barrier):
                continue
            elif len(qubits) == 2:
                q1, q2 = qubits

                add1 = depends_on[q2].copy()
                add1.add(q2)
                add2 = depends_on[q1].copy()
                add2.add(q1)

                depends_on[q1].update(add1)
                depends_on[q2].update(add2)
            else:
                raise ValueError("More than 2 qubits in instruction")
        for qubit in self.qubits:
            depends_on[qubit].discard(qubit)
        return depends_on

    def num_dependencies(self) -> int:
        return sum(len(deps) for deps in self.qubit_dependencies().values())

    def add_instr_node(self, instr: CircuitInstruction) -> int:
        new_id = max(self.nodes) + 1 if len(self.nodes) > 0 else 0
        self.add_node(new_id, instr=instr)
        return new_id


    def get_node_instr(self, node: int) -> CircuitInstruction:
        return self.nodes[node]["instr"]

    def qubits_of_edge(self, u: int, v: int) -> set[Qubit]:
        qubits1 = self.get_node_instr(u).qubits
        qubits2 = self.get_node_instr(v).qubits
        return set(qubits1) & set(qubits2)

    def remove_nodes_of_type(self, instr_type: type[Instruction]) -> None:
        # TODO Might not be correct actually
        nodes_to_remove = []
        for node in self.nodes:
            if isinstance(self.get_node_instr(node).operation, instr_type):
                predecessors = list(self.predecessors(node))
                successors = list(self.successors(node))

                for pred, succ in itertools.product(predecessors, successors):
                    pred_qubits = set(self.get_node_instr(pred).qubits)
                    succ_qubits = set(self.get_node_instr(succ).qubits)
                    if pred_qubits & succ_qubits:
                        self.add_edge(pred, succ)

                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            self.remove_node(node)

    def replace_node(self, node_before: int, node_after: int) -> None:
        prev = list(self.predecessors(node_before))
        for p in prev:
            self.remove_edge(p, node_before)
            self.add_edge(p, node_after)

        nxt = list(self.successors(node_before))
        for n in nxt:
            self.remove_edge(node_before, n)
            self.add_edge(node_after, n)

    def compact(self) -> None:
        # get the used qubits
        used_qubits: set[Qubit] = set()
        for node in self.nodes:
            used_qubits.update(self.get_node_instr(node).qubits)

        new_qreg = QuantumRegister(len(used_qubits), "q")
        qubit_mapping: dict[Qubit, Qubit] = {
            qubit: new_qreg[i] for i, qubit in enumerate(used_qubits)
        }
        # update the circuit
        for node in self.nodes:
            instr = self.get_node_instr(node)
            self.nodes[node]["instr"] = instr.replace(qubits=[qubit_mapping[qubit] for qubit in instr.qubits])

        self._qregs = [new_qreg]

    def instructions_on_qubit(self, qubit: Qubit) -> Iterator[CircuitInstruction]:
        for node in nx.topological_sort(self):
            instr = self.get_node_instr(node)
            if qubit in instr.qubits:
                yield instr

    def nodes_on_qubit(self, qubit: Qubit) -> Iterator[int]:
        for node in nx.topological_sort(self):
            instr = self.get_node_instr(node)
            if qubit in instr.qubits:
                yield node

    def fragment(self) -> None:
        con_qubits = list(nx.connected_components(dag_to_qcg(self)))
        new_frags = [
            QuantumRegister(len(qubits), name=f"frag{i}")
            for i, qubits in enumerate(con_qubits)
        ]
        qubit_map: dict[Qubit, Qubit] = {}  # old -> new Qubit
        for nodes, circ in zip(con_qubits, new_frags):
            node_l = list(nodes)
            for i in range(len(node_l)):
                qubit_map[node_l[i]] = circ[i]

        for node in self.nodes:
            instr = self.get_node_instr(node)
            instr.qubits = [qubit_map[qubit] for qubit in instr.qubits]
        self._qregs = new_frags


def dag_to_qcg(dag: DAG, use_qubit_idx: bool = False) -> nx.Graph:
    graph = nx.Graph()
    bb = nx.edge_betweenness_centrality(graph, normalized=False)
    nx.set_edge_attributes(graph, bb, "weight")
    if use_qubit_idx:
        graph.add_nodes_from(range(len(dag.qubits)))
    else:
        graph.add_nodes_from(dag.qubits)

    for node in dag.nodes:
        cinstr = dag.get_node_instr(node)
        op, qubits = cinstr.operation, cinstr.qubits
        if isinstance(op, Barrier):
            continue
        if len(qubits) >= 2:
            for qubit1, qubit2 in itertools.combinations(qubits, 2):
                if use_qubit_idx:
                    qubit1, qubit2 = dag.qubits.index(qubit1), dag.qubits.index(qubit2)

                if not graph.has_edge(qubit1, qubit2):
                    graph.add_edge(qubit1, qubit2, weight=0)
                graph[qubit1][qubit2]["weight"] += 1
    return graph


def get_qubit_dependencies(dag: DAG) -> dict[Qubit, Counter[Qubit]]:
    """Converts a graph into a qubit-dependency relations

    qubit -> {qubits it depends on}
    """

    qubit_depends_on: dict[Qubit, Counter[Qubit]] = {
        qubit: Counter() for qubit in dag.qubits
    }

    for node in nx.topological_sort(dag):
        instr = dag.get_node_instr(node)
        qubits = instr.qubits

        if len(qubits) == 1 or isinstance(instr.operation, Barrier):
            continue
        elif len(qubits) == 2:
            q1, q2 = qubits

            to_add1 = Counter(qubit_depends_on[q2].keys()) + Counter([q2])
            to_add2 = Counter(qubit_depends_on[q1].keys()) + Counter([q1])

            qubit_depends_on[q1] += to_add1
            qubit_depends_on[q2] += to_add2

        elif len(qubits) > 2:
            raise ValueError("Cannot convert dag to qdg, too many qubits")

    for qubit in dag.qubits:
        qubit_depends_on[qubit].pop(qubit, None)

    return qubit_depends_on

def circuit_to_dependency_graph(circuit: QuantumCircuit) -> nx.DiGraph:
    """
    Converts a quantum circuit to an operation dependency graph.
    Nodes are operations (gates/measurements) and edges represent qubit dependencies.
    
    Args:
        circuit: Input quantum circuit
        
    Returns:
        nx.DiGraph: Directed graph where nodes are operations and edges represent dependencies
    """
    # Create a directed graph
    op_dag = nx.DiGraph()
    
    # Keep track of last operation on each qubit
    last_op = {qubit: None for qubit in circuit.qubits}
    
    # Add nodes and edges for each instruction
    for idx, instr in enumerate(circuit):
        op, qubits = instr.operation, instr.qubits
        
        # Skip barriers
        if isinstance(op, Barrier):
            continue
            
        # Add node for current operation
        op_dag.add_node(idx, operation=op, qubits=qubits)
        
        # Add edges from last operations on these qubits
        for qubit in qubits:
            if last_op[qubit] is not None:
                op_dag.add_edge(
                    last_op[qubit], 
                    idx, 
                    qubit=qubit
                )
            # Update last operation for this qubit
            last_op[qubit] = idx
            
    return op_dag

def print_dependency_graph(dag: nx.DiGraph, filename: str, figsize: tuple = (10, 8)):
    """
    Plots the operation dependency graph and saves it to a file.
    
    Args:
        dag: Operation dependency graph
        filename: Path to save the plot
        figsize: Figure size as (width, height) tuple
    """
    # Create figure
    plt.figure(figsize=figsize)

    # Create node labels
    node_labels = {}
    for node in dag.nodes():
        op = dag.nodes[node]['operation']
        qubits = dag.nodes[node]['qubits']
        # Create readable label
        if hasattr(op, 'name'):
            label = f"{name_map[op.name]}\n{[q.index for q in qubits]}"
        else:
            label = f"{op}\n{[q.index for q in qubits]}"
        node_labels[node] = label

        # Set up layout
    pos = nx.spring_layout(dag)

    # Draw graph elements
    nx.draw_networkx_nodes(dag, pos, node_color='lightblue', 
                        node_size=700, alpha=0.6)
    nx.draw_networkx_labels(dag, pos, node_labels, font_size=8)

    # Draw edges with qubit labels
    edge_labels = nx.get_edge_attributes(dag, 'qubit')
    edge_labels = {k: f"q[{v.index}]" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(dag, pos, edge_labels, font_size=8)
    nx.draw_networkx_edges(dag, pos, edge_color='gray', 
                        arrowsize=10, alpha=0.5)

    plt.title("Operation Dependency Graph")
    plt.axis('off')

    # Save plot
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()