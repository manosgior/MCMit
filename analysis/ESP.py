from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info import hellinger_fidelity
from qiskit.converters import circuit_to_dag
from qiskit.providers.fake_provider import GenericBackendV2

import numpy as np
from collections import defaultdict

def idleError(time: float, t1: float, t2: float):
    t2 = min(t1, t2)
    rate1 = 1/t1
    rate2 = 1/t2
    p_reset = 1-np.exp(-time*rate1)
    p_z = (1-p_reset)*(1-np.exp(-time*(rate2-rate1)))/2

    return p_z + p_reset

def calculateExpectedSuccessProbability(circuit: QuantumCircuit, backend: GenericBackendV2, onlyIdling: bool = False):
    dag = circuit_to_dag(circuit)
    fidelity = 1
    #decoherence_fidelity = 1
    dt = backend.configuration().dt
    touched = set()
    active_times = defaultdict()
    delays = defaultdict()

    for gate in dag.gate_nodes():
        if gate.name in ["ecr", "cx", "cz", "rzz"]:
            q0, q1 = gate.qargs[0]._index, gate.qargs[1]._index
            fidelity *= (1 - backend.target[gate.name][(q0, q1)].error)
            active_times[q0] +=  backend.target[gate.name][(q0, q1)].duration
            active_times[q1] +=  backend.target[gate.name][(q0, q1)].duration
            touched.add(q0)
            touched.add(q1)
        elif gate.name == 'delay':
            q0 = gate.qargs[0]._index
            if q0 in touched:
                #qp = backend.qubit_properties(q0._index)
                time = backend.target[gate.name][(q0)].duration * dt
                delays[q0] += time
                #fid *= 1-idleError(time, qp.t1, qp.t1)
        else:
            q = gate.qargs[0]._index
            fidelity *= (1 - backend.target[gate.name][(q)].error)
            touched.add(q)
    if onlyIdling:
        for qubit, time in delays.items():
            qp = backend.qubit_properties(qubit)
            fid *= 1-idleError(time, qp.t1, qp.t1)
    else:
        total_times = defaultdict()
        for qubit, time in active_times.items():
            total_times[qubit] = time + delays[qubit]

        for qubit, time in total_times.items():
            qp = backend.qubit_properties(qubit)
            fid *= 1-idleError(time, qp.t1, qp.t1)

    #for wire in dag.wires:
       # duration = 0.0
       # for gate in dag.nodes_on_wire(wire, only_ops=True):
           # if gate.name == "barrier":
               # continue
           # elif gate.name in ["ecr", "cx", "cz", "rzz"]:
               # q1, q2 = gate.qargs[0]._index, gate.qargs[1]._index
              #  q = gate.qargs[0]._index
              #  duration += backend.target[gate.name][(q1, q2)].duration
           # else:
              #  q = gate.qargs[0]._index
              #  duration += backend.target[gate.name][(q,)].duration
       # if duration > 0:
           # qp = backend.qubit_properties(wire._index)
           # t1 = np.exp(-duration / qp.t1)
           # t2 = np.exp(-duration / qp.t2)
           # decoherence_fidelity *= t1 * t2

    return fidelity


def calculateFidelity(circuit: QuantumCircuit, backend: GenericBackendV2, nshots: int = 8192, onlyIdling: bool = False):
    return  calculateExpectedSuccessProbability(circuit, backend, onlyIdling) * nshots

def getEstimateShots(circuit: QuantumCircuit, backend: GenericBackendV2, desired_fidelity: float, onlyIdling: bool = False):
    fid = calculateFidelity(circuit=circuit, backend=backend, onlyIdling=onlyIdling)

    return np.log(1 - desired_fidelity) / np.log(1 - fid)