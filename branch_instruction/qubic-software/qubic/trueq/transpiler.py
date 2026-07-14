import matplotlib.pyplot as plt
import trueq as tq
import numpy as np
from typing import List, Dict

def transpile(trueq_circuit, label_to_qubit, entangler='cz', gateware_twirl=False, tq_compile=True, delay_before_circuit=500.e-6):
    """
    Parameters
    ----------
    trueq_circuit: trueq.circuits.Circuit or CircuitCollection
    label_to_qubit: dict or list
        if dict, keys are trueq labels (ints) and values are qubitids ('Q0', 'Q1', etc)
        if list of qubitids, label 0 is assumed to be the first element, etc
    entangler: str
        either 'cz' or 'cnot'
    gateware_twirl: bool
    tq_compile: bool
        If True, run the TrueQ compiler to compile to native gates. Otherwise,
        the circuit is assumed to have already been compiled and is transpiled as is.
    delay_before_circuit: float
        delay (in seconds) for qubit relaxation

    Returns
    -------
    qubic_circuits: List[List | Dict]
        transpiled qubic circuit or list of qubic circuits (depending on whether trueq_circuit was a single Circuit or CircuitCollection)
    """
    if isinstance(trueq_circuit, tq.circuits.Circuit):
        return _transpile(trueq_circuit, label_to_qubit, entangler, gateware_twirl, tq_compile, delay_before_circuit)
    elif isinstance(trueq_circuit, tq.circuits.CircuitCollection):
        qubic_circuits = []
        for circuit in trueq_circuit:
            qubic_circuits.append(_transpile(circuit, label_to_qubit, entangler, gateware_twirl, tq_compile, delay_before_circuit))
        return qubic_circuits
    else:
        raise TypeError

def _transpile(trueq_circuit, label_to_qubit, entangler='cz', gateware_twirl=False, tq_compile=True, delay_before_circuit=500.e-6):
    """
    Parameters
    ----------
    trueq_circuit : trueq.circuits.Circuit
    label_to_qubit : dict or list
        if dict, keys are trueq labels (ints) and values are qubitids ('Q0', 'Q1', etc)
        if list of qubitids, label 0 is assumed to be the first element, etc
    entangler : str
        either 'cz' or 'cnot'
    gateware_twirl: bool
    tq_compile: bool
        If True, run the TrueQ compiler to compile to native gates. Otherwise,
        the circuit is assumed to have already been compiled and is transpiled as is.
    delay_before_circuit : float
        delay (in seconds) for qubit relaxation
    """
    if entangler.lower() == 'cz':
        entangler = tq.Gate.cz
    elif entangler.lower() == 'cnot':
        entangler = tq.Gate.cnot
    else:
        raise Exception('{} entangler not supported'.format(entangler))

    if tq_compile:
        # default passes from tq.compilation.Compiler.HARDWARE_PASSES, minus Merge
        passes = (tq.compilation.two_qubit.Native2Q,
                  tq.compilation.one_qubit.Native1Q,
                  tq.compilation.common.InvolvingRestrictions,
                  tq.compilation.common.RemoveEmptyCycle)

        # Need to do this to prevent TrueQ from merging single-qubit
        # gates between cycles, if using gateware RC
        if not gateware_twirl:
            passes = (passes[0], tq.compilation.Merge, *passes[1:])

        compiler = tq.Compiler.basic(entangler, mode='ZXZXZ', passes=passes)
        compiled_circuit = compiler.compile(trueq_circuit)
    else:
        compiled_circuit = trueq_circuit

    qubic_circuit = [{'name': 'delay', 't': delay_before_circuit}]
    if gateware_twirl:
        qubits = [label_to_qubit[label] for label in trueq_circuit.labels]
        qubic_circuit.append({'name': 'begin_rc', 'qubits': qubits})

    for cycle in compiled_circuit:
        qubic_circuit.append({'name': 'barrier'})
        measured_qubits = []
        for labels, operation in cycle:
            qubits = [label_to_qubit[l] for l in labels]
            if isinstance(operation, tq.operations.Meas):
                assert(len(labels) == 1)
                if gateware_twirl: # end RC cycle when performing measurement(s)
                    if len(measured_qubits) == 0:
                        qubic_circuit.append({'name': 'end_rc', 'qubits': measured_qubits})
                    measured_qubits.extend(qubits)
                qubic_circuit.append({'name': 'read', 'qubit': qubits})
            elif operation.name == 'sx':
                assert len(labels) == 1
                assert operation.parameters == {}
                qubic_circuit.append({'name': 'X90', 'qubit': qubits})
            elif operation.name == 'z':
                assert len(labels) == 1
                qubic_circuit.append({'name': 'virtual_z', 'qubit': qubits,
                                      'phase': np.deg2rad(operation.parameters['phi'])})
            elif operation.name == 'cz':
                qubic_circuit.append({'name': 'CZ', 'qubit': qubits})
            elif operation.name == 'cnot' or operation.name == 'cx':
                qubic_circuit.append({'name': 'CNOT', 'qubit': qubits})

            else:
                raise Exception('{} not supported'.format(operation))

    return qubic_circuit

