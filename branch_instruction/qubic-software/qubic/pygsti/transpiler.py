import pygsti
from typing import List, Dict, Tuple

pygsti_to_qubic = {
        'Gxpi2': 'X90',
        'Gypi2': 'Y90',
        'Gcr': 'CR',
        'Gcz': 'CZ',
        'Gcphase': 'CZ',
        'Gzpi2': 'Z90',
        'Gzr': 'virtual_z',
        'Gxx': ['X90', 'X90'],
        'Gxy': ['X90', 'Y90'],
        'Gyx': ['Y90', 'X90']}


def _parse_layer(layertup, qubit_map):
    layercirc = []
    if layertup.name == 'COMPOUND':
        for layer in layertup:
            layercirc.extend(_parse_layer(layer, qubit_map))
    else:
        if isinstance(pygsti_to_qubic[layertup.name], str):
            layercirc = [{'name': pygsti_to_qubic[layertup.name],
                          'qubit': [qubit_map[n] for n in layertup.qubits]}]
            if layertup.name == 'Gzr':
                layercirc[0]['phase'] = layertup.args[0]
        else:
            layercirc = []
            for i, gatename in enumerate(pygsti_to_qubic[layertup.name]):
                layercirc.append({'name': gatename,
                                  'qubit': [qubit_map[n] for n in layertup.qubits]})
    return layercirc


def transpile(pygsti_circuit: pygsti.circuits.Circuit, 
              qubit_map: List[str] | Dict[int, str], 
              gateware_twirl: bool = False,
              delay_before_circuit: float = 500.e-6) -> List[Dict]:
    """
    Transpile a pygsti circuit into a qubic program. 

    Parameters
    ----------
    pygsti_circuit: pygsti.circuits.Circuit
        circuit to transpile
    qubit_map: List[str] | Dict[int, str]
        Maps pygsti quantum registers to physical qubits. If list
        of physical qubits, register 0 is assumed to correspond to the
        first element, etc. If dict, register -> qubit mapping is given
        directly as key value pairs.
    gateware_twirl: bool = False
    delay_before_circuit: float = 500.e-6
        relaxation delay to insert at the beginning of each circuit

    Returns
    -------
    qubic_circuits: List[Dict]
        transpiled qubic circuit 
    """
    qubic_circuit = list()
    qubic_circuit.append({'name': 'delay', 't': delay_before_circuit})
    if gateware_twirl:
        qubits = [qubit_map[qid] for qid in pygsti_circuit.line_labels]
        qubic_circuit.append({'name': 'begin_rc', 'qubits': qubits})
    for layer in pygsti_circuit:
        qubic_circuit.extend(_parse_layer(layer, qubit_map))
        qubic_circuit.append({'name': 'barrier'})
    if gateware_twirl:
        qubits = [qubit_map[qid] for qid in pygsti_circuit.line_labels]
        qubic_circuit.append({'name': 'end_rc', 'qubits': qubits})
    for qid in pygsti_circuit.line_labels:
        qubic_circuit.append({'name': 'read', 'qubit': qubit_map[qid]})
    return qubic_circuit

