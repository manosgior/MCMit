from chemistry import HamiltonianSimulationBenchmark, VQEBenchmark
from error_correction import BitCodeBenchmark, PhaseCodeBenchmark
from optimization import VanillaQAOABenchmark, FermionicSwapQAOABenchmark
from quantum_information import GHZBenchmark, MerminBellBenchmark

import sys
from qiskit import QuantumCircuit

lower_limit = int(sys.argv[1])
upper_limit = int(sys.argv[2])
step = int(sys.argv[3])

for i in range(lower_limit, upper_limit, step):
    circ = GHZBenchmark(i).circuit()
    circ.qasm(formatted=True, filename="ghz/n" + str(i) + ".qasm")