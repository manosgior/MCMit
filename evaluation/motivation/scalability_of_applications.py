from applications.constant_depth_GHZ import *
from applications.quantum_teleportation import *
from applications.long_range_CNOT import *

from backends.simulator import *
from backends.backend import *

from analysis.fidelity import *
from analysis.distribution_processing import *

teleportation_circuits = generate_long_range_cnots(10)

simulator = getNoiselessSimulator()

for tc in teleportation_circuits:
    print(tc)
    #results = simulator.run(tc, shots=10000).result()
    #counts = results.get_counts()
    #clean_counts = process_distribution_teleportation(counts)
    #print(counts, clean_counts)

    #expectation_value = calculateExpectationValue(clean_counts, 10000, "sum")

    #print(expectation_value)
