import pygsti
from collections import OrderedDict

class SimulationManager:
    def __init__(self, pygsti_model, seed=None, simulation_type='multinomial', readout_register=None):
        self.model = pygsti_model
        self.seed = seed
        self.simulation_type = simulation_type
        self.readout_register = readout_register

    def collect_classified_shots(self, job_dict, num_shots_per_circuit,
                       reads_per_shot=1, delay_per_shot=500.e-6):
        """
        Mimics the qubic collect classified shots, simulates the outcomes and arranges into
        a dictionary of datastreams with keys that are the readout qubits

        must unpack the pygsti dataset and convert it into qubic form,
        there is freedom in the ordering of the data, I'll just stack bitstrings in an arbitrary order
        """

        if self.simulation_type == 'multinomial':
            assert type(job_dict) is OrderedDict
            ds = pygsti.data.simulate_data(self.model, list(job_dict.keys()), num_shots_per_circuit,
                                       seed=self.seed)
            num_outcomes = len(ds.outcome_labels)
            output_stream = {
                qid: np.zeros((len(job_dict), num_shots_per_circuit, 1)) for qid in self.readout_register
            }
            for id_circ, circ in enumerate(job_dict.keys()):
                counts = ds[circ].counts
                iterator_counts = 0  # used to iterate over all the counts
                for bitstring in counts.keys():
                    for id_bit, bit in enumerate(bitstring[0]):
                        if bit == '1':
                            assert (counts[bitstring] - int(counts[bitstring])) < 1e-12
                            for i in range(iterator_counts, iterator_counts + int(counts[bitstring])):
                                output_stream[self.readout_register[id_bit]][id_circ, i, 0] = 1
                    iterator_counts += int(counts[bitstring])
            return output_stream



        elif self.simulation_type == 'scaled_probability':
            # TODO: implement scaled probability simulation type
            pass

    def collect_dataset(self, job_dict, num_shots_per_circuit, qchip):
        """
                Mimics the qubic collect classified shots, simulates the outcomes and arranges into
                a dictionary of datastreams
                """
        if self.simulation_type == 'multinomial':
            ds = pygsti.data.simulate_data(self.model, list(job_dict.keys()), num_shots_per_circuit,
                                           seed=self.seed)
            return ds
        elif self.simulation_type == 'scaled_probability':
            pass
        return -1
