from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Sequence, Union, Dict, Optional, Any


from qiskit import QuantumCircuit

from _utils import ProbDistribution


class Benchmark(ABC):
    @abstractmethod
    def circuit(self) -> Union[QuantumCircuit, Sequence[QuantumCircuit]]:
        """Returns the quantum circuit corresponding to the current benchmark parameters."""

    @abstractmethod
    def score(self, counts: Union[Counter, List[Counter]]) -> float:
        """Returns a normalized [0,1] score reflecting device performance."""


class Device(ABC):
    @abstractmethod
    def run(self, circuits: List[QuantumCircuit], shots: int) -> List[ProbDistribution]:
        pass


