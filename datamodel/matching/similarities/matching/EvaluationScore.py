from abc import ABC, abstractmethod
from typing import Union

from RaQuN_Lab.experiment.ExperimentResult import ExperimentResult


class EvaluationScore(ABC):
    @abstractmethod
    def _pevaluate(self, matching:'Matching') -> float:
        pass

    def evaluate(self, matching : Union['Matching', 'ExperimentResult']) -> float:
        if isinstance(matching, ExperimentResult):
            matching = matching.match
        return self._pevaluate(matching)
