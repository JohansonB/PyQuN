from abc import ABC, abstractmethod
from typing import List


class Statistic(ABC):
    @abstractmethod
    def evaluate(self, match_scores : List[float]) -> float:
        pass