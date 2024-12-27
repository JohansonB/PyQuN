from abc import ABC, abstractmethod
from typing import List


class Aggregator(ABC):
    #assume that each sublist has the same length
    @abstractmethod
    def _private_aggregate(self, vals: List[List[float]]) -> List[float]:
        pass

    def aggregate(self, vals: List[List[float]]) -> List[float]:
        if vals is None or len(vals) == 0:
            return None
        ze_len = len(vals[0])
        for li in vals:
            if len(li) != ze_len:
                raise Exception("All sublists are expected to have the same length")
        return self._private_aggregate(vals)