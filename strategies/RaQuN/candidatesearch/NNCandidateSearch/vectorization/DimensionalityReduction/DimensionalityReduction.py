from abc import ABC, abstractmethod

import numpy as np


class DimensionalityReduction(ABC):
    @abstractmethod
    def reduce(self, in_mat : np.ndarray) -> np.ndarray:
        pass
