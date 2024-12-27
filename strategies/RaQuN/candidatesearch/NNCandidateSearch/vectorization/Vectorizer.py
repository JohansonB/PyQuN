from abc import ABC, abstractmethod

import numpy as np


class Vectorizer(ABC):

    @abstractmethod
    def vectorize(self, element:'Element') -> np.ndarray:
        #takes as input an element and returns a vector representation of it
        pass

    @abstractmethod
    def dim(self) -> int:
        pass

    @abstractmethod
    def innit(self, m_s:'ModelSet') -> None:
        pass




