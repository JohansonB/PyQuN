from abc import ABC, abstractmethod
from typing import Iterable


class ShouldMatchStrategy(ABC):
    @abstractmethod
    def should_match(self, similarity:'Similarity', matches:Iterable['Match'], elements:Iterable['Element']) -> bool:
        pass

