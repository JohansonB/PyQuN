from abc import ABC, abstractmethod
from typing import Iterable

from RaQuN_Lab.datamodel.matching.similarities.match.shouldmatch.ShouldMatchStrategy import ShouldMatchStrategy


class Similarity(ABC):
    def __init__(self, strategy:ShouldMatchStrategy) -> None:
        self.strategy = strategy

    @abstractmethod
    def innit(self, m_s : 'Modelset'):
        pass

    @abstractmethod
    def similarity(self, match:'Match'):
        pass
    @abstractmethod
    def should_match(self, matches:Iterable['Match'], elements:Iterable['Element']):
        pass
