from abc import ABC, abstractmethod
from typing import Any


class Attribute(ABC):
    '''smallest component of the RaQuN data model
        Attributes with concrete datatypes extend this class
    '''

    def __init__(self, value:Any = None)-> None:
        self.value = value

    @abstractmethod
    def dist(self, other:'Attribute') -> float:
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    # takes a string representation of the attribute and parses it
    @abstractmethod
    def parse_string(self, encoding:str) -> None:
        pass

    @abstractmethod
    def clone(self) -> 'Attribute':
        pass
