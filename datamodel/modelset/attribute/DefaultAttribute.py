from abc import ABC
from typing import Any

from RaQuN_Lab.datamodel.modelset.attribute.Attribute import Attribute


class DefaultAttribute(Attribute, ABC):
    def __init__(self, value:Any = None) -> None:
        super().__init__(value)

    def dist(self, other):
        pass

    def __eq__(self, other):
        return self.value == other.value

    def __len__(self):
        return len(self.value)

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"{self.__class__.__name__}(value={self.value})"


    def clone(self) -> 'Attribute':
        return type(self)(self.value)

    def parse_string(self, encoding):
        self.value = encoding
