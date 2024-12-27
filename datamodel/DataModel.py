from abc import ABC, abstractmethod
from typing import Union

from RaQuN_Lab.datamodel.modelset.Element import Element
from RaQuN_Lab.datamodel.modelset.Model import Model


class DataModel(ABC):
    @abstractmethod
    def get_by_id(self, id: int) -> Union[Model, Element]:
        pass






