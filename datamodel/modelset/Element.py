from typing import Set

from RaQuN_Lab.datamodel.modelset.attribute.Attribute import Attribute


class Element:

    def __init__(self, ze_id : int, attributes:Set['Attribute'] = None, name:str = None) -> None:
        if attributes is None:
            attributes = set()
        else:
            if not isinstance(attributes, set):
                raise ValueError(f"Attributes must be initialized with a set. Got {type(attributes)} instead.")

            if not all(isinstance(attr, Attribute) for attr in attributes):
                raise ValueError("All elements of the set must be instances of Attribute.")

        self.attributes = attributes
        self.ele_id = ze_id
        self.name = name
        self.model_id = None
        self.custom_id = None


    def set_custom_id(self, custom_id:int) -> None:
        self.custom_id = custom_id

    def get_custom_id(self) -> int:
        return self.custom_id

    def get_name(self) -> str:
        return self.name

    def add_attr(self, attribute:Attribute) -> None:
        self.attributes.add(attribute)

    def set_model_id(self,model_id:int) -> None:
        self.model_id = model_id

    def get_id(self) -> int:
        return self.ele_id

    def set_element_id(self, element_id:int) -> None:
        self.ele_id = element_id

    def get_model_id(self) -> int:
        return self.model_id

    def set_name(self, name : str):
        self.name = name

    def __iter__(self):
        return iter(self.attributes)

    def __len__(self):
        return len(self.attributes)

    def __eq__(self, other):
        return self.ele_id == other.get_id()

    def __repr__(self):
        attrs = ', '.join([repr(attr) for attr in self.attributes])
        return f"Element(id={self.ele_id}, name={self.name}, attrs=[{attrs}])"

    def __hash__(self):
        return hash(self.ele_id)

    #cloned elements retain the id of the original element
    def clone(self) -> 'Element':
        attr_set = set()
        for attr in self.attributes:
            attr_set.add(attr.clone())

        ele_clone = Element(self.ele_id, attr_set)
        ele_clone.name = self.name
        ele_clone.model_id = self.model_id
        ele_clone.custom_id = self.custom_id

        return ele_clone
