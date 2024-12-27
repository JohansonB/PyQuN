from typing import Set, Iterable

from RaQuN_Lab.datamodel.modelset.Element import Element


class Model:

    def __init__(self,ze_id: int, elements:Set['Element'] = None) -> None:
        if elements is None:
            elements = set()
        else:
            if not isinstance(elements, set):
                raise ValueError(f"Attributes must be initialized with a set. Got {type(elements)} instead.")

            if not all(isinstance(ele, Element) for ele in elements):
                raise ValueError("All elements of the set must be instances of Attribute.")

        self.elements = elements
        self.model_set = None
        self.id = ze_id

    def clone(self) -> 'Model':
        ele_set = set()
        for ele in self.elements:
            ele_set.add(ele.clone())
        model_clone = Model(self.id, ele_set)
        return model_clone

    def set_model_set(self, model_set : 'ModelSet') -> None:
        self.model_set = model_set

    def set_elements(self, eles : Iterable[Element]) -> None:
        if self.model_set is not None:
            for e in self.elements:
                self.model_set.notify_ele_del(e)
            for e in eles:
                self.model_set.notify_ele_add(e)
        self.elements = set(eles)




    def add_element(self, element:Element) -> None:
        self.elements.add(element)
        element.set_model_id(self.id)
        if self.model_set is not None:
            self.model_set.notify_ele_add(element)



    def remove_element(self, element:Element) -> None:
        self.elements.remove(element)
        if self.model_set is not None:
            self.model_set.notify_ele_del(element)

    def get_id(self) -> int:
        return self.id

    def get_elements(self) -> Set[Element]:
        return self.elements

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __eq__(self, other):
        return self.id == other.get_id()

    def __hash__(self):
        # Use `self.id` if it’s initialized; otherwise, fall back to `id(self)`
        return hash(getattr(self, 'id', id(self)))

    def __repr__(self):
        return f"Model(id={self.id}, num_elements={len(self.elements)})"
