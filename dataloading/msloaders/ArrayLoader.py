#takes as input an array representation of a modelset
#each row of the matrix represents an element.
#the entries of the matrix are the elements attributes
#the second list contains the size of each model
from typing import Type, List

import numpy as np

from RaQuN_Lab.dataloading.msloaders.MSLoader import MSLoader
from RaQuN_Lab.datamodel.modelset.attribute.StringAttribute import StringAttribute


class ArrayLoader(MSLoader):
    def __init__(self, attribute_class: Type = StringAttribute, data: np.ndarray = None, separations: List[int] = None, element_names: List[str] = None):
        self.element_names = element_names
        self.rows = data
        self.separations = separations
        self.attr_index = None
        self.ele_index = None
        if separations is not None:
            self.next_stop = separations[0]
        else:
            self.next_stop = None
        self.stop_count = 1
        super().__init__(attribute_class)

    def set_data(self, data: np.array) -> None:
        self.rows = data

    def set_separations(self, sep :List[int]) -> None:
        self.separations = sep

    def set_element_names(self, names: List[str]) -> None:
        self.element_names = names

    def next_attribute(self):
        if self.attr_index is None:
            self.attr_index = 0
        else:
            self.attr_index += 1

        return self.rows[self.ele_index][self.attr_index]

    def next_element(self):
        if self.ele_index is None:
            self.ele_index = 0
        else:
            self.ele_index += 1
            self.attr_index = None

    def next_model(self):
        return

    def read_meta_data(self):
        return None

    def last_attribute(self):
        return len(self.rows[self.ele_index]) - 1 == self.attr_index

    def last_element(self):
        if self.next_stop is None:
            self.next_stop = self.separations[0]
        if self.ele_index is None:
            return False
        if len(self.rows) - 1 == self.ele_index:
            return True
        if self.ele_index + 1 == self.next_stop:
            self.next_stop += self.separations[self.stop_count]
            self.stop_count += 1
            return True
        return False

    def parse_ele_name(self):
        if self.element_names is None:
            return None
        return self.element_names[self.ele_index]


    def last_model(self):
        if self.ele_index is None:
            return False
        return len(self.rows)-self.separations[-1] <= self.ele_index

