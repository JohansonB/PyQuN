from abc import ABC, abstractmethod
from typing import Type

from RaQuN_Lab.dataloading.DataLoading import DataLoader
from RaQuN_Lab.datamodel.DataSet import DataSet
from RaQuN_Lab.datamodel.modelset.Element import Element
from RaQuN_Lab.datamodel.modelset.Model import Model
from RaQuN_Lab.datamodel.modelset.ModelSet import ModelSet
from RaQuN_Lab.datamodel.modelset.attribute.StringAttribute import StringAttribute
from RaQuN_Lab.utils.IDFactory import IDFactory


class MSLoader(DataLoader,ABC):
    def __init__(self, attribute_class: Type = StringAttribute):
        self.attribute_class = attribute_class
        self.url = None

    def set_attribute_class(self, ze_class: Type) -> None:
        self.attribute_class = ze_class

    def __hash__(self):
        return hash((self.__class__,self.attribute_class))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.attribute_class == other.attribute_class

    #returns a string encoding of the next attribute in the input file
    #should return the first attribute in the file when called the first time
    @abstractmethod
    def next_attribute(self):
        pass
    #move the parser to the next element (on the first call to the first element)
    @abstractmethod
    def next_element(self):
        pass

    # move the parser to the next model (on first call to the first model)
    @abstractmethod
    def next_model(self):
        pass
    #extracts the meta data from the input file and returns it as a metadata object
    @abstractmethod
    def read_meta_data(self):
        pass

    #returns true if the current attribute is the last attribute of the element
    @abstractmethod
    def last_attribute(self):
        pass

     # returns true if the parser has hit the end of the current element
    @abstractmethod
    def last_element(self):
        pass

    @abstractmethod
    def parse_ele_name(self):
        pass

    #returns true if the parser has hit the end of the file (the last attribute of the last element of the last model)
    @abstractmethod
    def last_model(self):
        pass

    def read_file(self, url: str) -> 'DataSet':
        self.url = url
        return self.parse_input()




    def parse_input(self):
        id_fac = IDFactory()
        metadata = self.read_meta_data()
        model_set = ModelSet()
        while not self.last_model():
            self.next_model()
            cur_model = Model(id_fac.generate_id())
            model_set.add_model(cur_model)
            while not self.last_element():
                self.next_element()
                cur_element = Element(id_fac.generate_id())
                cur_element.set_name(self.parse_ele_name())
                cur_model.add_element(cur_element)
                while not self.last_attribute():
                    cur_attr = self.attribute_class()
                    cur_attr.parse_string(self.next_attribute())
                    cur_element.add_attr(cur_attr)
        return DataSet(metadata,model_set)

