import csv
import pickle
from typing import Type, List
from abc import ABC, abstractmethod

import numpy as np

from PyQuN_Lab.DataModel import ModelSet, Model, Element, StringAttribute, DataModel
from PyQuN_Lab.IDFactory import IDFactory
from PyQuN_Lab.Utils import store_obj, load_obj


class MetaData:
    pass


class DataSet:
    def __init__(self, metadata: MetaData, data_model: DataModel):
        self.metadata = metadata
        self.data_model = data_model

    def get_data_model(self) -> DataModel:
        return self.data_model

    def store(self, path: str) -> None:
        store_obj(self,path)

    @staticmethod
    def load(path: str) -> 'DataSet':
        return load_obj(path)


class DataLoader(ABC):
    @abstractmethod
    def read_file(self, url: str) -> DataSet:
        pass


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

    def read_file(self, url: str) -> DataSet:
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


    #takes as input an array representation of a modelset
    #each row of the matrix represents an element.
    #the entries of the matrix are the elements attributes
    #the second list contains the size of each model
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




#parses a modelSet stored as csv format.
#Asumptions:
#The csv file contains no metadata
#Each row of the csv is assumed to represent an element.
#Each column of the csv represents an attribute (except for the fist 3 columns of each row)
#The elements of each model are grouped.
#the first 3 collums of each  row contain the following information:
# row[0] = Model_ID ; row[1] = element_ID ; row[2] = element_name
class CSVLoader(ArrayLoader):
    def __init__(self, attribute_class: Type = StringAttribute):
        super().__init__(attribute_class)

    def read_file(self, url: str) -> DataSet:
        rows = []
        names = []
        lengths = []
        with open(url, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            cur_id = None
            count = 0
            for row in reader:
                # something is wrong with the encoding, the metadata column entries and the first attribute (model-id, ele-id, ele-name) are
                # seperated by commas while the remaining with semicolons
                first_element_split = row[0].split(',')
                rows.append([first_element_split[3]] + row[1:])
                names.append(first_element_split[2])
                if cur_id is None:
                    cur_id = first_element_split[0]
                elif not cur_id == first_element_split[0]:
                    lengths.append(count)
                    cur_id = first_element_split[0]
                    count = 0
                count += 1
            lengths.append(count)
            self.rows = rows
            self.separations = lengths
            self.element_names = names
        return self.parse_input()







if __name__ == "__main__":
    loader = CSVLoader("C:/Users/41766/Desktop/full_subjects/hospitals.csv",StringAttribute)
    out = loader.parse_input()
    print(out.model_set)
