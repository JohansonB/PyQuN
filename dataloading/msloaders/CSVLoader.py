
#parses a modelSet stored as csv format.
#Asumptions:
#The csv file contains no metadata
#Each row of the csv is assumed to represent an element.
#Each column of the csv represents an attribute (except for the fist 3 columns of each row)
#The elements of each model are grouped.
#the first 3 collums of each  row contain the following information:
# row[0] = Model_ID ; row[1] = element_ID ; row[2] = element_name
import csv
from typing import Type

from RaQuN_Lab.dataloading.msloaders.ArrayLoader import ArrayLoader
from RaQuN_Lab.datamodel.DataSet import DataSet
from RaQuN_Lab.datamodel.modelset.attribute.StringAttribute import StringAttribute


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


