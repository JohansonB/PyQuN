from abc import ABC, abstractmethod
import numpy as np



class Vectorizer(ABC):



    @abstractmethod
    def vectorize(self, element:'Element') -> np.ndarray:
        #takes as input an element and returns a vector representation of it
        pass

    @abstractmethod
    def dim(self) -> int:
        pass


#create a 0-1 vector where each entry of the vetor represents an attribute in the attribute space
#for each element a 0 represents the absence and a 1 the presence of said attribute
class ZeroOneVectorizer(Vectorizer):
    def __init__(self, model_set:'ModelSet'):
        self.attr_index_map = {}
        att_count = 0
        # iterate over all elements and attributes and store in a dic the mapping from attributes to vector indices
        for elements in model_set:
            for element in elements:
                for attribute in element:
                    if attribute not in self.attr_index_map:
                        self.attr_index_map[attribute] = att_count
                        att_count += 1
        self.vec_dim = len(self.attr_index_map)


    def vectorize(self, element:'Element'):
        vec = np.zeros(self.vec_dim)
        for attr in element:
            vec[self.attr_index_map[attr]] = 1
        return vec

    def dim(self):
        return self.vec_dim


class MeanCountVectorizer(Vectorizer):

    def vectorize(self, element:'Element'):
        vec = np.zeros(2)
        vec[0] = len(element)
        ave = 0
        for att in element:
            ave += len(att)
        ave /= len(element)
        vec[1] = ave
        return vec

    def dim(self):
        return 2



class VectorizedModelSet:
    #takes as input a vectorizer and augments each element with a vector representation of it
    def __init__(self, model_set:'ModelSet', vectorizer:Vectorizer) -> None:
        self.model_set = model_set
        self.index_id_map = []
        self.id_index_map = {}
        if vectorizer is not None:
            elements = self.model_set.get_elements()
            self.vec_mat = np.zeros((len(elements), vectorizer.dim()), dtype=np.float64)
            count = 0
            for element in elements:
                self.index_id_map.append(element.get_id())
                self.id_index_map[element] = count
                self.vec_mat[count] = vectorizer.vectorize(element)
                count += 1

    def get_model_set(self):
        return self.model_set

    def get_vec_mat(self):
        return self.vec_mat

    def get_vec_index(self, element:'Element') -> int:
        assert element in self.id_index_map
        return self.id_index_map[element]

    def get_ele_by_index(self, index:int) -> 'Element':

        return self.model_set.get_by_id(self.index_id_map[index])





