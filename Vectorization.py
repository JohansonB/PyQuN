from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class DimensionalityReduction(ABC):
    @abstractmethod
    def reduce(self, in_mat : np.ndarray) -> np.ndarray:
        pass


class SVDReduction(DimensionalityReduction):
    def __init__(self, n_components = 10):
        self.n_components = n_components

    def reduce(self, in_mat: np.ndarray) -> np.ndarray:
        svd = TruncatedSVD(n_components=self.n_components)
        return svd.fit_transform(csr_matrix(in_mat))


class Vectorizer(ABC):



    @abstractmethod
    def vectorize(self, element:'Element') -> np.ndarray:
        #takes as input an element and returns a vector representation of it
        pass

    @abstractmethod
    def dim(self) -> int:
        pass

    @abstractmethod
    def innit(self, m_s:'ModelSet') -> None:
        pass


#create a 0-1 vector where each entry of the vetor represents an attribute in the attribute space
#for each element a 0 represents the absence and a 1 the presence of said attribute
class ZeroOneVectorizer(Vectorizer):
    def __init__(self):
        self.attr_index_map = {}
        self.vec_dim = 0

    def innit(self, m_s: 'ModelSet') -> None:
        att_count = 0
        # iterate over all elements and attributes and store in a dic the mapping from attributes to vector indices
        for element in m_s.get_elements():
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

    def innit(self, m_s: 'ModelSet') -> None:
        pass

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
    def __init__(self, model_set:'ModelSet', vectorizer:Vectorizer, dim_reduction: DimensionalityReduction = None) -> None:
        self.dim_reduction = dim_reduction
        self.model_set = model_set
        self.index_id_map = []
        self.id_index_map = {}
        vectorizer.innit(model_set)
        if vectorizer is not None:
            elements = self.model_set.get_elements()
            self.vec_mat = np.zeros((len(elements), vectorizer.dim()), dtype=np.float64)
            count = 0
            for element in elements:
                self.index_id_map.append(element.get_id())
                self.id_index_map[element] = count
                self.vec_mat[count] = vectorizer.vectorize(element)
                count += 1

        if dim_reduction is not None:
            self.vec_mat = dim_reduction.reduce(self.vec_mat)

    def get_model_set(self):
        return self.model_set


    def get_vec_mat(self):
        return self.vec_mat

    def get_vec_index(self, element:'Element') -> int:
        assert element in self.id_index_map
        return self.id_index_map[element]

    def get_ele_by_index(self, index:int) -> 'Element':

        return self.model_set.get_by_id(self.index_id_map[index])





