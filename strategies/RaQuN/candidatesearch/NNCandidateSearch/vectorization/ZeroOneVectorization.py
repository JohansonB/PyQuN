
#create a 0-1 vector where each entry of the vetor represents an attribute in the attribute space
#for each element a 0 represents the absence and a 1 the presence of said attribute
import numpy as np

from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.vectorization.Vectorizer import Vectorizer


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

