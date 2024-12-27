import numpy as np

from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.vectorization.Vectorizer import Vectorizer


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


