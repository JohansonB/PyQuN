from abc import ABC, abstractmethod
from typing import List

from sklearn.neighbors import KDTree, BallTree

from PyQuN_Lab.Similarity import Similarity
from PyQuN_Lab.DataModel import MatchCandidates, Match, MatchViolation, Element, ModelSet
from PyQuN_Lab.Vectorization import MeanCountVectorizer, VectorizedModelSet, Vectorizer


class KNN(ABC):
    #takes as input a vectorized model set to initialize the KNN search
    @abstractmethod
    def set_data(self, vm_set:VectorizedModelSet) -> None:
        pass
    #takes as input an element form the vm_set and return the closest neighbours elements of the input element
    #the parameter size determines the size of the neighbourhood
    @abstractmethod
    def get_neighbours(self,element:Element, size:int) -> List[Element]:
        pass


class BFKNN(KNN):
    def __init__(self, similarity:Similarity) -> None:
        self.similarity = similarity
        self.vm_set = None


    def set_data(self, vm_set):
        self.vm_set =  vm_set

    def get_neighbours(self, element, size):
        ms = self.vm_set.get_model_set()
        sim = []
        for ele2 in ms.get_elements():
            sim.append((ele2,self.similarity.similarity(Match({element,ele2}))))
        sim.sort(key=lambda x : x[1],reverse=True)
        return [x[0] for x in sim[:size]]

class TreeKNN(KNN):
    def __init__(self, tree_type="KDTree", leaf_size=40, metric="euclidean" ):
        self.tree_type = tree_type
        self.leaf_size = leaf_size
        self.metric = metric
        self.tree = None
        self.data_matrix = None
        self.vm_set = None

    def set_data(self, vm_set):

        if self.tree_type == "KDTree":
            self.tree = KDTree(vm_set.get_vec_mat(), leaf_size=self.leaf_size, metric=self.metric)
        elif self.tree_type == "BallTree":
            self.tree = BallTree(vm_set.get_vec_mat(), leaf_size=self.leaf_size, metric=self.metric)
        else:
            Exception("parameter type needs to be \"KDTree\" or \"BallTree\"")
        self.data_matrix = vm_set.get_vec_mat()
        self.vm_set = vm_set

    def get_neighbours(self, element, size):
        ret = []
        dist, ind = self.tree.query(self.data_matrix[self.vm_set.get_vec_index(element)].reshape(1,-1), k=size)
        ind = ind[0]
        for i in ind:
            ret.append(self.vm_set.get_ele_by_index(int(i)))
        return ret


class CandidateSearch(ABC):
    # takes as input a model set and produces a candidate set of matches which are non mutually exclusive
    @abstractmethod
    def candidate_search(self, model_set:ModelSet) -> MatchCandidates:
        pass


class NNCandidateSearch(CandidateSearch):
    def __init__(self,neighbourhood_size:int = 10, vectorizer:Vectorizer = MeanCountVectorizer(), knn:KNN = TreeKNN()) -> None:
        self.neighbourhood_size = neighbourhood_size
        self.vectorizer = vectorizer
        self.knn = knn
    def candidate_search(self, model_set):
        mc = MatchCandidates()

        vm_set = model_set.vectorize(self.vectorizer)
        self.knn.set_data(vm_set)
        for ele in model_set.get_elements():
            assert ele in vm_set.id_index_map
            neighbours = self.knn.get_neighbours(ele,self.neighbourhood_size)
            for neigh in neighbours:
                match = Match()
                try:
                    match = match.add_elements(ele,neigh)
                    mc.add_matches(match)
                except MatchViolation as e:
                    pass
        return mc



