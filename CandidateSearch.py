from abc import ABC, abstractmethod
from typing import List, Tuple

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.distances import EuclideanDistance

from sklearn.neighbors import KDTree, BallTree

from PyQuN_Lab.Similarity import Similarity, WeightMetric
from PyQuN_Lab.DataModel import MatchCandidates, Match, MatchViolation, Element, ModelSet
from PyQuN_Lab.Stopwatch import Stopwatch
from PyQuN_Lab.Vectorization import MeanCountVectorizer, VectorizedModelSet, Vectorizer


class KNN(ABC):
    def timed_set_data(self, vm_set:VectorizedModelSet) -> Stopwatch:
        s = Stopwatch()
        s.start_timer("knn_set_data")
        self.set_data(vm_set)
        s.stop_timer("knn_set_data")
        return s

    def timed_get_neighbours(self, element: Element, size: int) -> Tuple[List[Element],Stopwatch]:
        s = Stopwatch()
        s.start_timer("knn_get_neighbours")
        li = self.get_neighbours(element,size)
        s.stop_timer("knn_get_neighbours")
        return li, s

    #takes as input a vectorized model set to initialize the KNN search
    @abstractmethod
    def set_data(self, vm_set:VectorizedModelSet) -> None:
        pass
    #takes as input an element form the vm_set and return the closest neighbours elements of the input element
    #the parameter size determines the size of the neighbourhood
    @abstractmethod
    def get_neighbours(self,element:Element, size:int) -> List[Element]:
        pass


class LSHKNN(KNN):

    def __init__(self, n_projections: int = 5, n_bits: int = 16) -> None:

        self.n_projections = n_projections
        self.n_bits = n_bits
        self.engine = None
        self.vm_set = None

    def set_data(self, vm_set: VectorizedModelSet) -> None:
        self.vm_set = vm_set
        projs = []
        for i in range(self.n_projections):
            rbp = RandomBinaryProjections(f'rbp_{i}', self.n_bits)
            projs.append(rbp)
        self.engine = Engine(vm_set.get_vec_mat().shape[1],lshashes=projs, distance=EuclideanDistance())
        for idx, vec in enumerate(vm_set.get_vec_mat()):
            self.engine.store_vector(vec, idx)

    def get_neighbours(self, element: Element, size: int) -> List[Element]:
        query_vec = self.vm_set.get_vec_mat()[self.vm_set.get_vec_index(element)]
        results = self.engine.neighbours(query_vec)

        neighbour_indices = [result[1] for result in results[:size]]
        return [self.vm_set.get_ele_by_index(idx) for idx in neighbour_indices]





class BFKNN(KNN):
    def __init__(self, similarity:Similarity = WeightMetric()) -> None:
        self.similarity = similarity
        self.vm_set = None


    def set_data(self, vm_set):
        self.vm_set =  vm_set

    def get_neighbours(self, element, size):
        ms = self.vm_set.get_model_set()
        sim = []
        for m in ms:
            if m.get_id() == element.get_model_id():
                continue
            for ele2 in m:
                sim.append((ele2,self.similarity.similarity(Match({element,ele2},do_check=False))))
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
    def timed_candidate_search(self, model_set:ModelSet) -> Tuple[MatchCandidates, Stopwatch]:
        s = Stopwatch()
        s.start_timer("candidate_search")
        res, s2 = self.candidate_search(model_set)
        s.stop_timer("candidate_search")
        s.merge(s2)

        return res, s

    # takes as input a model set and produces a candidate set of matches which are non mutually exclusive
    @abstractmethod
    def candidate_search(self, model_set:ModelSet) -> Tuple[MatchCandidates, Stopwatch]:
        pass


class NNCandidateSearch(CandidateSearch):
    def __init__(self,neighbourhood_size:int = 10, vectorizer:Vectorizer = MeanCountVectorizer(), knn:KNN = TreeKNN(), reduction: 'DimensionalityReduction' = None) -> None:
        self.neighbourhood_size = neighbourhood_size
        self.vectorizer = vectorizer
        self.knn = knn
        self.reduction = reduction

    def candidate_search(self, model_set):
        mc = MatchCandidates()

        vm_set, s = model_set.timed_vectorize(self.vectorizer, self.reduction)
        s1 = self.knn.timed_set_data(vm_set)
        s1.merge(s)
        for ele in model_set.get_elements():
            neighbours, s2 = self.knn.timed_get_neighbours(ele,self.neighbourhood_size)
            s1.merge(s2)
            for neigh in neighbours:
                match = Match()
                try:
                    match = match.add_elements(ele,neigh)
                    mc.add_matches(match)
                except MatchViolation as e:
                    pass
        return mc, s1



