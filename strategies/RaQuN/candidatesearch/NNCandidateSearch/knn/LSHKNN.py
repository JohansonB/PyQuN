from typing import List

from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.knn.KNN import KNN
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.distances import EuclideanDistance

class LSHKNN(KNN):

    def __init__(self, n_projections: int = 5, n_bits: int = 16) -> None:

        self.n_projections = n_projections
        self.n_bits = n_bits
        self.engine = None
        self.vm_set = None

    def set_data(self, vm_set: 'VectorizedModelSet') -> None:
        self.vm_set = vm_set
        projs = []
        for i in range(self.n_projections):
            rbp = RandomBinaryProjections(f'rbp_{i}', self.n_bits)
            projs.append(rbp)
        self.engine = Engine(vm_set.get_vec_mat().shape[1],lshashes=projs, distance=EuclideanDistance())
        for idx, vec in enumerate(vm_set.get_vec_mat()):
            self.engine.store_vector(vec, idx)

    def get_neighbours(self, element: 'Element', size: int) -> List['Element']:
        query_vec = self.vm_set.get_vec_mat()[self.vm_set.get_vec_index(element)]
        results = self.engine.neighbours(query_vec)

        neighbour_indices = [result[1] for result in results[:size]]
        return [self.vm_set.get_ele_by_index(idx) for idx in neighbour_indices]



