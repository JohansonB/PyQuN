from abc import ABC, abstractmethod
from typing import Tuple, List

from RaQuN_Lab.utils.Stopwatch import Stopwatch


class KNN(ABC):
    def timed_set_data(self, vm_set:'VectorizedModelSet') -> 'Stopwatch':
        s = Stopwatch()
        s.start_timer("knn_set_data")
        self.set_data(vm_set)
        s.stop_timer("knn_set_data")
        return s

    def timed_get_neighbours(self, element: 'Element', size: int) -> Tuple[List['Element'],Stopwatch]:
        s = Stopwatch()
        s.start_timer("knn_get_neighbours")
        li = self.get_neighbours(element,size)
        s.stop_timer("knn_get_neighbours")
        return li, s

    #takes as input a vectorized model set to initialize the KNN search
    @abstractmethod
    def set_data(self, vm_set:'VectorizedModelSet') -> None:
        pass
    #takes as input an element form the vm_set and return the closest neighbours elements of the input element
    #the parameter size determines the size of the neighbourhood
    @abstractmethod
    def get_neighbours(self,element:'Element', size:int) -> List['Element']:
        pass
