#This datastructure holds an array which marks for each element if it has already been selected for a matching or not.
#When queried with an index in the range (0,self.free-1) it returns the index of the next free element and does all
#the remaining required bookkeeping
from typing import List, Tuple

from RaQuN_Lab.datamodel.matching.Match import Match
from RaQuN_Lab.datamodel.matching.Matching import Matching
from RaQuN_Lab.strategies.Strategy import Strategy
from RaQuN_Lab.utils.Stopwatch import Stopwatch


class ModelTrackingStruct:
    def __init__(self, elements: List['Element']) -> None:
        self.arr = RandomMatching.false_list(len(elements))
        self.elements = elements
        self.free = len(elements)

    def get_next_free_element(self, rand: int) -> int:
        count = 0
        count2 = 0
        while count <= rand + count2:
            if self.arr[count]:
                count2 += 1
            if count == rand+count2:
                self.arr[count] = True
                self.free -= 1
                return self.elements[count]
            count += 1

#this structure does the higher level bookeeping. When given an index it returns the next model of said index that
#is free
class MSTrackingStruct:
    def __init__(self, model_trackers: List['ModelTrackingStruct']) -> None:
        self.arr = RandomMatching.false_list(len(model_trackers))
        self.trackers = model_trackers
        self.free = len(model_trackers)

    #assume indices are sorted
    def get_next_free_models(self, indices: List[int]) -> List['ModelTrackingStruct']:
        count = 0
        count2 = 0
        indices_arr_count = 0
        ret = []
        while indices_arr_count < len(indices):
            if self.arr[count]:
                count2 += 1
            if indices[indices_arr_count]+count2 == count:
                ret.append(self.trackers[count])
                indices_arr_count += 1
                if self.trackers[count].free == 1:
                    self.arr[count] = True
                    self.free -= 1

            count += 1

        return ret


class RandomMatching(Strategy):

    @staticmethod
    def false_list(size: int) -> List[int]:
        ret = []
        for i in range(size):
            ret.append(False)
        return ret

    def __init__(self,name : str, seed: int = None) -> None:
        super().__init__(name)
        self.seed = seed

    def match(self, model_set: 'ModelSet') -> Tuple['Matching', 'Stopwatch']:
        import random
        matching = Matching()
        if self.seed is not None:
            random.seed(self.seed)

        trackers = []
        for m in model_set:
            trackers.append(ModelTrackingStruct(list(m.get_elements())))

        ms_tracker = MSTrackingStruct(trackers)

        while ms_tracker.free > 0:
            cur_match = Match()
            match_size = random.randint(1, ms_tracker.free)
            model_indices = random.sample(range(0, ms_tracker.free), match_size)
            model_indices.sort()
            cur_trackers = ms_tracker.get_next_free_models(model_indices)
            for track in cur_trackers:
                ele_idx = random.randint(0, track.free-1)
                cur_match = cur_match.add_elements(track.get_next_free_element(ele_idx))
            matching.add_match(cur_match)

        return matching, Stopwatch()


