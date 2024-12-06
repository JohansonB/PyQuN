from abc import ABC, abstractmethod
from typing import List

from PyQuN_Lab.Similarity import Similarity
from PyQuN_Lab.DataModel import Matching, MatchViolation, Match, ModelSet, MatchCandidates
from PyQuN_Lab.Similarity import JaccardIndex

class MatchingStrategy(ABC):
    # the matching function which takes as input the model_set, a Candidate_set, a similarity and produces a matching
    @abstractmethod
    def match(self, model_set: ModelSet, candidates: MatchCandidates = None) -> Matching:
        pass

#This datastructure holds an array which marks for each element if it has already been selected for a matching or not.
#When queried with an index in the range (0,self.free-1) it returns the index of the next free element and does all
#the remaining required bookkeeping
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


class RandomMatching(MatchingStrategy):

    @staticmethod
    def false_list(size: int) -> List[int]:
        ret = []
        for i in range(size):
            ret.append(False)
        return ret

    def __init__(self, seed: int = None) -> None:
        self.seed = seed

    def match(self, model_set: ModelSet, candidates: MatchCandidates = None) -> Matching:
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

        return matching







class GreedyMatching(MatchingStrategy):
    def __init__(self, similarity: Similarity = JaccardIndex(), filter_threshold: float = 0.001) -> None:
        self.similarity = similarity
        self.filter_threshold = filter_threshold


    def match(self, model_set: ModelSet, candidates: MatchCandidates = None) -> Matching:
        if candidates is None:
            raise Exception("The GreedyMatching Strategy requires MatchCandidates as input")

        matching = Matching.trivial_matching(model_set)
        candidates.filter(self.similarity, self.filter_threshold)
        candidates.sort(self.similarity)
        while not candidates.is_empty():
            next_candidate = candidates.pop()
            cur_matches = set()
            for ele in next_candidate:
                cur_matches.add(matching.get_match_by_element(ele))

            if len(cur_matches) > 1 and Match.valid_merge(*cur_matches) and self.similarity.should_match(cur_matches):
                matching.merge_matches(*cur_matches, do_check=False)

        return matching


class ExploringGreedyMatching(GreedyMatching):
    def __init__(self, similarity: Similarity = JaccardIndex(), filter_threshold: float = 0.001, shuffle_threshold : float = 0.2):
        super(ExploringGreedyMatching, self).__init__(similarity, filter_threshold)
        self.shuffle_threshold = shuffle_threshold

    def match(self, model_set: ModelSet, candidates: MatchCandidates = None) -> Matching:
        candidates.filter(self.similarity, self.filter_threshold)
        candidates.sort(self.similarity)
        candidates.shuffle(self.shuffle_threshold)
        return super().match(model_set,candidates)




if __name__ == "__main__":
    from PyQuN_Lab.Tests import test_element, test_model

    #Model tests
    #input
    b = test_element("b","c","d","e")
    a = test_element("a","b","c")
    c = test_element("a","r", "f","g","h")
    d = test_element("a","f","h","t","i","j")
    e = test_element("l","o","p","q","a","s")
    f = test_element("l","r","a","s")
    g = test_element("h","i","j","t")

    model1 = test_model(a, b)
    model2 = test_model(c,d)
    model3 = test_model(e,f)
    model4 = test_model(g)

    ms = ModelSet()
    ms.add_model(model1)
    ms.add_model(model2)
    ms.add_model(model3)

    ran = RandomMatching()
    ze_match = ran.match(ms)
    ze_match.store("test")
    ze_match2 = Matching.load("test")
    print(ze_match)
    print('\n')
    print(ze_match2)
