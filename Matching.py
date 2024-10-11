from abc import ABC, abstractmethod

from PyQuN_Lab.DataModel import Matching


class MatchingStrategy(ABC):
    #the matching function which takes as input the model_set, a Candidate_set, a similarity and produces a matching
    @abstractmethod
    def match(self,model_set,candidates, similarity):
        pass


class GreedyMatching(MatchingStrategy):

    def match(self, model_set, candidates, similarity, filter_threshold=0.0000000000001):
        matching = Matching()
        matching.trivial_matching(model_set)
        candidates.filter(similarity,filter_threshold)
        candidates.sort(similarity)
        while not candidates.is_empty():
            next_candidate = candidates.pop()
            cur_matches = []
            for ele in next_candidate:
                cur_matches.append(matching.get_match_by_element(ele))
            print(cur_matches)
            if similarity.should_match(cur_matches):
                Matching.merge_matches(cur_matches, do_check=False)
        return matching


