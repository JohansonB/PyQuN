from abc import ABC, abstractmethod

from PyQuN.Similarity import Similarity
from PyQuN_Lab.DataModel import Matching, MatchViolation, Match, ModelSet, MatchCandidates
from PyQuN_Lab.Similarity import JaccardIndex


class MatchingStrategy(ABC):
    # the matching function which takes as input the model_set, a Candidate_set, a similarity and produces a matching
    @abstractmethod
    def match(self, model_set: ModelSet, candidates: MatchCandidates = None) -> Matching:
        pass


class RandomMatching(MatchingStrategy):
    def __init__(self, seed: int = None) -> None:
        self.seed = seed

    def match(self, model_set: ModelSet, candidates: MatchCandidates = None) -> Matching:
        pass


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
