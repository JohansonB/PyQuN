from RaQuN_Lab.datamodel.matching.Match import Match
from RaQuN_Lab.datamodel.matching.MatchCandidates import MatchCandidates
from RaQuN_Lab.datamodel.matching.Matching import Matching
from RaQuN_Lab.datamodel.matching.similarities.match.JaccardIndex import JaccardIndex
from RaQuN_Lab.datamodel.matching.similarities.matching.SimilarityScore import SimilarityScore
from RaQuN_Lab.strategies.RaQuN.matching.Matching import MatchingStrategy


import heapq

class RunTrace():
    def __init__(self):
        self.trace = []

class Jaccard_avrage:
    def __init__(self, num_models : int):
        self.jac = JaccardIndex()
        self.cur = 0
        self.n = num_models

    def update(self,*matches):
        assert len(matches) == 2

        




class BeamSearchGreedyMatching(MatchingStrategy):
    def __init__(self, v : 'EvaluationScore' = SimilarityScore(JaccardIndex()), beam_length : int = 10):
        self.v = v
        self.beam_length = beam_length
        self.heap = []

    def match(self, model_set: 'ModelSet', candidates: 'MatchCandidates' = None) -> 'Matching':
        if candidates is None:
            raise Exception("The GreedyMatching Strategy requires MatchCandidates as input")
        heapq.heappush(self.heap,self.v.evaluate())


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



