from RaQuN_Lab.datamodel.matching.Match import Match
from RaQuN_Lab.datamodel.matching.Matching import Matching
from RaQuN_Lab.datamodel.matching.similarities.match.JaccardIndex import JaccardIndex
from RaQuN_Lab.strategies.RaQuN.matching.Matching import MatchingStrategy


class GreedyMatching(MatchingStrategy):
    def __init__(self, similarity: 'Similarity' = JaccardIndex(), filter_threshold: float = 0.001) -> None:
        self.similarity = similarity
        self.filter_threshold = filter_threshold


    def match(self, model_set: 'ModelSet', candidates: 'MatchCandidates' = None) -> 'Matching':
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

