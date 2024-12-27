from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.NNCandidateSearch import NNCandidateSearch
from RaQuN_Lab.strategies.RaQuN.matching.GreedyMatching import GreedyMatching
from RaQuN_Lab.strategies.Strategy import Strategy


class VanillaRaQuN(Strategy):

    def __init__(self, name:str, candidate_search: 'CandidateSearch' = NNCandidateSearch(), matching_strategy: 'MatchingStrategy' = GreedyMatching()) -> None:
        self.search = candidate_search
        self.match_strat = matching_strategy
        super().__init__(name)

    def match(self, data_model: 'ModelSet') -> ['Matching', 'Stopwatch']:
        cans, s = self.search.timed_candidate_search(data_model)
        return self.match_strat.match(data_model, cans), s