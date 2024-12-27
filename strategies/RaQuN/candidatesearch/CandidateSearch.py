from abc import ABC, abstractmethod
from typing import Tuple

from RaQuN_Lab.utils.Stopwatch import Stopwatch


class CandidateSearch(ABC):
    def timed_candidate_search(self, model_set:'ModelSet') -> Tuple['MatchCandidates', 'Stopwatch']:
        s = Stopwatch()
        s.start_timer("candidate_search")
        res, s2 = self.candidate_search(model_set)
        s.stop_timer("candidate_search")
        s.merge(s2)

        return res, s

    # takes as input a model set and produces a candidate set of matches which are non mutually exclusive
    @abstractmethod
    def candidate_search(self, model_set:'ModelSet') -> Tuple['MatchCandidates', 'Stopwatch']:
        pass



