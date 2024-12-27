from typing import List

from RaQuN_Lab.datamodel.matching.similarities.matching.statistics.Statistic import Statistic


class Sum(Statistic):
    def evaluate(self, match_scores: List[float]) -> float:
        return sum(match_scores)