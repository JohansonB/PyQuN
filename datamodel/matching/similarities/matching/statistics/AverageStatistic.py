from typing import List

from RaQuN_Lab.datamodel.matching.similarities.matching.statistics.Statistic import Statistic


class AverageStatistic(Statistic):
    def evaluate(self, match_scores: List[float]) -> float:
        return sum(match_scores) / len(match_scores)