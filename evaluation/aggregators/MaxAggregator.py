from typing import List

from RaQuN_Lab.evaluation.aggregators.Aggregator import Aggregator


class MaxAgg(Aggregator):
    def _private_aggregate(self, vals: List[List[float]]) -> List[float]:
        maxs = []
        for i in range(len(vals[0])):
            max_val = float('-inf')
            for li in vals:
                if max_val < li[i]:
                    max_val = li[i]
            maxs.append(max_val)
        return maxs
