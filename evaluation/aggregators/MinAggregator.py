from typing import List

from RaQuN_Lab.evaluation.aggregators.Aggregator import Aggregator


class MinAgg(Aggregator):
    def _private_aggregate(self, vals: List[List[float]]) -> List[float]:
        mins = []
        for i in range(len(vals[0])):
            min_val = float('inf')
            for li in vals:
                if min_val > li[i]:
                    min_val = li[i]
            mins.append(min_val)
        return mins
