from typing import List

from RaQuN_Lab.evaluation.aggregators.Aggregator import Aggregator


class AverageAgg(Aggregator):

    def _private_aggregate(self, vals: List[List[float]]):
        avgs = []
        for i in range(len(vals[0])):
            avg = 0
            for li in vals:
                avg += li[i]
            avgs.append(avg/len(vals))
        return avgs
