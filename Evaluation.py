from abc import ABC, abstractmethod
from typing import List, Union

from PyQuN_Lab.EvaluationMetrics import EvaluationMetric
from PyQuN_Lab.Experiment import Experiment, ExperimentSummary


class Aggregator(ABC):
    #assume that each sublist has the same length
    @abstractmethod
    def __private_aggregate(self, vals: List[List[float]]) -> List[float]:
        pass

    def aggregate(self, vals: List[List[float]]) -> List[float]:
        if vals is None or len(vals) == 0 or len(vals[0]):
            return None
        ze_len = len(vals[0])
        for li in vals:
            if len(li) != ze_len:
                raise Exception("All sublists are expected to have the same length")
        return self.__private_aggregate(vals)

class AverageAgg(Aggregator):

    def __private_aggregate(self, vals: List[List[float]]):
        avgs = []
        for i in range(len(vals[0])):
            avg = 0
            for li in vals:
                avg += li[i]
            avgs.append(avg/len(vals))
        return avgs

class MaxAgg(Aggregator):
    def __private_aggregate(self, vals: List[List[float]]) -> List[float]:
        maxs = []
        for i in range(len(vals[0])):
            max_val = float('-inf')
            for li in vals:
                if max_val < li[i]:
                    max_val = li[i]
            maxs.append(max_val)
        return maxs

class MinAgg(Aggregator):
    def __private_aggregate(self, vals: List[List[float]]) -> List[float]:
        mins = []
        for i in range(len(vals[0])):
            min_val = float('inf')
            for li in vals:
                if min_val > li[i]:
                    min_val = li[i]
            mins.append(min_val)
        return mins




class XYPlot:
    def plot(self, experiment : Union[Experiment, str], metric: EvaluationMetric, aggregator: Aggregator):
        if isinstance(experiment, str):
            experiment = Experiment.load(experiment)

        summary = ExperimentSummary.get_summary(experiment)

        []
        for d in experiment.get_datasets():
            for s in experiment.get_strategies():
                summary


