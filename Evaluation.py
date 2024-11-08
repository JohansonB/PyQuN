from abc import ABC, abstractmethod
from typing import List, Union, Dict

from matplotlib import pyplot as plt

from PyQuN_Lab.EvaluationMetrics import EvaluationMetric, FalseNegative
from PyQuN_Lab.Experiment import Experiment,  ResultsIterator, VaryDimension, VarySize, ExperimentResult


class Aggregator(ABC):
    #assume that each sublist has the same length
    @abstractmethod
    def _private_aggregate(self, vals: List[List[float]]) -> List[float]:
        pass

    def aggregate(self, vals: List[List[float]]) -> List[float]:
        if vals is None or len(vals) == 0 or len(vals[0]):
            return None
        ze_len = len(vals[0])
        for li in vals:
            if len(li) != ze_len:
                raise Exception("All sublists are expected to have the same length")
        return self._private_aggregate(vals)

class AverageAgg(Aggregator):

    def _private_aggregate(self, vals: List[List[float]]):
        avgs = []
        for i in range(len(vals[0])):
            avg = 0
            for li in vals:
                avg += li[i]
            avgs.append(avg/len(vals))
        return avgs

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

class XYPlot:
    def plot(self, experiment : Union[Experiment, str], metric: EvaluationMetric, aggregator: Aggregator):
       if isinstance(experiment, str):
           experiment = Experiment.load(experiment)
       iterator = ResultsIterator(experiment)
       iterator.evaluate_metric(metric)
       error_matrix = iterator.to_error_matrix(metric)
       result_map = {}
       for dataset, dic2 in error_matrix.items():
           result_map[dataset] = {}
           for strat, error_mat in dic2.items():
               result_map[dataset][strat] = aggregator.aggregate(error_mat)
       self.plot_results(result_map, experiment)

    def plot_results(self, result_map: Dict[str, Dict[str, List[float]]], experiment: Experiment):
        """
        Plot the aggregated results stored in `result_map`.
        Each dataset will be plotted separately, comparing different strategies.
        """
        # Get the index set from the experiment
        index_set = experiment.index_set()

        # Get the name of the x-axis from the experiment
        index_name = experiment.index_name()

        # Iterate through each dataset in the result_map
        for dataset, strategies in result_map.items():
            plt.figure(figsize=(10, 6))
            plt.title(f"Comparison of Strategies for {dataset}")
            plt.xlabel(index_name)  # Use the index_name for the x-axis label
            plt.ylabel("Error Value")

            # For each strategy, plot the aggregated error values
            for strat, aggregated_errors in strategies.items():
                print(index_set)
                print("\n")
                print(aggregated_errors)
                # Plot the error values for the current strategy
                plt.plot(index_set, aggregated_errors, marker='o', linestyle='-', label=strat)

            # Add legend, grid, and adjust layout
            plt.legend(title="Strategies")
            plt.grid(True)
            plt.tight_layout()

            # Display the plot for the current dataset
            plt.show()

if __name__ == "__main__":
    XYPlot().plot("vary_dim", FalseNegative(), AverageAgg())