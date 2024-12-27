from typing import Union, List, Dict

from matplotlib import pyplot as plt

from RaQuN_Lab.experiment.Experiment import Experiment
from RaQuN_Lab.experiment.ResultIterator import ResultsIterator


class XYPlot:
    def plot(self, experiment: Union['Experiment', str], metric: 'EvaluationMetric', aggregator: 'Aggregator',
             excluded_strategies: List[str] = []):
        if isinstance(experiment, str):
            experiment = Experiment.load(experiment)
        iterator = ResultsIterator(experiment)
        error_matrix = iterator.to_error_matrix(metric)
        runtime_matrix = iterator.runtime_matrix()
        result_map = {}
        runtime_map = {}

        for dataset, dic2 in error_matrix.items():
            result_map[dataset] = {}
            runtime_map[dataset] = {}
            for strat, error_mat in dic2.items():
                if strat not in excluded_strategies:
                    result_map[dataset][strat] = aggregator.aggregate(error_mat)
                    runtime_map[dataset][strat] = aggregator.aggregate(runtime_matrix[dataset][strat])

        self.plot_results(result_map, runtime_map, experiment)

    def plot_results(self, result_map: Dict[str, Dict[str, List[float]]],
                     runtime_map: Dict[str, Dict[str, List[float]]], experiment: Experiment):
        """
        Plot the aggregated results stored in `result_map`.
        Each dataset will be plotted separately, comparing different strategies.
        """
        index_set = experiment.index_set()
        index_name = experiment.index_name()

        for dataset, strategies in result_map.items():
            plt.figure(figsize=(10, 6))
            plt.title(f"Comparison of Strategies for {dataset} - Similarity")
            plt.xlabel(index_name)
            plt.ylabel("Similarity")

            for strat, aggregated_errors in strategies.items():
                plt.plot(index_set, aggregated_errors, marker='o', linestyle='-', label=strat)

            plt.legend(title="Strategies")
            plt.grid(True)
            plt.tight_layout()

            plt.show()

        for dataset, strategies in runtime_map.items():
            plt.figure(figsize=(10, 6))
            plt.title(f"Comparison of Strategies for {dataset} - Runtime")
            plt.xlabel(index_name)
            plt.ylabel("Runtime (s)")

            for strat, aggregated_runtimes in strategies.items():
                plt.plot(index_set, aggregated_runtimes, marker='o', linestyle='-', label=strat)

            plt.legend(title="Strategies")
            plt.grid(True)
            plt.tight_layout()

            plt.show()
