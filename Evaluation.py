from abc import ABC, abstractmethod
from typing import List, Union, Dict

from matplotlib import pyplot as plt
from tabulate import tabulate

from PyQuN_Lab.EvaluationMetrics import EvaluationMetric, FalseNegative, Weight
from PyQuN_Lab.Experiment import Experiment,  ResultsIterator, VaryDimension, VarySize, ExperimentResult, DoMatching


class Aggregator(ABC):
    #assume that each sublist has the same length
    @abstractmethod
    def _private_aggregate(self, vals: List[List[float]]) -> List[float]:
        pass

    def aggregate(self, vals: List[List[float]]) -> List[float]:
        if vals is None or len(vals) == 0:
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

class ResultView:
    def __init__(self,  experiment : Union[Experiment, str], run: int = 0, repetition: int = -1, dataset : str = None, agg:Aggregator = AverageAgg()):
        self.iterator = ResultsIterator(experiment)
        self.dataset = dataset
        self.repetition = repetition
        self.run = run
        self.agg = agg

    def print_statistics(self, metric: EvaluationMetric):
        e = self.iterator.get_experiment()
        if self.dataset is None:
            datasets = e.get_datasets()
        else:
            datasets = [self.dataset]
        for d in datasets:
            self._p_print_statistics(metric,d,self.repetition)

    def _p_print_statistics(self, metric: EvaluationMetric, dataset: str, repetition: int) -> None:
        """
        Pretty print the similarity statistics for each strategy within each dataset.
        """
        from collections import defaultdict

        # Prepare a list to store rows of our statistics table
        table_rows = []
        e = self.iterator.get_experiment()
        if repetition == -1:
            result_maps_list = [self.iterator.result_map(dataset, rep, self.run) for rep in
                                range(e.get_num_experiments())]
        else:
            result_maps_list = [self.iterator.result_map(dataset, repetition, self.run)]

        score_map = {}
        runtime_map = defaultdict(dict)
        all_stopwatch_keys = set()

        # Process strategies
        for strat in e.get_strategies():
            scores = []
            runtimes = defaultdict(list)

            for result_map in result_maps_list:
                if result_map[strat] is None:
                    continue

                # Evaluate the similarity score
                scores.append(metric.evaluate(result_map[strat]))

                # Collect runtime data, extracting `elapsed` times
                for key, timing_struct in result_map[strat].stopwatch.timers.items():
                    runtimes[key+" runtime"].append(timing_struct.elapsed)
                    all_stopwatch_keys.add(key+" runtime")

            # Aggregate the similarity scores and runtimes if data exists
            if scores:
                score_map[strat] = self.agg.aggregate([scores])[0]
                for key, times in runtimes.items():
                    runtime_map[strat][key] = self.agg.aggregate([times])[0]

        # Prepare the headers dynamically
        headers = ["Strategy", "Similarity Score"] + sorted(all_stopwatch_keys)

        # Build rows for the table
        for strategy in e.get_strategies():
            if strategy not in score_map:
                continue  # Skip strategies without results

            # Start the row with the strategy name and its similarity score
            row = [strategy, score_map[strategy]]

            # Add runtime values for each stopwatch key, or empty if missing
            for key in sorted(all_stopwatch_keys):
                row.append(runtime_map[strategy].get(key, ""))  # Default to empty if key not present

            table_rows.append(row)

        # Print dataset and repetition details
        print(f"\nDataset: {dataset}")
        print(f"Repetition: {repetition}")
        print(f"Run: {self.run}")

        # Print the table with tabulate for pretty output
        print(tabulate(table_rows, headers=headers, floatfmt=".4f", tablefmt="fancy_grid"))


class MatchingView:
    @staticmethod
    def display(res: 'ExperimentResult') -> None:
        """
        Displays a matching where each match is displayed in a table format.
        Attributes are ordered by their occurrence count: most common attributes
        appear first, and less-common attributes are shifted to the right.
        """
        matching = res.match
        for match in matching.get_matches():
            print(f"Match ID: {match.get_id()}")  # Display Match ID

            # Collect all attribute keys and their occurrence counts
            attribute_counts = {}
            all_keys = set()
            for element in match:
                for attr in element:
                    key = attr.value
                    all_keys.add(key)
                    attribute_counts[key] = attribute_counts.get(key, 0) + 1

            # Sort attributes by their count (descending), breaking ties alphabetically
            sorted_keys = sorted(all_keys, key=lambda k: (-attribute_counts[k], k))

            # Prepare table header
            header = ["Element ID", "Name"] + sorted_keys

            # Calculate column widths
            col_widths = [len(col) for col in header]
            rows = []

            # Prepare rows for each element
            for element in match:
                row = [str(element.get_id()), element.get_name() or ""]
                for key in sorted_keys:
                    value = next((str(attr.value) for attr in element if attr.value == key), "")
                    row.append(value)
                rows.append(row)
                col_widths = [max(col_widths[i], len(row[i])) for i in range(len(row))]

            # Print header
            formatted_header = " | ".join(
                f"{header[i]:<{col_widths[i]}}" for i in range(len(header))
            )
            print(formatted_header)
            print("-" * len(formatted_header))  # Separator

            # Print each row
            for row in rows:
                formatted_row = " | ".join(
                    f"{row[i]:<{col_widths[i]}}" for i in range(len(row))
                )
                print(formatted_row)

            print("\n")




class XYPlot:
    def plot(self, experiment : Union[Experiment, str], metric: EvaluationMetric, aggregator: Aggregator, excluded_strategies:List[str] = []):
       if isinstance(experiment, str):
           experiment = Experiment.load(experiment)
       iterator = ResultsIterator(experiment)
       error_matrix = iterator.to_error_matrix(metric)
       result_map = {}
       for dataset, dic2 in error_matrix.items():
           result_map[dataset] = {}
           for strat, error_mat in dic2.items():
               if strat not in excluded_strategies:
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
                # Plot the error values for the current strategy
                plt.plot(index_set, aggregated_errors, marker='o', linestyle='-', label=strat)

            # Add legend, grid, and adjust layout
            plt.legend(title="Strategies")
            plt.grid(True)
            plt.tight_layout()

            # Display the plot for the current dataset
            plt.show()




if __name__ == "__main__":
    from PyQuN_Lab.Experiment import ExperimentResult
    #XYPlot().plot("vary_dim", Weight(), AverageAgg(),["random"])
    view = ResultView("do_matching")
    view.print_statistics(Weight())
    #MatchingView.display(ExperimentResult.load("do_matching_all", "svd_k_10_high_dim_raqun", "hosp", 0, 0))