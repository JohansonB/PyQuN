from collections import defaultdict
from typing import Union, List

from tabulate import tabulate

from RaQuN_Lab.evaluation.aggregators.AverageAggregator import AverageAgg
from RaQuN_Lab.experiment.ResultIterator import ResultsIterator


class NormalizedResultView:
    def __init__(self, experiment: Union['Experiment', str], run: int = 0, repetition: int = -1, agg: 'Aggregator' = AverageAgg(),
                 exclude_strategies: List[str] = None, exclude_datasets: List[str] = None):
        self.iterator = ResultsIterator(experiment)
        self.repetition = repetition
        self.run = run
        self.agg = agg
        self.exclude_strategies = exclude_strategies if exclude_strategies else []
        self.exclude_datasets = exclude_datasets if exclude_datasets else []

    def print_normalized_statistics(self, metric: 'EvaluationMetric', baseline_strategy: str):
        e = self.iterator.get_experiment()
        datasets = [d for d in e.get_datasets() if d not in self.exclude_datasets]

        normalized_scores = defaultdict(list)

        for dataset in datasets:
            if self.repetition == -1:
                result_maps_list = [self.iterator.result_map(dataset, rep, self.run) for rep in range(e.get_num_experiments())]
            else:
                result_maps_list = [self.iterator.result_map(dataset, self.repetition, self.run)]

            baseline_scores = []
            for result_map in result_maps_list:
                if baseline_strategy not in result_map or result_map[baseline_strategy] is None:
                    continue
                baseline_scores.append(metric.evaluate(result_map[baseline_strategy]))

            if not baseline_scores:
                print(f"Warning: No baseline data for dataset {dataset}. Skipping.")
                continue

            baseline_agg_score = self.agg.aggregate([baseline_scores])[0]

            for strat in e.get_strategies():
                if strat == baseline_strategy or strat in self.exclude_strategies:
                    continue

                strat_scores = []
                for result_map in result_maps_list:
                    if result_map[strat] is None:
                        continue
                    strat_scores.append(metric.evaluate(result_map[strat]) / baseline_agg_score)

                if strat_scores:
                    normalized_scores[strat].extend(strat_scores)

        aggregated_scores = {
            strat: self.agg.aggregate([scores])[0] for strat, scores in normalized_scores.items()
        }

        headers = ["Strategy", "Normalized Similarity Score"]
        table_rows = [[strategy, aggregated_scores[strategy]] for strategy in e.get_strategies() if strategy != baseline_strategy and strategy not in self.exclude_strategies and strategy in aggregated_scores]

        print(f"\nRepetition: {self.repetition}")
        print(f"Run: {self.run}")

        print(tabulate(table_rows, headers=headers, floatfmt=".4f", tablefmt="fancy_grid"))

class ResultView:
    def __init__(self,  experiment : Union['Experiment', str], run: int = 0, repetition: int = -1, dataset : str = None, agg:'Aggregator' = AverageAgg()):
        self.iterator = ResultsIterator(experiment)
        self.dataset = dataset
        self.repetition = repetition
        self.run = run
        self.agg = agg

    def print_statistics(self, metric: 'EvaluationMetric'):
        e = self.iterator.get_experiment()
        if self.dataset is None:
            datasets = e.get_datasets()
        else:
            datasets = [self.dataset]
        for d in datasets:
            self._p_print_statistics(metric,d,self.repetition)

    def _p_print_statistics(self, metric: 'EvaluationMetric', dataset: str, repetition: int) -> None:
        """
        Pretty print the similarity statistics for each strategy within each dataset.
        """
        from collections import defaultdict

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

        for strat in e.get_strategies():
            scores = []
            runtimes = defaultdict(list)

            for result_map in result_maps_list:
                if result_map[strat] is None:
                    continue

                scores.append(metric.evaluate(result_map[strat]))

                for key, timing_struct in result_map[strat].stopwatch.timers.items():
                    runtimes[key+" runtime"].append(timing_struct.elapsed)
                    all_stopwatch_keys.add(key+" runtime")

            if scores:
                score_map[strat] = self.agg.aggregate([scores])[0]
                for key, times in runtimes.items():
                    runtime_map[strat][key] = self.agg.aggregate([times])[0]

        headers = ["Strategy", "Similarity Score"] + sorted(all_stopwatch_keys)

        for strategy in e.get_strategies():
            if strategy not in score_map:
                continue

            row = [strategy, score_map[strategy]]

            for key in sorted(all_stopwatch_keys):
                row.append(runtime_map[strategy].get(key, ""))

            table_rows.append(row)

        print(f"\nDataset: {dataset}")
        print(f"Repetition: {repetition}")
        print(f"Run: {self.run}")

        print(tabulate(table_rows, headers=headers, floatfmt=".4f", tablefmt="fancy_grid"))
