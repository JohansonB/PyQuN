from collections import Iterator
from typing import Union, Tuple, Dict, List

from RaQuN_Lab.experiment.Experiment import Experiment
from RaQuN_Lab.experiment.ExperimentResult import ExperimentResult
from RaQuN_Lab.experiment.ExperimentResults import ExperimentResults


class ResultsIterator(Iterator):
    def __init__(self, experiment: Union['Experiment', str], memo_size=float("inf")):
        if isinstance(experiment, str):
            experiment = Experiment.load(experiment)

        self.experiment = experiment
        self.datasets = experiment.get_datasets()
        self.strategies = experiment.get_strategies()
        self.num_experiments = experiment.get_num_experiments()
        self.memo_size = memo_size
        self._cache = {}

        self.items = [
            (dataset, strategy, i)
            for dataset in self.datasets
            for strategy in self.strategies
            for i in range(self.num_experiments)
        ]
        self.index = 0

    def __iter__(self):
        self.index = 0
        self._cache.clear()
        return self

    def __next__(self) -> Tuple:
        if self.index >= len(self.items):
            raise StopIteration

        dataset, strategy, i = self.items[self.index]
        self.index += 1

        return dataset, strategy, i, self.get_result(dataset, strategy, i)

    def result_map(self, dataset: str, repetition: int, run: int) -> Dict[str, 'ExperimentResult']:
        results_map = {}
        for s in self.experiment.get_strategies():
            try:
                results_map[s] = ExperimentResult.load(self.experiment.get_name(),s,dataset,repetition, run)
            except FileNotFoundError as e:
                results_map[s] = None

        return results_map


    def get_result(self, dataset, strategy, i) -> 'ExperimentResults':
        if (dataset, strategy, i) not in self._cache:
            result = ExperimentResults.load(self.experiment.get_name(), strategy, dataset, i)
            if len(self._cache) < self.memo_size:
                self._cache[(dataset, strategy, i)] = result
        else:
            result = self._cache[(dataset, strategy, i)]

        return result

    def to_dict(self) -> Dict[str, Dict[str, List[ExperimentResults]]]:
        ret = {}
        for dataset, strategy, i, res in self:
            if dataset not in ret:
                ret[dataset] = {}

            if strategy not in ret[dataset]:
                ret[dataset][strategy] = []

            ret[dataset][strategy].append(self.get_result(dataset,strategy, i))

        return ret

    def evaluate_metric(self, metric: 'EvaluationMetric') -> None:
        for dataset, strategy, i, res in self:
            res.evaluate_metric(metric)

    def runtime_matrix(self) -> Dict[str, Dict[str, List[List[float]]]]:
        ret = {}
        for dataset, strategy, i, res in self:
            if dataset not in ret:
                ret[dataset] = {}

            if strategy not in ret[dataset]:
                ret[dataset][strategy] = []

            while len(ret[dataset][strategy]) <= i:
                ret[dataset][strategy].append([])

            ret[dataset][strategy][i] = [cur.stopwatch.get_time("matching") for cur in res.results]

        return ret

    def to_error_matrix(self, metric: 'EvaluationMetric') -> Dict[str, Dict[str, List[List[float]]]]:
        ret = {}
        for dataset, strategy, i, res in self:
            if dataset not in ret:
                ret[dataset] = {}

            if strategy not in ret[dataset]:
                ret[dataset][strategy] = []

            while len(ret[dataset][strategy]) <= i:
                ret[dataset][strategy].append([])

            ret[dataset][strategy][i] = res.evaluate_metric(metric)

        return ret

    def get_experiment(self):
        return self.experiment

