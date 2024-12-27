from typing import Iterable, List

from RaQuN_Lab.experiment.ExperimentResult import ExperimentResult


class ExperimentResults:
    def __init__(self, experiment_results: Iterable['ExperimentResult'] = None):
        if experiment_results is not None:
            self.results = list(experiment_results)
        else:
            self.results = []

    def evaluate_metric(self, metric: 'EvaluationMetric') -> List[float]:
        return [metric.evaluate(res) for res in self.results]


    def add_result(self, result: ExperimentResult) -> None:
        self.results.append(result)

    def store(self) -> None:
        for r in self.results:
            r.store()

    @staticmethod
    def load(experiment: str, strategy: str, dataset: str, experiment_count: int) -> 'ExperimentResults':
        from RaQuN_Lab.experiment.ExperimentManager import ExperimentManager
        ret = ExperimentResults()
        ze_dir = ExperimentManager.get_results_dir(experiment, strategy, dataset)
        if not ze_dir:
            return ExperimentResults()
        ze_dir /= str(experiment_count)
        for f in ze_dir.iterdir():
            ret.add_result(ExperimentResult.load(experiment, strategy, dataset, experiment_count, int(f.stem)))
        return ret

    def __iter__(self):
        return self.results.__iter__()
