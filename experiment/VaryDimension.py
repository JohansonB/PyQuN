from typing import List, Tuple

from RaQuN_Lab.experiment.Experiment import IndependentExperiment
from RaQuN_Lab.strategies.Strategy import Strategy


class VaryDimension(IndependentExperiment):

    def __init__(self, num_runs, name : str, num_experiments: int, strategies : List['Strategy'] = [], datasets : List[str] = []):
        self.__runs = num_runs
        super().__init__(name, num_experiments, strategies, datasets)

    def num_runs(self) -> int:
        return self.__runs

    def setup_experiment(self, index: int, ze_input: 'ModelSet', strategy: 'Strategy') -> Tuple['ModelSet', 'Strategy']:
        num_models = int(len(ze_input)*index/self.__runs)
        if num_models < 2:
            num_models = 2
        return ze_input.get_subset(num_models), strategy

    def index_set(self):
        return [((index+1)/self.__runs) for index in range(self.num_runs())]

    def index_name(self):
        return "relative dimension"

