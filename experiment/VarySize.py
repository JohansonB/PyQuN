from typing import List, Tuple

from RaQuN_Lab.experiment.Experiment import IndependentExperiment
from RaQuN_Lab.strategies.Strategy import Strategy


class VarySize(IndependentExperiment):
    def __init__(self, init_length: int, num_runs: int, name : str, num_experiments: int, strategies : List['Strategy'] = [], datasets : List[str] = []):
        self.init_length = init_length
        self.runs = num_runs
        super().__init__(name, num_experiments, strategies, datasets)

    def num_runs(self) -> int:
        return self.runs

    def setup_experiment(self, index: int, ze_input: 'ModelSet', strategy: 'Strategy') -> Tuple['DataModel', 'Strategy']:
        cur_factor = self.init_length + index/(self.num_runs()-1)*(1-self.init_length)
        return ze_input.shorten(cur_factor), strategy

    def index_set(self):
        return [self.init_length + index/self.num_runs()*(1-self.init_length) for index in range(self.num_runs())]

    def index_name(self):
        return "relative length"