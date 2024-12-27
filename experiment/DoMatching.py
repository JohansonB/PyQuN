from typing import List, Tuple

from RaQuN_Lab.experiment.Experiment import IndependentExperiment
from RaQuN_Lab.strategies.Strategy import Strategy


class DoMatching(IndependentExperiment):
    def __init__(self, name : str, num_experiments: int, strategies : List['Strategy'] = [], datasets : List[str] = []):
        super().__init__(name,num_experiments, strategies, datasets)

    def num_runs(self) -> int:
        return 1

    def setup_experiment(self, index: int, ze_input: 'DataModel', strategy: 'Strategy') -> Tuple['DataModel', 'Strategy']:
        return ze_input, strategy
