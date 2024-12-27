from typing import List, Any, Tuple

from RaQuN_Lab.experiment.Experiment import IndependentExperiment
from RaQuN_Lab.strategies.Strategy import Strategy
from RaQuN_Lab.utils.Utils import set_field


class VaryParameter(IndependentExperiment):
    def __init__(
        self,
        parameter_name: str,
        parameter_values: List[Any],
        name: str,
        num_experiments: int,
        strategies: List[str] = [],
        datasets: List[str] = []
    ):
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.__runs = len(parameter_values)
        super().__init__(name, num_experiments, strategies, datasets)

    def num_runs(self) -> int:
        return self.__runs

    def setup_experiment(self, index: int, ze_input: 'DataModel', strategy: 'Strategy') -> Tuple['DataModel', Strategy]:

        strategy = Strategy.load(strategy.get_name())
        set_field(strategy, self.parameter_name, self.parameter_values[index])
        return ze_input, strategy

    def index_set(self) -> List[Any]:
        return [repr(self.parameter_values)]

    def index_name(self) -> str:
        return self.parameter_name
