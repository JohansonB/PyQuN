from abc import ABC, abstractmethod
from typing import List, Union, Tuple

from RaQuN_Lab.experiment.ExperimentResult import ExperimentResult
from RaQuN_Lab.experiment.ExperimentResults import ExperimentResults
from RaQuN_Lab.utils.Utils import store_obj, load_obj


class Experiment(ABC):

    def __init__(self, name : str, num_experiments: int, strategies : List[str] = None, datasets : List[str] = None):
        self.num_experiments = num_experiments
        self.strategies = list(strategies) if strategies is not None else []
        self.datasets = list(datasets) if datasets is not None else []
        self.name = name

    def store(self) -> None:
        store_obj(self, "MetaData/Experiment/" + str(self.get_name()))

    def get_num_experiments(self):
        return self.num_experiments

    def add_strategy(self, strategy: str) -> None:
        self.strategies.append(strategy)

    def add_dataset(self, dataset: str) -> None:
        self.datasets.append(dataset)

    @staticmethod
    def load(experiment_name: str) -> 'Experiment':
        return load_obj("MetaData/Experiment/" + experiment_name)


    # returns the number of experimental runs performed
    @abstractmethod
    def num_runs(self) -> int:
        pass

    def index_set(self):
        return [i for i in range(self.num_runs())]

    def index_name(self):
        return "run"

    @abstractmethod
    def is_sequential(self) -> bool:
        pass

    def run_instance(self, index: Union[int,ExperimentResults], ze_input: 'DataModel', strategy: 'Strategy', dataset: str, experiment_count : int) -> ExperimentResult:
        inp, strat = self.setup_experiment(index, ze_input, strategy)
        match, stopwatch = strat.timed_match(inp)
        if isinstance(index,ExperimentResults):
            index = len(index)
        return ExperimentResult(match=match, stopwatch=stopwatch, datamodel=inp, strategy=strat,experiment_count=experiment_count
                                , dataset_name=dataset,run_count=index,experiment_name=self.name,og_strategy=strategy.get_name())

    def setup_experiment(self, state : Union[ExperimentResults, int], ze_input : 'DataModel', strategy : 'Strategy') ->Tuple['DataModel', 'Strategy']:
        pass

    def get_name(self) -> str:
        return self.name

    def get_strategies(self) -> List[str]:
        return self.strategies

    def get_datasets(self) -> List[str]:
        return self.datasets


class DependantExperiment(Experiment, ABC):
    def __init__(self,name : str, num_experiments: int, strategies : List['Strategy'] = [], datasets : List[str] = []):
        super().__init__(name, num_experiments, strategies, datasets)
    # returns the input for the experimental run
    @abstractmethod
    def setup_experiment(self, prev: ExperimentResults, ze_input: 'DataModel', strategy: 'Strategy') -> Tuple['DataModel', 'Strategy']:
        pass

    def is_sequential(self)-> bool:
        return True


class IndependentExperiment(Experiment, ABC):
    def __init__(self, name : str, num_experiments: int, strategies : List['Strategy'] = [], datasets : List[str] = []):
        super().__init__(name,num_experiments,strategies,datasets)
    #returns the input for the experimental run
    @abstractmethod
    def setup_experiment(self, index: int, ze_input: 'DataModel', strategy: 'Strategy') -> Tuple['DataModel', 'Strategy']:
        pass

    def is_sequential(self) -> bool:
        return False










