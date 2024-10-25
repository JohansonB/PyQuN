import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from types import Union
from typing import Tuple, List, Iterable

from PyQuN_Lab.Stopwatch import Stopwatch
from PyQuN_Lab.DataLoading import DataLoader
from PyQuN_Lab.DataModel import DataModel, Matching, ModelSet
from PyQuN_Lab.Strategy import Strategy
from PyQuN_Lab.Utils import store_obj, load_obj
import threading
from queue import Queue
import time
from pathlib import Path
import os

class ExperimentResult:
    def __init__(self, experiment_count: int, run_count: int, strategy: Strategy, datamodel: DataModel,
                 og_strategy: str, dataset_name: str, experiment_name: str, stopwatch: Stopwatch, match: Matching):
        self.stopwatch = stopwatch
        self.experiment_count = experiment_count
        self.run_count = run_count
        self.strategy = strategy
        self.datamodel = datamodel
        self.og_strategy = og_strategy
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.match = match

    def store(self):
        path = str(ExperimentManager.get_results_dir(self.experiment_name, self.og_strategy, self.dataset_name))\
               +'/'+str(self.experiment_count)+'/'+str(self.run_count)
        store_obj(self,path)

    @staticmethod
    def load(experiment: str, strategy: str, dataset: str, experiment_count: int, run_count: int) -> 'ExperimentResult':
        path = str(ExperimentManager.get_results_dir(experiment, strategy, dataset)) \
               + '/' + str(experiment_count) + '/' + str(run_count)
        return load_obj(path)


class ExperimentResults:
    def __init__(self, experiment_results: Iterable[ExperimentResult] = None):
        if experiment_results is not None:
            self.results = list(experiment_results)
        else:
            self.results = []

    def add_result(self, result: ExperimentResult) -> None:
        self.results.append(result)

    def store(self) -> None:
        for r in self.results:
            r.store()

    @staticmethod
    def load(experiment: str, strategy: str, dataset: str, experiment_count: int) -> 'ExperimentResults':
        ret = ExperimentResults()
        ze_dir = ExperimentManager.get_results_dir(experiment, strategy, dataset)
        if not ze_dir:
            return ExperimentResults()
        ze_dir /= str(experiment_count)
        for f in ze_dir.iterdir():
            ret.add_result(ExperimentResult.load(experiment, strategy, dataset, experiment_count, int(f.name)))
        return ret


class Experiment(ABC):
    def __init__(self, name : str, num_experiments: int, strategies : List[str] = [], datasets : List[str] = []):
        self.num_experiments = num_experiments
        self.strategies = strategies
        self.datasets = datasets
        self.name = name

    def store(self) -> None:
        store_obj("MetaData/Experiment/" + str(self.get_name()))

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

    @abstractmethod
    def is_sequential(self) -> bool:
        pass

    def run_instance(self, index: Union[int,ExperimentResults], ze_input: DataModel, strategy: Strategy, dataset: str, experiment_count : int) -> ExperimentResult:
        inp, strat = self.setup_experiment(index, ze_input, strategy)
        match, stopwatch = strat.timed_match(inp)
        if isinstance(index,ExperimentResults):
            index = len(index)
        return ExperimentResult(match=match, stopwatch=stopwatch, datamodel=inp, strategy=strat,experiment_count=experiment_count
                                , dataset_name=dataset,run_count=index,experiment_name=self.name,og_strategy=strategy.get_name())

    def setup_experiment(self, state : Union[ExperimentResults, int], ze_input : DataModel, strategy : Strategy) ->Tuple[DataModel, Strategy]:
        pass

    def get_name(self) -> str:
        return self.name

    def get_strategies(self) -> List[str]:
        self.strategies

    def get_datasets(self) -> List[str]:
        self.datasets


class DependantExperiment(Experiment, ABC):
    def __init__(self,name : str, num_experiments: int, strategies : List[Strategy] = [], datasets : List[str] = []):
        super().__init__(name, num_experiments, strategies, datasets)
    # returns the input for the experimental run
    @abstractmethod
    def setup_experiment(self, prev: ExperimentResults, ze_input: DataModel, strategy: Strategy) -> Tuple[DataModel, Strategy]:
        pass

    def is_sequential(self)-> bool:
        return True


class IndependentExperiment(Experiment, ABC):
    def __init__(self, name : str, num_experiments: int, strategies : List[Strategy] = [], datasets : List[str] = []):
        super().__init__(name,num_experiments,strategies,datasets)
    #returns the input for the experimental run
    @abstractmethod
    def setup_experiment(self, index: int, ze_input: DataModel, strategy: Strategy) -> Tuple[DataModel, Strategy]:
        pass

    def is_sequential(self) -> bool:
        return False



class DoMatching(IndependentExperiment):
    def __init__(self, name : str, num_experiments: int, strategies : List[Strategy] = [], datasets : List[str] = []):
        super().__init__(name,num_experiments, strategies, datasets)

    def num_runs(self) -> int:
        return 1

    def setup_experiment(self, index: int, ze_input: DataModel, strategy: Strategy) -> Tuple[DataModel, Strategy]:
        return ze_input, strategy


class VaryDimension(IndependentExperiment):

    def __init__(self, num_runs, name : str, num_experiments: int, strategies : List[Strategy] = [], datasets : List[str] = []):
        self.num_runs = num_runs
        super().__init__(name, num_experiments, strategies, datasets)

    def num_runs(self) -> int:
        return self.num_runs

    def setup_experiment(self, index: int, ze_input: ModelSet, strategy: Strategy) -> Tuple[ModelSet, Strategy]:
        num_models = int(len(ze_input)*index/self.num_runs)
        if num_models < 2:
            num_models = 2
        ze_input.shuffle_models()
        return ze_input.get_subset(num_models), strategy

class VarySize(IndependentExperiment):
    def __init__(self, init_length: int, num_runs: int, name : str, num_experiments: int, strategies : List[Strategy] = [], datasets : List[str] = []):
        self.init_length = init_length
        self.runs = num_runs
        super().__init__(name, num_experiments, strategies, datasets)

    def num_runs(self) -> int:
        return self.runs

    def setup_experiment(self, index: int, ze_input: ModelSet, strategy: Strategy) -> Tuple[DataModel, Strategy]:
        cur_factor = self.init_length + index/self.num_runs()*(1-self.init_length)
        ze_input.shuffle_elements()
        ze_input.shuffle_models()
        return ze_input.shorten(cur_factor), strategy





#goto make dictionaries thread safe

class ExperimentEnv:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.datasets = {}
        self.loader_map = {}
        self.strategies = {}

    def run_experiment(self,index: Union[int,ExperimentResults], strat: str, dataset: str, experiment_count: int) -> ExperimentResult:
        if strat not in self.strategies:
            self.strategies[strat] = Strategy.load(strat)
            self.loader_map[strat] = {}
        ze_strat = self.strategies[strat]

        if dataset not in self.datasets:
            self.datasets[dataset] = ExperimentManager.load_dataset_path(dataset)
        path = self.datasets[dataset]

        if dataset not in self.loader_map[strat]:
            self.loader_map[strat][dataset] = self.strategies[strat].get_data_loader(path).read_file(path).get_data_model()
        ze_data = self.loader_map[strat][dataset]

        return self.experiment.run_instance(self, index, ze_data, ze_strat, dataset, experiment_count)

class ExperimentConfig:
    def __init__(self, index: Union[int,ExperimentResults], strat: str, dataset: str, experiment_count: int, experiment: str):
        self.index = index
        self.strat = strat
        self.dataset = dataset,
        self.experiment_count = experiment_count




class ExperimentManager:
    def __init__(self):
        self.experiment_envs = {}

    def get_unfinished_experiments(self, experiment: str, strategy: str, dataset: str) -> List[ExperimentConfig]:
        ze_dir = self.get_results_dir(experiment, strategy, dataset)
        ret = []

        experiment = Experiment.load(experiment)
        num_exp = experiment.get_num_experiments()
        num_runs = experiment.num_runs()
        max_eles = {}

        run_arr = {}
        if ze_dir.exists():
            for i in range(num_exp):
                run_arr[i] = [False] * num_runs
            for f in ze_dir.iterdir():
                f_int = int(str(f.name))
                for ff in f.iterdir():
                    ff_stem = ff.stem
                    ff_int = int(str(ff_stem))
                    if (f_int not in max_eles) or ff_int > max_eles[f_int]:
                        max_eles[f_int] = ff_int
                    run_arr[f_int][ff_int] = True
        else:
            for i in range(num_exp):
                run_arr[i] = [False]*num_runs

        if experiment.is_sequential():
            for exp_id, maxi in max_eles.items():
                if maxi < num_runs:
                    if exp_id not in max_eles:
                        past = None
                    else:
                        past = ExperimentResults.load(experiment,strategy,dataset)
                    ret.append(ExperimentConfig(past,strategy,dataset,exp_id, experiment))

        else:
            for exp_id, run_flags in run_arr.items():
                for i in range(len(run_flags)):
                    if not run_flags[i]:
                        ret.append(ExperimentConfig(i, strategy, dataset, exp_id, experiment))

        return ret

    def run_sequential_experiment(self, job, experiment, executor):
        """
        Wrapper function to run an experiment sequentially.
        After running the experiment, it checks for the next job and submits it.
        """
        # Run the experiment
        result = self.experiment_envs[experiment.get_name()].run_experiment(
            job.index, job.strat, job.dataset, job.experiment_count
        )

        # Find next unfinished job (you can define the logic for getting the next job)
        unfinished_jobs = self.get_unfinished_experiments(experiment.get_name(), job.strat, job.dataset)
        next_job = None
        for j in unfinished_jobs:
            if j.index > job.index:  # Find the next job after the current one
                next_job = j
                break

        # If there is a next job, submit it to the executor
        if len(job.index)<experiment.num_runs():
            job.index = ExperimentResults.load(experiment, job.strat, job.dataset)
            executor.submit(self.run_sequential_experiment, job, experiment, executor)

    def run_unfinished_experiments(self, executor: ThreadPoolExecutor):
        path =  Path('MetaData/Experiment/')
        for f in path.iterdir():
            experiment = Experiment.load(f)
            if experiment.get_name() not in self.experiment_envs:
                self.experiment_envs[experiment.get_name()] = ExperimentEnv(experiment)
            for s in experiment.get_strategies():
                for d in experiment.get_datasets():
                    jobs = self.get_unfinished_experiments(experiment.name,s,d)
                    for j in jobs:
                        if not experiment.is_sequential():
                            executor.submit(
                                self.experiment_envs[experiment.get_name()].run_experiment,
                                j.index,
                                j.strat,
                                j.dataset,
                                j.experiment_count
                            )
                        else:
                            self.run_sequential_experiment(j,experiment,executor)







    @staticmethod
    def get_results_dir(experiment: str, strategy: str, dataset: str) -> Path:
        return Path('MetaData/Results/'+experiment+'/'+strategy+'/'+dataset)

    def get_results(self, experiment: str, strategy: str, dataset: str) -> ExperimentResults:
        dir = self.get_results_dir(experiment,strategy,dataset)



    def add_strategy(self, experiment: Union[str, Experiment], strategy: Union[Strategy, str]) -> None:
        experiment_name = self.add_experiment(experiment)
        if isinstance(strategy, Strategy):
            if not self.is_stored_strategy(strategy):
                strategy.store()
            strategy_name = strategy.get_name()
        elif isinstance(strategy, str):
            if not self.is_stored_strategy(strategy):
                raise Exception("no Strategy with that name exists")
            strategy_name = strategy
        
        if isinstance(experiment, str):
            experiment = Experiment.load(experiment)
        
        experiment.add_strategy(strategy_name)
        experiment.store()


    def add_dataset(self, experiment: Union[str, Experiment], name: str, path: str = None) -> None:
        experiment_name = self.add_experiment(experiment)
        store_obj(path, 'MetaData/DatasetPaths/'+name)
        if isinstance(experiment, str):
            experiment = Experiment.load(experiment_name)
        experiment.add_dataset(name)
        experiment.store()

    def load_dataset_path(self, dataset :str) -> str:
        return load_obj('MetaData/DatasetPaths/'+dataset)

    #checks if the experiment allready exists if only the name is given,
    #If the experiment is given by object it is stored in case it hasn't been yet
    #returns the name of the experiment to unify the representation
    def add_experiment(self, experiment: Union[str, Experiment]) -> str:
        if isinstance(experiment, Experiment):
            if not self.is_stored_experiment(experiment.get_name()):
                experiment.store()
            experiment_name = experiment.get_name()
        else:
            if not self.is_stored_experiment(experiment):
                raise Exception("no Experiment with that name exists")
            experiment_name = experiment
        return experiment_name

    def set_data_loader(self, strategy: str, file_ending: str, data_loader: DataLoader) -> None:
        if not self.is_stored_strategy(strategy):
            raise Exception("No strategy with the provided name exists")
        s = Strategy.load("MetaData/Strategies/" + strategy)
        s.set_data_loader(file_ending,data_loader)
        s.store()



    @staticmethod
    def is_stored_strategy(strategy: Union[Strategy,str]) -> bool:
        if isinstance(strategy,Strategy):
            strategy = strategy.get_name()
        return ExperimentManager.dir_contains_name('MetaData/Strategies/', strategy)

    @staticmethod
    def dir_contains_name(dir: str, name: str) -> bool:
        directory_path = dir
        for filename in os.listdir(directory_path):
            if os.path.isfile(os.path.join(directory_path, filename)) and name == filename[:-4]:
                return True
        return False

    @staticmethod
    def is_stored_experiment(experiment: Union[str, Experiment]) -> bool:
        if isinstance(experiment, Experiment):
            experiment = experiment.get_name()
        return ExperimentManager.dir_contains_name('MetaData/Experiments/', experiment)



class JobScout(threading.Thread):
    _instance = None
    _lock = threading.lock()
    _job_queue = None

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, check_interval=5):
        self.check_interval = check_interval
        self.stop_event = threading.Event()
        self._job_queue = self.init_queue()

    def job_search(self):
        pass

    def init_queue(self):
        pass


    def run(self):
        pass
