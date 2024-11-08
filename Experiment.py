import copy
import threading
import time
from abc import ABC, abstractmethod
from collections import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Dict, Any
from typing import Tuple, List, Iterable


from PyQuN_Lab.Stopwatch import Stopwatch
from PyQuN_Lab.DataLoading import DataLoader
from PyQuN_Lab.DataModel import DataModel, Matching, ModelSet
from PyQuN_Lab.Strategy import Strategy
from PyQuN_Lab.Utils import store_obj, load_obj
from pathlib import Path
from PyQuN_Lab.concurrency_tools import ReadWriteLock
import os
import shutil

class ExperimentResult:
    def __init__(self, experiment_count: int, run_count: int, strategy: Strategy, datamodel: DataModel,
                 og_strategy: str, dataset_name: str, experiment_name: str, stopwatch: Stopwatch, match: Matching, error: float = None ):
        self.stopwatch = stopwatch
        self.experiment_count = experiment_count
        self.run_count = run_count
        self.strategy = strategy
        self.datamodel = datamodel
        self.og_strategy = og_strategy
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.match = match
        self.error = error


    def store(self):
        path = str(ExperimentManager.get_results_dir(self.experiment_name, self.og_strategy, self.dataset_name))\
               +'/'+str(self.experiment_count)+'/'+str(self.run_count)
        store_obj(self,path)

    @staticmethod
    def load(experiment: str, strategy: str, dataset: str, experiment_count: int, run_count: int) -> 'ExperimentResult':
        path = str(ExperimentManager.get_results_dir(experiment, strategy, dataset)) \
               + '/' + str(experiment_count) + '/' + str(run_count)
        return load_obj(path)

    def get_match(self) -> Matching:
        return self.match

    def set_error(self, error) -> None:
        self.error = error

    def get_error(self) -> float:
        return self.error

class ExperimentResults:
    def __init__(self, experiment_results: Iterable[ExperimentResult] = None):
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

        # Create a flat list of all the (dataset, strategy, experiment index) tuples
        self.items = [
            (dataset, strategy, i)
            for dataset in self.datasets
            for strategy in self.strategies
            for i in range(self.num_experiments)
        ]
        self.index = 0

    def __iter__(self):
        # Every time we start a new iteration, we reset the index and cache
        self.index = 0
        self._cache.clear()  # Reset the cache for each iteration
        return self

    def __next__(self) -> Tuple:
        if self.index >= len(self.items):
            raise StopIteration

        dataset, strategy, i = self.items[self.index]
        self.index += 1



        return dataset, strategy, i, self.get_result(dataset, strategy, i)


    def get_result(self, dataset, strategy, i) -> ExperimentResults:
        if (dataset, strategy, i) not in self._cache:
            # Load the ExperimentResults for the current (dataset, strategy, index) combination
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

    def to_error_matrix(self, metric: 'EvaluationMetric') -> Dict[str, Dict[str, List[List[float]]]]:
        ret = {}
        for dataset, strategy, i, res in self:
            if dataset not in ret:
                ret[dataset] = {}

            if strategy not in ret[dataset]:
                ret[dataset][strategy] = []

            # Ensure the list is large enough to hold the value at index `i`
            while len(ret[dataset][strategy]) <= i:
                ret[dataset][strategy].append([])  # Append an empty list if needed

            # Append the errors at index `i` for this dataset-strategy combination
            ret[dataset][strategy][i] = res.evaluate_metric(metric)

        return ret


class Experiment(ABC):

    def __init__(self, name : str, num_experiments: int, strategies : List[str] = [], datasets : List[str] = []):
        self.num_experiments = num_experiments
        self.strategies = strategies
        self.datasets = datasets
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
        return self.strategies

    def get_datasets(self) -> List[str]:
        return self.datasets


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
        return ze_input, Strategy.load(strategy.get_name())


class VaryDimension(IndependentExperiment):

    def __init__(self, num_runs, name : str, num_experiments: int, strategies : List[Strategy] = [], datasets : List[str] = []):
        self.__runs = num_runs
        super().__init__(name, num_experiments, strategies, datasets)

    def num_runs(self) -> int:
        return self.__runs

    def setup_experiment(self, index: int, ze_input: ModelSet, strategy: Strategy) -> Tuple[ModelSet, Strategy]:
        num_models = int(len(ze_input)*index/self.__runs)
        if num_models < 2:
            num_models = 2
        ze_input.shuffle_models()
        return ze_input.get_subset(num_models), Strategy.load(strategy.get_name())

    def index_set(self):
        return [((index+1)/self.__runs) for index in range(self.num_runs())]

    def index_name(self):
        return "relative dimension"


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
        return ze_input.shorten(cur_factor), Strategy.load(strategy.get_name())

    def index_set(self):
        return [self.init_length + index/self.num_runs()*(1-self.init_length) for index in range(self.num_runs())]

    def index_name(self):
        return "relative length"








#goto make dictionaries thread safe

class ExperimentEnv:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

        self.loader_map_lock = ReadWriteLock()
        self.loader_map = {}
        self.strategies_lock = ReadWriteLock()
        self.strategies = {}

    def _access_and_fill_dict(self, lock, dictionary: Dict[str, Any], key: str, load_func) -> Any:
        """
        Access a dictionary with read-write locks.
        If the key is not present, load the value using load_func.
        Returns the value from the dictionary.
        """
        # Try to read the value first
        lock.read_lock()
        try:
            if key in dictionary:
                return dictionary[key]
        finally:
            lock.read_release()

        # If not found, acquire write lock to fill the dictionary
        lock.write_lock()
        try:
            # Check again after acquiring write lock
            if key not in dictionary:
                dictionary[key] = load_func(key)
            return dictionary[key]
        finally:
            lock.write_release()

    def run_experiment(self, index: Union[int, ExperimentResults], strat: str, dataset: str,
                       experiment_count: int) -> ExperimentResult:
        try:
            # Load object into strategies dictionary if necessary
            ze_strat = self._access_and_fill_dict(
                self.strategies_lock,
                self.strategies,
                strat,
                lambda s: Strategy.load(s)
            )

            # Load object into loader_map dictionary if necessary
            self._access_and_fill_dict(
                self.loader_map_lock,
                self.loader_map,
                strat,
                lambda d: {}
            )
            path = ExperimentManager.get_dataset_path(dataset)
            loader = ze_strat.get_data_loader(path)
            ze_data = self._access_and_fill_dict(
                self.loader_map_lock,
                self.loader_map[strat],
                dataset,
                lambda d: ze_strat.get_data_loader(path).read_file(path).get_data_model()
            )

            # Run the experiment instance
            result = self.experiment.run_instance(index, ze_data, ze_strat, dataset, experiment_count)
            result.store()
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()  # This will print the full traceback of the exception




class ExperimentConfig:
    def __init__(self, index: Union[int,ExperimentResults], strat: str, dataset: str, experiment_count: int, experiment: str):
        self.index = index
        self.strat = strat
        self.dataset = dataset
        self.experiment_count = experiment_count
        self.experiment = experiment




class ExperimentManager:
    __experiment_envs = {}

    @staticmethod
    def get_unfinished_experiments(experiment: Union[Experiment,str], strategy: str, dataset: str) -> List[ExperimentConfig]:
        ret = []

        if isinstance(experiment, str):
            experiment_name = experiment
            experiment = Experiment.load(experiment)
        elif isinstance(experiment,Experiment):
            experiment_name = experiment.get_name()

        ze_dir = ExperimentManager.get_results_dir(experiment_name, strategy, dataset)

        num_exp = experiment.get_num_experiments()
        runs = experiment.num_runs()
        max_eles = {}

        run_arr = {}
        if ze_dir.exists():
            for i in range(num_exp):
                run_arr[i] = [False] * runs
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
                run_arr[i] = [False]*runs

        if experiment.is_sequential():
            for exp_id, maxi in max_eles.items():
                if maxi < runs:
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

    @staticmethod
    def run_sequential_experiment(job, experiment, executor):
        """
        Wrapper function to run an experiment sequentially.
        After running the experiment, it checks for the next job and submits it.
        """
        # Run the experiment
        result = ExperimentManager.__experiment_envs[experiment.get_name()].run_experiment(
            job.index, job.strat, job.dataset, job.experiment_count
        )

        # If there is a next job, submit it to the executor
        if len(job.index) < experiment.num_runs():
            job.index = ExperimentResults.load(experiment, job.strat, job.dataset)
            executor.submit(ExperimentManager.run_sequential_experiment, job, experiment, executor)

    @staticmethod
    def run_unfinished_experiments(executor: ThreadPoolExecutor):
        path =  Path('MetaData/Experiment/')
        for f in path.iterdir():
            experiment = Experiment.load(f.stem)
            if experiment.get_name() not in ExperimentManager.__experiment_envs:
                ExperimentManager.__experiment_envs[experiment.get_name()] = ExperimentEnv(experiment)

            for s in experiment.get_strategies():
                for d in experiment.get_datasets():
                    jobs = ExperimentManager.get_unfinished_experiments(experiment,s,d)
                    for j in jobs:
                        if not experiment.is_sequential():
                            executor.submit(
                                ExperimentManager.__experiment_envs[experiment.get_name()].run_experiment,
                                j.index,
                                j.strat,
                                j.dataset,
                                j.experiment_count
                            )
                        else:
                            executor.submit(ExperimentManager.run_sequential_experiment, j, experiment, executor)







    @staticmethod
    def get_results_dir(experiment: str, strategy: str, dataset: str) -> Path:
        return Path('MetaData/Results/'+experiment+'/'+strategy+'/'+dataset)

    @staticmethod
    def add_strategy( experiment: Union[str, Experiment], strategy: Union[Strategy, str]) -> None:
        experiment_name = ExperimentManager.add_experiment(experiment)
        if isinstance(strategy, Strategy):
            if not ExperimentManager.is_stored_strategy(strategy):
                strategy.store()
            strategy_name = strategy.get_name()
        elif isinstance(strategy, str):
            if not ExperimentManager.is_stored_strategy(strategy):
                raise Exception("no Strategy with that name exists")
            strategy_name = strategy
        
        if isinstance(experiment, str):
            experiment = Experiment.load(experiment)
        
        experiment.add_strategy(strategy_name)
        experiment.store()


    @staticmethod
    def get_dataset_path(dataset):
        # Define the base path to the directory containing the datasets
        base_path = 'Metadata/Datasets'

        # Iterate over all files in the directory
        for filename in os.listdir(base_path):
            # Split the filename and its extension
            name, ext = os.path.splitext(filename)

            # Check if the filename (without extension) matches the dataset name
            if name == dataset:
                # Return the full path to the matched file
                return base_path+'/'+filename

        # Return None if no matching file is found
        return None

    @staticmethod
    def add_dataset(name: str,experiment: Union[str, Experiment] = None, path: str = None) -> None:




        if path is not None:
            # Extract the extension from the provided path
            _, ext = os.path.splitext(path)
            target = os.path.join("Metadata/Datasets", name + ext)
            # Create directories if they do not exist
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(path, target)
        if experiment is not None:
            experiment_name = ExperimentManager.add_experiment(experiment)
            if isinstance(experiment, str):
                experiment = Experiment.load(experiment_name)
            experiment.add_dataset(name)
            experiment.store()



    #checks if the experiment allready exists if only the name is given,
    #If the experiment is given by object it is stored in case it hasn't been yet
    #returns the name of the experiment to unify the representation
    @staticmethod
    def add_experiment(experiment: Union[str, Experiment]) -> str:
        if isinstance(experiment, Experiment):
            if not ExperimentManager.is_stored_experiment(experiment.get_name()):
                experiment.store()
            experiment_name = experiment.get_name()
        else:
            if not ExperimentManager.is_stored_experiment(experiment):
                raise Exception("no Experiment with that name exists")
            experiment_name = experiment
        return experiment_name

    @staticmethod
    def set_data_loader(strategy: str, file_ending: str, data_loader: DataLoader) -> None:
        if not ExperimentManager.is_stored_strategy(strategy):
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
        # Check if the directory exists
        if not os.path.exists(dir):
            return False
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


def monitor(executor, check_interval=1):
    while True:
        time.sleep(check_interval)
        que = executor._work_queue
        if not executor._work_queue.empty():

            print("Work queue is not empty, tasks may be waiting...")
        # Additional checks can be added here





if __name__ == "__main__":
    from PyQuN_Lab.Strategy import RandomMatcher, VanillaRaQuN
    s1 = RandomMatcher("random")
    s2 = VanillaRaQuN("raqun")
    e1 = VaryDimension(5,"vary_dim",5)
    e2 = VarySize(0.7, 5,"vary_len",5)
    ExperimentManager.add_experiment(e1)
    ExperimentManager.add_experiment(e2)
    ExperimentManager.add_strategy(e1,s1)
    ExperimentManager.add_strategy(e1,s2)
    ExperimentManager.add_strategy(e2,s1)
    ExperimentManager.add_strategy(e2,s2)
    ExperimentManager.add_dataset("hosp", path="Data/Apogames.csv")
    ExperimentManager.add_dataset("ppu", path="Data/ppu.csv")
    ExperimentManager.add_dataset("hosp",e1)
    ExperimentManager.add_dataset("ppu",e1)
    ExperimentManager.add_dataset("hosp",e2)
    ExperimentManager.add_dataset("ppu",e2)
    e = Experiment.load("vary_len")
    executor = ThreadPoolExecutor(max_workers=5)
    ExperimentManager.run_unfinished_experiments(executor)
    #result = ExperimentResults.load(e1,s2,"hosp",0)
    print("pepe")








