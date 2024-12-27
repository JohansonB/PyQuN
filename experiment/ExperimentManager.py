import os
import shutil
import time
from pathlib import Path
from typing import Union, List, Iterable

from RaQuN_Lab.experiment.Experiment import Experiment
from RaQuN_Lab.experiment.ExperimentResults import ExperimentResults
from RaQuN_Lab.strategies.Strategy import Strategy
from RaQuN_Lab.utils.concurrency_tools import ThreadSafeHashMap


class ExperimentEnv:
    def __init__(self):

        self.loader_map = ThreadSafeHashMap()
        self.strategies = ThreadSafeHashMap()

    def run_experiment(self, experiment: 'Experiment', index: Union[int, 'ExperimentResults'], strat: str, dataset: str,
                       experiment_count: int) -> 'ExperimentResult':
        try:
            ze_strat = Strategy.load(strat)
            path = ExperimentManager.get_dataset_path(dataset)
            ze_data = self.loader_map.setdefault((dataset, ze_strat.get_data_loader(path)),
                                                 lambda t: t[1].read_file(path).get_data_model())

            result = experiment.run_instance(index, ze_data, ze_strat, dataset, experiment_count)
            result.store()
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()


class ExperimentConfig:
    def __init__(self, index: Union[int, 'ExperimentResults'], strat: str, dataset: str, experiment_count: int,
                 experiment: str):
        self.index = index
        self.strat = strat
        self.dataset = dataset
        self.experiment_count = experiment_count
        self.experiment = experiment


class ExperimentManager:
    __experiments_env = ExperimentEnv()

    @staticmethod
    def get_unfinished_experiments(experiment: Union['Experiment', str], strategy: str, dataset: str) -> List[
        ExperimentConfig]:
        ret = []

        if isinstance(experiment, str):
            experiment_name = experiment
            experiment = Experiment.load(experiment)
        elif isinstance(experiment, Experiment):
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
                run_arr[i] = [False] * runs

        if experiment.is_sequential():
            for exp_id, maxi in max_eles.items():
                if maxi < runs:
                    if exp_id not in max_eles:
                        past = None
                    else:
                        past = ExperimentResults.load(experiment, strategy, dataset)
                    ret.append(ExperimentConfig(past, strategy, dataset, exp_id, experiment))

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
        result = ExperimentManager.__experiments_env.run_experiment(
            experiment, job.index, job.strat, job.dataset, job.experiment_count
        )

        if len(job.index) < experiment.num_runs():
            job.index = ExperimentResults.load(experiment, job.strat, job.dataset)
            executor.submit(ExperimentManager.run_sequential_experiment, job, experiment, executor)

    @staticmethod
    def run_unfinished_experiments(executor: 'ThreadPoolExecutor', excluded_experiments: List[str] = [],
                                   excluded_strategies: List[str] = []):
        path = Path('MetaData/Experiment/')
        for f in path.iterdir():
            experiment = Experiment.load(f.stem)
            if experiment.get_name() in excluded_experiments:
                continue

            for s in experiment.get_strategies():
                if s in excluded_strategies:
                    continue
                for d in experiment.get_datasets():
                    jobs = ExperimentManager.get_unfinished_experiments(experiment, s, d)
                    for j in jobs:
                        if not experiment.is_sequential():
                            executor.submit(
                                ExperimentManager.__experiments_env.run_experiment,
                                experiment,
                                j.index,
                                j.strat,
                                j.dataset,
                                j.experiment_count
                            )
                        else:
                            executor.submit(ExperimentManager.run_sequential_experiment, j, experiment, executor)

    @staticmethod
    def get_results_dir(experiment: str, strategy: str, dataset: str) -> Path:
        return Path('MetaData/Results/' + experiment + '/' + strategy + '/' + dataset)

    @staticmethod
    def add_strategies(experiment: Union[str, Experiment], strategies: Iterable[Union[Strategy, str]]):
        for s in strategies:
            ExperimentManager.add_strategy(experiment, s)

    @staticmethod
    def add_strategy(experiment: Union[str, Experiment], strategy: Union[Strategy, str]) -> None:
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
        base_path = 'Metadata/Datasets'

        for filename in os.listdir(base_path):
            name, ext = os.path.splitext(filename)

            if name == dataset:
                return base_path + '/' + filename

        return None

    @staticmethod
    def add_datasets(names: Iterable[str], experiment: Union[str, Experiment] = None,
                     paths: Iterable[str] = None) -> None:
        if paths is None:
            paths = [None] * len(names)

        for n, p in zip(names, paths):
            ExperimentManager.add_dataset(n, experiment, p)

    @staticmethod
    def add_dataset(name: str, experiment: Union[str, Experiment] = None, path: str = None) -> None:

        if path is not None:
            _, ext = os.path.splitext(path)
            target = os.path.join("Metadata/Datasets", name + ext)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(path, target)
        if experiment is not None:
            experiment_name = ExperimentManager.add_experiment(experiment)
            if isinstance(experiment, str):
                experiment = Experiment.load(experiment_name)
            experiment.add_dataset(name)
            experiment.store()

    # checks if the experiment allready exists if only the name is given,
    # If the experiment is given by object it is stored in case it hasn't been yet
    # returns the name of the experiment to unify the representation
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
    def set_data_loader(strategy: str, file_ending: str, data_loader: 'DataLoader') -> None:
        if not ExperimentManager.is_stored_strategy(strategy):
            raise Exception("No strategy with the provided name exists")
        s = Strategy.load("MetaData/Strategies/" + strategy)
        s.set_data_loader(file_ending, data_loader)
        s.store()

    @staticmethod
    def is_stored_strategy(strategy: Union[Strategy, str]) -> bool:
        if isinstance(strategy, Strategy):
            strategy = strategy.get_name()
        return ExperimentManager.dir_contains_name('MetaData/Strategies/', strategy)

    @staticmethod
    def dir_contains_name(dir: str, name: str) -> bool:
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


