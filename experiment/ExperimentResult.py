from RaQuN_Lab.utils.Utils import store_obj, load_obj


class ExperimentResult:
    def __init__(self, experiment_count: int, run_count: int, strategy: 'Strategy', datamodel: 'DataModel',
                 og_strategy: str, dataset_name: str, experiment_name: str, stopwatch: 'Stopwatch', match: 'Matching', error: float = None, store_datamodel:bool = False, store_strategy:bool = False):

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
        self.store_datamodel = store_datamodel
        self.store_strategy = store_strategy


    def store(self):
        from RaQuN_Lab.experiment.ExperimentManager import ExperimentManager
        if not self.store_datamodel:
            self.datamodel = None
        if not self.store_strategy:
            self.strategy = None
        path = str(ExperimentManager.get_results_dir(self.experiment_name, self.og_strategy, self.dataset_name))\
               +'/'+str(self.experiment_count)+'/'+str(self.run_count)
        store_obj(self,path)

    @staticmethod
    def load(experiment: str, strategy: str, dataset: str, experiment_count: int, run_count: int) -> 'ExperimentResult':
        from RaQuN_Lab.experiment.ExperimentManager import ExperimentManager
        path = str(ExperimentManager.get_results_dir(experiment, strategy, dataset)) \
               + '/' + str(experiment_count) + '/' + str(run_count)
        return load_obj(path)

    def get_match(self) -> 'Matching':
        return self.match

    def set_error(self, error) -> None:
        self.error = error

    def get_error(self) -> float:
        return self.error

