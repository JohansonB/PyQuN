import uuid
from abc import ABC, abstractmethod
from typing import Tuple

from RaQuN_Lab.dataloading.msloaders.CSVLoader import CSVLoader
from RaQuN_Lab.utils.Stopwatch import Stopwatch
from RaQuN_Lab.utils.Utils import store_obj, load_obj


class Strategy(ABC):
    def __init__(self, name: str) -> None:
        self.id = uuid.uuid4()
        self.name = name
        self.data_loaders = {}


    def get_id(self) -> uuid.UUID:
        return self.id


    def set_data_loader(self, file_ending: str, data_loader: 'DataLoader'):
        last_dot_index = file_ending.rfind('.')
        substring = file_ending[last_dot_index + 1:]
        self.data_loaders[substring] = data_loader


    def get_data_loader(self, file_ending: str) -> 'DataLoader':
        last_dot_index = file_ending.rfind('.')
        substring = file_ending[last_dot_index + 1:]
        if substring in self.data_loaders:
            return self.data_loaders[substring]

        if substring == 'csv':
            return CSVLoader()
        else:
            raise Exception("no dataloader set for said file ending")


    def store(self) -> None:
        store_obj(self, "MetaData/Strategies/" + str(self.get_name()))

    @staticmethod
    def load(strategy_name: str) -> 'Strategy':
        return load_obj("MetaData/Strategies/" + strategy_name)


    @abstractmethod
    def match(self, data_model:'DataModel') -> Tuple['Matching', 'Stopwatch']:
        pass


    def get_name(self):
        return self.name

    def timed_match(self, data_model:'DataModel') -> Tuple['Matching', 'Stopwatch']:
        stopwatch = Stopwatch()
        stopwatch.start_timer("matching")
        match, s2 = self.match(data_model)
        stopwatch.stop_timer("matching")
        s2.merge(stopwatch)
        return match, s2






