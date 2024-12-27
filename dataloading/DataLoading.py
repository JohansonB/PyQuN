from abc import ABC, abstractmethod


class DataLoader(ABC):
    @abstractmethod
    def read_file(self, url: str) -> 'DataSet':
        pass





