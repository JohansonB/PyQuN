import pickle
import uuid
from abc import ABC, abstractmethod

from typing import Tuple

from PyQuN_Lab.CandidateSearch import CandidateSearch, NNCandidateSearch
from PyQuN_Lab.DataModel import DataModel, Matching, ModelSet
from PyQuN_Lab.Matching import MatchingStrategy, GreedyMatching, RandomMatching
from PyQuN_Lab.Stopwatch import StopWatchManager, Stopwatch
from PyQuN_Lab.Utils import store_obj, load_obj
from PyQuN_Lab.DataLoading import DataLoader


class Strategy(ABC):
    def __init__(self, name: str) -> None:
        self.id = uuid.uuid4()
        self.name = name
        self.data_loaders = {}

    def get_id(self) -> uuid.UUID:
        return id

    def set_data_loader(self, file_ending: str, data_loader: DataLoader):
        last_dot_index = file_ending.rfind('.')
        substring = file_ending[last_dot_index + 1:]
        self.data_loaders[substring] = data_loader


    def get_data_loader(self, file_ending: str) -> DataLoader:
        last_dot_index = file_ending.rfind('.')
        substring = file_ending[last_dot_index + 1:]
        return self.data_loaders[substring]


    def store(self) -> None:
        store_obj(self, "MetaData/Strategies/" + str(self.get_name()))

    @staticmethod
    def load(strategy_name: str) -> 'Strategy':
        load_obj("MetaData/Strategies/" + strategy_name)


    @abstractmethod
    def match(self, data_model:DataModel) -> Tuple[Matching, Stopwatch]:
        pass


    def get_name(self):
        return self.name

    def timed_match(self, data_model:DataModel) -> Tuple[Matching, Stopwatch]:
        stopwatch = StopWatchManager()
        stopwatch.start_timer(self.id,"matching")
        match, s2 = self.match(data_model)
        stopwatch.stop_timer(self.id,"matching")
        s2.merge(stopwatch)
        return match, s2


class RandomMatcher(Strategy):

    def match(self, data_model:ModelSet) -> [Matching, Stopwatch]:
        return RandomMatching().match(data_model), Stopwatch()


class VanillaRaQuN(Strategy):

    def __init__(self, name:str, candidate_search: CandidateSearch = NNCandidateSearch(), matching_strategy: MatchingStrategy = GreedyMatching()) -> None:
        self.search = candidate_search
        self.match_strat = matching_strategy
        super().__init__(name)

    def match(self, data_model: ModelSet) -> [Matching, Stopwatch]:
        return self.match_strat.match(data_model, self.search.candidate_search(data_model)), Stopwatch()


if __name__ == "__main__":
    from PyQuN_Lab.DataLoading import CSVLoader, DataLoader
    from PyQuN_Lab.DataModel import StringAttribute
    from PyQuN_Lab.Tests import test_element, test_model

    b = test_element("b", "c", "d", "e")
    a = test_element("a", "b", "c")
    c = test_element("a", "r", "f", "g", "h")
    d = test_element("a", "f", "h", "t", "i", "j")
    e = test_element("l", "o", "p", "q", "a", "s")
    f = test_element("l", "r", "a", "s")
    g = test_element("h", "i", "j", "t")

    model1 = test_model(a, b)
    model2 = test_model(c, d)
    model3 = test_model(e, f)
    model4 = test_model(g)

    ms = ModelSet()
    ms.add_model(model1)
    ms.add_model(model2)
    ms.add_model(model3)




    v_r2 = Strategy.load('test')
    out2 = v_r2.match(ms)
