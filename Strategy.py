import uuid
from abc import ABC, abstractmethod


from PyQuN_Lab.CandidateSearch import CandidateSearch, NNCandidateSearch
from PyQuN_Lab.DataModel import DataModel, Matching, ModelSet
from PyQuN_Lab.Matching import MatchingStrategy, GreedyMatching
from PyQuN_Lab.Stopwatch import StopWatchManager


class Strategy(ABC):
    def __init__(self) -> None:
        self.id = uuid.uuid4()

    @abstractmethod
    def match(self, data_model:DataModel) -> Matching:
        pass

    def timed_match(self, data_model:DataModel) -> Matching:
        stopwatch_manager = StopWatchManager()
        stopwatch_manager.

class VanillaRaQuN(Strategy):
    def __init__(self, candidate_search:CandidateSearch = NNCandidateSearch(), matching_strategy:MatchingStrategy = GreedyMatching()) -> None:
        self.search = candidate_search
        self.match_strat = matching_strategy
        super().__init__()

    def match(self, data_model: ModelSet) -> Matching:
        return self.match_strat.match(data_model, self.search.candidate_search(data_model))

if __name__ == "__main__":
    from PyQuN_Lab.DataLoading import CSVLoader
    from PyQuN_Lab.DataModel import StringAttribute
    loader = CSVLoader("C:/Users/41766/Desktop/full_subjects/hospitals.csv", StringAttribute)
    out = loader.parse_input()
    model_set = out.model_set
    out = VanillaRaQuN().match(model_set)
    print(out)