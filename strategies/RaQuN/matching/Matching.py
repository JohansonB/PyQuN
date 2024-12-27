from abc import ABC, abstractmethod


class MatchingStrategy(ABC):
    # the matching function which takes as input the model_set, a Candidate_set, a similarity and produces a matching
    @abstractmethod
    def match(self, model_set: 'ModelSet', candidates: 'MatchCandidates' = None) -> 'Matching':
        pass


