from abc import ABC, abstractmethod
from typing import Iterable

from PyQuN_Lab.DataModel import Match, Element, ModelSet


class ShouldMatchStrategy(ABC):
    @abstractmethod
    def should_match(self, similarity:'Similarity', matches:Iterable[Match], elements:Iterable[Element]) -> bool:
        pass


class GreedySMStrategy(ShouldMatchStrategy):
    def should_match(self, similarity, matches, elements=None):
        return similarity.similarity(Match.merge_matches(*matches)) > sum([similarity.similarity(m) for m in matches])


class ThresholdSMStrategy(ShouldMatchStrategy):
    def __init__(self, threshold=0.25):
        self.threshold = threshold

    def should_match(self, similarity, matches, elements=None):
        return similarity.similarity(Match.merge_matches(*matches)) >= self.threshold


class Similarity(ABC):
    def __init__(self, strategy:ShouldMatchStrategy) -> None:
        self.strategy = strategy
    @abstractmethod
    def similarity(self, match:Match):
        pass
    @abstractmethod
    def should_match(self, matches:Iterable[Match], elements:Iterable[Element]):
        pass

#the Modelset is required to compute the weihtmetric, since the formula includes the number of models in the model_set
#however this is just a normalizing constant which does not influence the greedy should_match strategy, hence n
#is set to 1 incase no model_set is provided
class WeightMetric(Similarity):
    def __init__(self, num_modles:int = 1, strategy:ShouldMatchStrategy = GreedySMStrategy()):

        self.n = num_modles
        super().__init__(strategy)

    def similarity(self, match):
        attr_count = {}
        for ele in match:
            for att in ele:
                if att not in attr_count:
                    attr_count[att] = 1
                else:
                    attr_count[att] += 1

        attr_count_count = {}
        for key, value in attr_count.items():
            if value not in attr_count_count:
                attr_count_count[value] = 1
            else:
                attr_count_count[value] += 1
        sum = 0
        for key, value in attr_count_count.items():
            if key >= 2:
                sum += key ** 2 * value
        return sum / (self.n ** 2 * len(attr_count))

    def should_match(self, matches, elements=None):
        return self.strategy.should_match(self,matches,elements)

    def set_modelset(self, model_set:ModelSet) -> None:
        self.n = len(model_set)


class JaccardIndex(Similarity):
    def __init__(self, strategy:ShouldMatchStrategy = ThresholdSMStrategy()) -> None:
        super().__init__(strategy)

    def similarity(self, match):
        sets = [set(element) for element in match]
        intersection = set.intersection(*sets)
        union = set.union(*sets)
        return len(intersection)/len(union)

    def should_match(self, matches, elements=None):
        return self.strategy.should_match(self,matches,elements)
