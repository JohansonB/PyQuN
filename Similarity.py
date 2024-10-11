from abc import ABC, abstractmethod

from PyQuN_Lab.DataModel import Match


class ShouldMatchStrategy(ABC):
    @abstractmethod
    def should_match(self, similarity, m1, m2, e1, e2):
        pass


class GreedySMStrategy(ShouldMatchStrategy):
    def should_match(self, similarity, matches, elements=None):
        similarity.similarity(Match.merge_matches(matches)) > sum([similarity.similarity(m) for m in matches])


class ThresholdSMStrategy(ShouldMatchStrategy):
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def should_match(self, similarity, matches, elements=None):
        similarity(Match.merge_matches(matches)) >= self.threshold


class Similarity(ABC):
    def __init__(self, strategy):
        self.strategy = strategy
    @abstractmethod
    def similarity(self, match):
        pass
    @abstractmethod
    def should_match(self, matches, elements):
        pass

#needs debugging foso
class WeightMetric(Similarity):
    def __init__(self, model_set, strategy=GreedySMStrategy()):
        self.n = len(model_set)
        super().__init__(strategy)

    def similarity(self, match):
        dic = {}
        for ele in match:
            for att in ele:
                if att not in dic:
                    dic[att] = 1
                else:
                    dic[att] +=  1

        dic2 = {}
        for key, value in dic.items():
            if value not in dic2:
                dic2[value] = 1
            else:
                dic2[value] += 1
        sum = 0
        for key, value in dic2.items():
            if key>=2:
                sum += key**2*value
        return sum / (self.n**2*len(dic))

    def should_match(self, matches, elements=None):
        return self.strategy.should_match(self,matches,elements)


class JaccardIndex(Similarity):
    def __init__(self, threshold, strategy=ThresholdSMStrategy()):
        self.threshold = threshold
        super().__init__(strategy)

    def similarity(self, elements):
        sets = [set(element) for element in elements]
        intersection = set.intersection(*sets)
        union = set.union(*sets)
        return len(intersection)/len(union)

    def should_match(self, matches, elements=None):
        return self.strategy.should_match(self,matches,elements)
