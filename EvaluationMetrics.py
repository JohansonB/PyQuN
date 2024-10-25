from abc import ABC, abstractmethod
from typing import List, Tuple, Iterable

from PyQuN_Lab.DataModel import Matching, Match, Element
from PyQuN_Lab.Similarity import WeightMetric


def get_all_tuples(elements: Iterable[Element]) -> List[Tuple[Element]]:
    ret = []
    temp = elements.get_elements().copy()
    for e1 in elements:
        temp.remove(e1)
        for e2 in temp:
            ret.append((e1, e2))

    return ret


class EvaluationMetric(ABC):
    @abstractmethod
    def evaluate(self, matching:Matching) -> float:
        pass


class Weight(EvaluationMetric):

    def __init__(self, num_models:int = 1) -> None:
        self.weight_metric = WeightMetric(num_models)

    def evaluate(self, matching: Matching) -> float:
        tot = 0
        for m in matching:
            tot += self.weight_metric.similarity(m)

        return tot


class FalseNegative(EvaluationMetric):
    def evaluate(self, matching:Matching) -> float :
        count = 0
        for m1 in matching:
            for e1 in m1:
                for m2 in matching:
                    if m1 == m2:
                        continue
                    for e2 in m2:
                        if e1.get_custom_id() == e2.get_custom_id():
                            count += 1
        return count


class TruePositive(EvaluationMetric):

    def evaluate(self, matching: Matching) -> float:
        count = 0
        for match in matching:
            for t in get_all_tuples(match):
                if t[0].get_custom_id() == t[1].get_custom_id():
                    count += 1


class FalsePositive(EvaluationMetric):

    def evaluate(self, matching: Matching) -> float:
        count = 0
        for match in matching:
            for t in get_all_tuples(match):
                if t[0].get_custom_id() != t[1].get_custom_id():
                    count += 1


class Precision(EvaluationMetric):

    def evaluate(self, matching: Matching) -> float:
        tp = TruePositive().evaluate(matching)
        fp = FalsePositive.evaluate(matching)
        return tp / (tp+fp)


class Recall(EvaluationMetric):
    def evaluate(self, matching: Matching) -> float:
        tp = TruePositive().evaluate(matching)
        fn = FalseNegative().evaluate(matching)
        return tp/(tp+fn)


class FMeasure(EvaluationMetric):
    def evaluate(self, matching:Matching) -> float:
        prec = Precision().evaluate(matching)
        rec = Recall().evaluate(matching)
        return prec*rec/(prec+rec)
