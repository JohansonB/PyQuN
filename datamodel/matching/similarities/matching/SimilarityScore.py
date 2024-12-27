from RaQuN_Lab.datamodel.matching.similarities.match.WeightMetric import WeightMetric
from RaQuN_Lab.datamodel.matching.similarities.matching.EvaluationScore import EvaluationScore
from RaQuN_Lab.datamodel.matching.similarities.matching.statistics.AverageStatistic import AverageStatistic


class SimilarityScore(EvaluationScore):
    def __init__(self, similarity : 'Similarity' = WeightMetric(), statistic : 'Statistic' = AverageStatistic()):
        self.similarity = similarity
        self.statistic = statistic

    def _pevaluate(self, matching: 'Matching') -> float:
        m_s = matching.create_modelset()
        self.similarity.innit(m_s)

        similarities = [0 if len(m) == 1 else self.similarity.similarity(m) for m in matching]
        return self.statistic.evaluate(similarities)
