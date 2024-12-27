from RaQuN_Lab.datamodel.matching.Match import Match
from RaQuN_Lab.datamodel.matching.similarities.match.shouldmatch.ShouldMatchStrategy import ShouldMatchStrategy


class ThresholdSMStrategy(ShouldMatchStrategy):
    def __init__(self, threshold=0.25):
        self.threshold = threshold

    def should_match(self, similarity, matches, elements=None):
        return similarity.similarity(Match.merge_matches(*matches)) >= self.threshold

