from RaQuN_Lab.datamodel.matching.Match import Match
from RaQuN_Lab.datamodel.matching.similarities.match.shouldmatch.ShouldMatchStrategy import ShouldMatchStrategy


class GreedySMStrategy(ShouldMatchStrategy):
    def should_match(self, similarity, matches, elements=None):
        return similarity.similarity(Match.merge_matches(*matches)) > sum([similarity.similarity(m) for m in matches])

