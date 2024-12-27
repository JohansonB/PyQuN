
#the Modelset is required to compute the weihtmetric, since the formula includes the number of models in the model_set
#however this is just a normalizing constant which does not influence the greedy should_match strategy, hence n
#is set to 1 incase no model_set is provided
from RaQuN_Lab.datamodel.matching.similarities.match.Similarity import Similarity
from RaQuN_Lab.datamodel.matching.similarities.match.shouldmatch.GreedySMStrategy import GreedySMStrategy
from RaQuN_Lab.datamodel.matching.similarities.match.shouldmatch.ShouldMatchStrategy import ShouldMatchStrategy


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

    def set_modelset(self, model_set:'ModelSet') -> None:
        self.n = len(model_set)

    def set_num_models(self, n : int)-> None:
        self.n = n


    def innit(self, m_s: 'Modelset'):
        self.n = len(m_s)