import random
from abc import ABC, abstractmethod
from typing import Tuple, Iterable, List, Union

from PyQuN_Lab.DataModel import DataModel, Matching, Match
from PyQuN_Lab.EvaluationMetrics import Weight
from PyQuN_Lab.Similarity import WeightMetric
from PyQuN_Lab.Stopwatch import Stopwatch
from PyQuN_Lab.Strategy import Strategy, RandomMatcher




class Population:
    def __init__(self, matchings: Iterable[Matching] = None):
        if matchings is None:
            matchings = []
        self.matchings = list(matchings)

    def add(self, matching: Matching) -> None:
        self.matchings.append(matching)

    def __iter__(self):
        iter(self.matchings)


    def crossover(self):
        w = Weight()
        selector = Selector(self.matchings, lambda m: w.evaluate(m),True,False,False)
        count = 0
        children = []
        while not selector.is_empty():
            if count == 0:
                p1 = selector.next()
            else:
                p2 = selector.next()
                children.append(self.crossover_matching(p1,p2))

            count = (count + 1) % 2

        self.matchings.extend(children)

    @staticmethod
    def crossover_matching(m1: Matching, m2 : Matching):
        ret = set()
        matchings = list(m1.get_matches())
        matchings.extend(list(m2.get_matches()))
        met = WeightMetric()
        selector = Selector(matchings,lambda m : met.similarity(m),True,False,False)
        while not selector.is_empty():
            ret.add(selector.next())
        return Matching(ret)




    def mutate(self, mutation_rate:float = 0.1, merge_rate:float = 0.5, max_resamples:int = 10) -> None:
        ret = []
        for m in self.matchings:
            num_e = m.num_elements()
            ret.append(self.mutate_matching(m,mutation_rate,merge_rate, max_resamples))
            #assert num_e == ret[-1].num_elements() and ret[-1].is_mutual_exclusive()
        self.matchings = ret



    @staticmethod
    def mutate_matching(matching:Matching, mutation_rate:float = 0.1, merge_rate:float = 0.5, max_resamples:int = 10) -> Matching:
        ret = set()
        matchings_set = set(matching)
        for m in matching:
            if m not in matchings_set:
                continue
            r = random.uniform(0,1)
            if r <= mutation_rate:
                r = random.uniform(0,1)
                if r <= merge_rate or len(m) == 1:
                    r = random.sample(matchings_set,1)[0]
                    count = 0
                    while not Match.valid_merge(r,m) and count < max_resamples:
                        r = random.sample(matchings_set, 1)[0]
                        count += 1
                    if Match.valid_merge(r,m):
                        matchings_set.remove(m)
                        matchings_set.remove(r)
                        ret.add(Match.merge_matches(r,m))
                    else:
                        #no match could be found resampling so we dont do the merge afterall
                        ret.add(m)
                        matchings_set.remove(m)
                #split the match instead of merging it
                else:
                    r = random.randint(1,len(m)-1)
                    ele_list = list(m.get_elements())
                    random.shuffle(ele_list)
                    lo = ele_list[:r]
                    hi = ele_list[r:]
                    ret.add(Match(lo,do_check=False))
                    ret.add(Match(hi,do_check=False))
                    matchings_set.remove(m)
            else:
                ret.add(m)
                matchings_set.remove(m)
        return Matching(ret)




class InitialPopulation(ABC):
    @abstractmethod
    def sample(self, m_s:'Modelset') -> Population:
        pass


class RNGPopulation(InitialPopulation):
    def __init__(self, num = 100):
        self.num = num
    def sample(self, m_s: 'Modelset') -> Population:
        ret = Population()
        r_s = RandomMatcher("temp")
        for i in range(self.num):
            print("random innit count: " + str(i))
            ret.add(r_s.match(m_s)[0])
        return ret



class SelectionStrategy(ABC):
    def __init__(self,size: int = 50):
        self.size = size
    @abstractmethod
    def select(self, pop : Population, size: int) -> Population:
        pass

class Selector:
    def __init__(self, members: List[Union[Matching, Match]], score, filter: bool = False, uniform: bool = False, is_sorted = False) -> Union[Matching, Match]:

        if not is_sorted:
            self.sorted_members = sorted(members, key=score, reverse=True)
        else:
            self.sorted_members = members
        self.tot = sum([score(m) for m in members])
        self.score = score
        self.deleted_indices = set()
        self.do_filter = filter
        self.uniform = uniform

    def next(self):
        rand = random.uniform(0,1)
        cur = 0
        count = 0
        while cur < rand and count < len(self.sorted_members):
            while count in self.deleted_indices:
                count += 1
            if count >= len(self.sorted_members):
                break
            if self.uniform:
                cur += 1/(len(self.sorted_members)-len(self.deleted_indices))
            else:
                if self.tot == 0:
                    self.uniform = True
                    return self.next()
                cur += self.score(self.sorted_members[count])/self.tot
            count += 1
        if cur >= rand:
            ret = self.sorted_members[count - 1]
            if self.do_filter:
                if isinstance(ret, Match):
                    self.filter(ret)
                if isinstance(ret,Matching):
                    self.filter(count-1)
            return ret
        else:
            #this should accutally not occure but due to low score values and precision errors it does occure
            #so in this case i switch the selector to unifrom selection
            self.uniform = True
            return self.next()

    def is_empty(self):
        return len(self.deleted_indices) == len(self.sorted_members)


    #this methods assumes that the members are of type match
    def filter(self, match: Union[Match,int]) -> None:
        if isinstance(match, Match):
            for i in range(len(self.sorted_members)):
                if i in self.deleted_indices:
                    continue
                m = self.sorted_members[i]
                if not Match.is_disjoint(match, m):
                    self.deleted_indices.add(i)
                    self.tot -= self.score(m)

        else:
            self.tot -= self.score(self.sorted_members[match])
            self.deleted_indices.add(match)










class TournamentSelection(SelectionStrategy):
    def __init__(self,size: int = 50, tournament_size=2, replacement : bool = True):
        self.tournament_size = tournament_size
        self.replacement = replacement
        super().__init__(size)


    def select(self, pop: Population) -> Population:
        ret = Population()
        weight = Weight()
        selector = Selector(pop.matchings, lambda m: weight.evaluate(m), uniform=True,filter=not self.replacement)
        for _ in range(self.size):
            candidates = [selector.next() for _ in range(self.tournament_size)]
            best_candidate = max(candidates, key=lambda m: weight.evaluate(m))
            ret.add(best_candidate)
        return ret


class RouletteSelection(SelectionStrategy):
    def __init__(self,size: int = 50, replacement: bool = True):
        self.replacement = replacement
        super().__init__(size)


    def select(self, pop: Population) -> Population:
        ret = Population
        weight = Weight()
        selector = Selector(pop, lambda m: weight.evaluate(m), uniform=False, filter=not self.replacement)
        for _ in range(self.size):
            ret.add(selector.next())
        return ret


class RankSelection(SelectionStrategy):
    def __init__(self,size: int = 50):
        super().__init__(size)
    def select(self, pop: Population) -> Population:
        w = Weight()
        return Population(sorted(list(pop), key=lambda m : w.evaluate(m), reverse=True)[:self.size])


class GeneticStrategy(Strategy):
    def __init__(self,name : str, initializer: 'InitialPopulation' = RNGPopulation() , selection : 'SelectionStrategy' = TournamentSelection(), mutation_rate: float = 0.1, merge_rate : float = 0.5,  num_iterations: int = 500):
        self.initializer = initializer
        self.selection = selection
        self.mutation_rate = mutation_rate
        self.merge_rate = merge_rate
        self.num_iterations = num_iterations
        super().__init__(name)
    def match(self, data_model: DataModel) -> Tuple[Matching, Stopwatch]:
        from PyQuN_Lab.EvaluationMetrics import Weight

        pop = self.initializer.sample(data_model)
        for i in range(self.num_iterations):
            print("iteration count: "+str(i))
            pop = self.selection.select(pop)
            pop.mutate(mutation_rate=self.mutation_rate,merge_rate=self.merge_rate)
            pop.crossover()

        return sorted(pop.matchings, key=lambda m : Weight().evaluate(m), reverse=True)[0], Stopwatch()



