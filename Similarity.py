from abc import ABC, abstractmethod

class Similarity(ABC):
    @abstractmethod
    def similarity(self, elements):
        pass
    @abstractmethod
    def should_match(self, t1, t2):
        pass

class Weight_Metric(Similarity):
    def __init__(self, n):
        self.n = n

    def similarity(self, elements):
        dic = {}
        for ele in elements:
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

    def should_match(self, t1, t2):
        return self.similarity(set.union(t1,t2)) > self.similarity(t1)+self.similarity(t2)

class Jaccard_Index(Similarity):
    def __init__(self, threshold):
        self.threshold = threshold
    def similarity(self, elements):
        sets = [set(element) for element in elements]
        intersection = set.intersection(*sets)
        union = set.union(*sets)
        return len(intersection)/len(union)

    def should_match(self, t1, t2):
        return self.similarity(set.union(t1,t2)) >= self.threshold
