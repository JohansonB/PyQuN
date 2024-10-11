import uuid
from abc import ABC, abstractmethod

import numpy as np
from collections import OrderedDict

from PyQuN_Lab.Vectorization import VectorizedModelSet


class Attribute(ABC):
    '''smallest component of the RaQuN data model
        Attributes with concrete datatypes extend this class
    '''

    def __init__(self, value=None):
        self.value = value

    @abstractmethod
    def dist(self, other):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    # takes a string representation of the attribute and parses it
    @abstractmethod
    def parse_string(self, encoding):
        pass


class DefaultAttribute(Attribute, ABC):
    def __init__(self, value=None):
        super().__init__(value)

    def dist(self, other):
        pass

    def __eq__(self, other):
        return self.value == other.value

    def __len__(self):
        return len(self.value)

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return repr(self.value)

    def parse_string(self, encoding):
        self.value = encoding


class StringAttribute(DefaultAttribute):
    def __init__(self, value=None):
        if not value is None and not isinstance(value, str):
            raise ValueError(f"StringAttribute must be initialized with a string. Got {type(value)} instead.")

        super().__init__(value)


class Element:

    def __init__(self, attributes=None, name=None):
        if attributes is None:
            attributes = set()
        else:
            if not isinstance(attributes, set):
                raise ValueError(f"Attributes must be initialized with a set. Got {type(attributes)} instead.")

                # Check if all elements in the set are instances of Attribute
            if not all(isinstance(attr, Attribute) for attr in attributes):
                raise ValueError("All elements of the set must be instances of Attribute.")

        self.attributes = attributes
        self.ele_id = uuid.uuid4()
        self.name = name
        self.model_id = None

    def get_name(self):
        return self.name

    def add_attr(self, attribute):
        self.attributes.add(attribute)

    def set_model_id(self,model_id):
        self.model_id = model_id

    def get_id(self):
        return self.ele_id

    def set_element_id(self, element_id):
        self.ele_id = element_id

    def get_model_id(self):
        return self.model_id

    def set_name(self, name):
        self.name = name

    def __iter__(self):
        return iter(self.attributes)

    def __len__(self):
        return len(self.attributes)

    def __eq__(self, other):
        return self.ele_id == other.get_id()

    def __repr__(self):
        print(self.attributes)
        return "Element_id: "+repr(self.ele_id)+"\n"+'\n'.join(["[" + repr(attr) + "]" for attr in self.attributes])

    def __hash__(self):
        return hash(self.ele_id)


class Model:

    def __init__(self, elements=None):
        if elements is None:
            elements = set()
        else:
            if not isinstance(elements, set):
                raise ValueError(f"Attributes must be initialized with a set. Got {type(elements)} instead.")

                # Check if all elements in the set are instances of Attribute
            if not all(isinstance(ele, Element) for ele in elements):
                raise ValueError("All elements of the set must be instances of Attribute.")

        self.elements = elements
        self.id = uuid.uuid4()

    def add_element(self, element):
        self.elements.add(element)
        element.set_model_id(self.id)

    def remove_element(self, element):
        self.elements.remove(element)

    def get_id(self):
        return self.id

    def get_elements(self):
        return self.elements

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __eq__(self, other):
        return self.id == other.get_id()

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return "Model_id: "+repr(self.id)+"\n"+"\n".join(["{" + repr(ele) + "}" for ele in self.elements])


class ModelSet:

    def __init__(self, models=None, id_dictionary=True):
        if models is None:
            models = set()
        else:
            if not isinstance(models, set):
                raise ValueError(f"Attributes must be initialized with a set. Got {type(models)} instead.")

                # Check if all elements in the set are instances of Attribute
            if not all(isinstance(model, Model) for model in models):
                raise ValueError("All elements of the set must be instances of Attribute.")

        self.models = models
        self.id_map = {}
        self.__elements = set()
        for m in models:
            for e in m.get_elements():
                self.__elements.add(e)
        self.use_dictionary = id_dictionary
        if id_dictionary:
            self._innit_id_map()


    def __iter__(self):
        return iter(self.models)

    def __len__(self):
        tot = 0
        for model in self.models:
            tot += len(model)
        return tot

    def __repr__(self):
        return "\n".join(["{"+repr(m)+"}" for m in self.models])

    def _innit_id_map(self):
        for model in self.models:
            self.id_map[model.get_id()] = model
            for ele in model:
                self.id_map[ele.get_id()] = ele

    def _linear_id_search(self, id):
        for model in self.models:
            if model.get_id() == id:
                return model
            for ele in model:
                if ele.get_id() == id:
                    return ele
        raise KeyError(f"Id '{id}' was not found in the ModelSet.")

    # returns the model/element corresponding to an id
    def get_by_id(self, id):
        if self.use_dictionary:
            if not self.id_map:
                self._innit_id_map()
            if id not in self.id_map:
                raise KeyError(f"Id '{id}' was not found in the ModelSet.")
            else:
                return self.id_map[id]
        else:
            return self._linear_id_search(id)

    # returns a set of all the elements in the models
    def get_elements(self):
        return self.__elements

    def add_model(self, model):
        self.models.add(model)
        for e in model:
            self.__elements.add(e)
        if self.use_dictionary:
            self.id_map[model.get_id()] = model
            for ele in model:
                self.id_map[ele.get_id()] = ele

    def remove_model(self, model):
        self.models.remove(model)
        for e in model:
            self.__elements.remove(e)
        if self.use_dictionary:
            del self.id_map[model.get_id()]
            for ele in model:
                del self.id_map[ele.get_id()]


    def vectorize(self, vectorizer):
        return VectorizedModelSet(self, vectorizer)


class MatchViolation(Exception):
    def __init__(self):
        super().__init__("the added elements cause a match violation \n" +
                         ", since atleast 2 elements belong to the same model.\n")


class Match:
    def __init__(self, elements=set(), do_check=True):
        self.id = uuid.uuid4()
        self.eles = set()
        if do_check and not self.are_valid(elements):
            raise MatchViolation()
        self.eles = elements
        self.do_check = do_check

    def __eq__(self, other):
        return self.id == other.get_id()

    def __hash__(self):
        return hash(self.id)

    def __iter__(self):
        return iter(self.eles)

    def get_elements(self):
        return self.eles

    def add_elements(self, *elements):

        if self.do_check and not self.are_valid(elements):
            raise MatchViolation()
        eles_copy = self.eles.copy()
        eles_copy.update(elements)
        return Match(eles_copy, self.do_check)

    def is_valid(self, element):
        for ele in self.eles:
            if element.get_model_id() == ele.get_model_id():
                return False
        return True

    def are_valid(self, elements):
        for ele in elements:
            if not self.is_valid(ele):
                return False
        return True

    @staticmethod
    def merge_matches(*matches, do_check=True):
        eles = {}
        if len(matches) == 1:
            return matches[0]

        for m in matches:
            eles.update(m.get_elements())
        return Match(eles, do_check)


# this class represents the output of phase 2 of the RaQuN algorithm.
# the matching Candidates consists of a ordered list of valid matches, without the requirement that the matches need to be
# mutually exclusive
class MatchCandidates:
    def __init__(self):
        self.is_sorted = True
        self.matches = OrderedDict()

    def add_matches(self, *matches):
        for match in matches:
            self.matches[match] = None
        if len(matches) > 0:
            self.is_sorted = False

    # removes all candidates whoms similarity lies below a certain threshold
    def filter(self, similarity, threshold):
        keys_to_remove = [key for key, value in self.matches.items() if similarity.similarity(key) < threshold]
        for key in keys_to_remove:
            del self.matches[key]

    # sort candidates matches accoriding to a similarity fucntion
    def sort(self, similarity):
        self.matches = OrderedDict(
            sorted(self.matches.items(), key=lambda item: similarity.similarity(item[0]), reverse=True))
        self.is_sorted = True

    # returns the first match of the match candidates, should be called after sorting the matches to get the candidate
    # with the highest similarity
    def head(self):
        if self.is_sorted:
            return next(iter(self.matches))
        else:
            raise Exception("you need to call the sort function before calling this method")

    def pop(self):
        if self.is_sorted:
            ne = self.head()
            self.delete(ne)
            return ne
        else:
            raise Exception("you need to call the sort function before calling this method")

    def is_empty(self):
        return len(self.matches) == 0

    def delete(self, match):
        del self.matches[match]


# this class represents the matching object on which the algorithm operates in the third phase of the algorithm.
# it enforces that the collection of matches is mutually exclusive

class Matching:
    def __init__(self):
        self.matches = set()

    def trivial_matching(self, model_set):
        self.matches = set()
        for element in model_set.get_elements():
            self.matches.add(Match({element}))

    def get_match_by_element(self, ele):
        for m in self.matches:
            if ele in m:
                return m
        raise ValueError("element is not contained in the matching")

    # merges a collection of matches
    def merge_matches(self, *matches):
        for match in matches:
            self.matches.remove(match)
        self.matches.add(Match.merge_matches(matches, do_chek=False))


if __name__ == "__main__":
    from PyQuN_Lab.Matching import GreedyMatching
    from PyQuN_Lab.CandidateSearch import NNCandidateSearch
    from PyQuN_Lab.DataLoading import CSVLoader
    from PyQuN_Lab.Similarity import WeightMetric

    loader = CSVLoader("C:/Users/41766/Desktop/full_subjects/hospitals.csv", StringAttribute)
    out = loader.parse_file()
    model_set = out.model_set
    cs = NNCandidateSearch()
    candidates = cs.candidate_search(model_set)
    similarity = WeightMetric(model_set)
    matching = GreedyMatching().match(model_set, candidates, similarity)
    print("attack")
