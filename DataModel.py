import copy
import random
import uuid
from abc import ABC, abstractmethod

import numpy as np
import pickle
from typing import Any, Union, Iterable, Set
from collections import OrderedDict

from PyQuN_Lab.Utils import store_obj, load_obj
from PyQuN_Lab.Vectorization import VectorizedModelSet, Vectorizer

class DataModel(ABC):
    pass

class Attribute(ABC):
    '''smallest component of the RaQuN data model
        Attributes with concrete datatypes extend this class
    '''

    def __init__(self, value:Any = None)-> None:
        self.value = value

    @abstractmethod
    def dist(self, other:'Attribute') -> int:
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
    def parse_string(self, encoding:str) -> None:
        pass

    @abstractmethod
    def clone(self) -> 'Attribute':
        pass


class DefaultAttribute(Attribute, ABC):
    def __init__(self, value:Any = None) -> None:
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
        return f"{self.__class__.__name__}(value={self.value})"


    def clone(self) -> 'Attribute':
        return type(self)(self.value)

    def parse_string(self, encoding):
        self.value = encoding


class StringAttribute(DefaultAttribute):
    def __init__(self, value:str = None) -> None:
        if not value is None and not isinstance(value, str):
            raise ValueError(f"StringAttribute must be initialized with a string. Got {type(value)} instead.")

        super().__init__(value)


class Element:

    def __init__(self, attributes:Set[Attribute] = None, name:str = None) -> None:
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
        self.custom_id = None

    def set_custom_id(self, custom_id:int) -> None:
        self.custom_id = custom_id

    def get_custom_id(self) -> int:
        return self.custom_id

    def get_name(self) -> str:
        return self.name

    def add_attr(self, attribute:Attribute) -> None:
        self.attributes.add(attribute)

    def set_model_id(self,model_id:uuid.UUID) -> None:
        self.model_id = model_id

    def get_id(self) -> uuid.UUID:
        return self.ele_id

    def set_element_id(self, element_id:uuid.UUID) -> None:
        self.ele_id = element_id

    def get_model_id(self) -> uuid.UUID:
        return self.model_id

    def set_name(self, name : str):
        self.name = name

    def __iter__(self):
        return iter(self.attributes)

    def __len__(self):
        return len(self.attributes)

    def __eq__(self, other):
        return self.ele_id == other.get_id()

    def __repr__(self):
        attrs = ', '.join([repr(attr) for attr in self.attributes])
        return f"Element(id={self.ele_id}, name={self.name}, attrs=[{attrs}])"

    def __hash__(self):
        return hash(self.ele_id)

    #cloned elements retain the id of the original element
    def clone(self) -> 'Element':
        attr_set = set()
        for attr in self.attributes:
            attr_set.add(attr.clone())

        ele_clone = Element(attr_set)
        ele_clone.ele_id = self.ele_id
        ele_clone.name = self.name
        ele_clone.model_id = self.model_id
        ele_clone.custom_id = self.custom_id

        return ele_clone


class Model:

    def __init__(self, elements:Set[Element] = None) -> None:
        if elements is None:
            elements = set()
        else:
            if not isinstance(elements, set):
                raise ValueError(f"Attributes must be initialized with a set. Got {type(elements)} instead.")

                # Check if all elements in the set are instances of Attribute
            if not all(isinstance(ele, Element) for ele in elements):
                raise ValueError("All elements of the set must be instances of Attribute.")

        self.elements = elements
        self.model_set = None
        self.id = uuid.uuid4()

    def clone(self) -> 'Model':
        ele_set = set()
        for ele in self.elements:
            ele_set.add(ele.clone())
        model_clone = Model(ele_set)
        model_clone.id = self.id
        return model_clone

    def set_model_set(self, model_set : 'ModelSet') -> None:
        self.model_set = model_set

    def set_elements(self, eles : Iterable[Element]) -> None:
        if self.model_set is not None:
            for e in self.elements:
                self.model_set.notify_ele_del(e)
            for e in eles:
                self.model_set.notify_ele_add(e)
        self.elements = set(eles)




    def add_element(self, element:Element) -> None:
        self.elements.add(element)
        element.set_model_id(self.id)
        if self.model_set is not None:
            self.model_set.notify_ele_add(element)

    def shuffle_elements(self):
        li = list(self.elements)
        random.shuffle(li)
        self.elements = set(li)

    def remove_element(self, element:Element) -> None:
        self.elements.remove(element)
        if self.model_set is not None:
            self.model_set.notify_ele_del(element)

    def get_id(self) -> uuid.UUID:
        return self.id

    def get_elements(self) -> Set[Element]:
        return self.elements

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __eq__(self, other):
        return self.id == other.get_id()

    def __hash__(self):
        # Use `self.id` if it’s initialized; otherwise, fall back to `id(self)`
        return hash(getattr(self, 'id', id(self)))

    def __repr__(self):
        return f"Model(id={self.id}, num_elements={len(self.elements)})"


class ModelSet(DataModel):

    def __init__(self, models:Set[Model] = None, id_dictionary:bool = True) -> None :
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
            self.__elements.update(m.get_elements())
            m.set_model_set(self)

        self.use_dictionary = id_dictionary
        if id_dictionary:
            self._innit_id_map()

    def clone(self) -> 'ModelSet':
        m_set = set()
        for m in self.models:
            m_set.add(m.clone())
        return ModelSet(m_set, self.use_dictionary)





    def __iter__(self):
        return iter(self.models)

    def __len__(self):
        return len(self.models)

    def __repr__(self):
        return f"ModelSet(num_models={len(self.models)}, num_elements={len(self.get_elements())})"

    def get_models(self) -> Set[Model]:
        return self.models

    def get_subset(self, models: Union[Iterable[uuid.UUID], int]) -> 'ModelSet':
        if isinstance(models, int):
            models = list(self.models)[:models]
        else:
            models = [self.get_by_id(m) for m in models]
        ze_copy = self.clone()
        to_remove = set()
        for m in ze_copy.get_models():
            if m not in models:
                to_remove.add(m)
        ze_copy.get_models().difference_update(to_remove)
        return ze_copy

    def shorten(self, factor : float) -> 'ModelSet':
        ze_copy = self.clone()
        for m in ze_copy:
            cur_len = int(len(m)*factor)
            m.set_elements(list(m.get_elements())[:cur_len])
        return ze_copy

    def shuffle_models(self):
        li = list(self.models)
        random.shuffle(li)
        self.models = set(li)

    def shuffle_elements(self):
        for m in self.models:
            m.shuffle_elements()



    def _innit_id_map(self) -> None:
        for model in self.models:
            self.id_map[model.get_id()] = model
            for ele in model:
                self.id_map[ele.get_id()] = ele

    def _linear_id_search(self, id:uuid.UUID) -> Union[Model, Element]:
        for model in self.models:
            if model.get_id() == id:
                return model
            for ele in model:
                if ele.get_id() == id:
                    return ele
        raise KeyError(f"Id '{id}' was not found in the ModelSet.")

    # returns the model/element corresponding to an id
    def get_by_id(self, id: uuid.UUID) -> Union[Model, Element]:
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
    def get_elements(self) -> Set[Element]:
        return self.__elements

    def notify_ele_add(self, ele:Element) -> None:
        self.__elements.add(ele)
        if self.use_dictionary:
            self.id_map[ele.get_id()] = ele

    def notify_ele_del(self, ele:Element) -> None:
        self.__elements.remove(ele)
        if self.use_dictionary:
            del self.id_map[ele.get_id()]


    def add_model(self, model:Model) -> None:
        self.models.add(model)
        self.__elements.update(model.get_elements())
        model.set_model_set(self)
        if self.use_dictionary:
            self.id_map[model.get_id()] = model
            for ele in model:
                self.id_map[ele.get_id()] = ele

    def remove_model(self, model:Model) -> None:
        self.models.remove(model)
        self.__elements.difference_update(model.get_elements())
        if self.use_dictionary:
            del self.id_map[model.get_id()]
            for ele in model:
                del self.id_map[ele.get_id()]


    def vectorize(self, vectorizer:Vectorizer) -> VectorizedModelSet:
        return VectorizedModelSet(self, vectorizer)


class MatchViolation(Exception):
    def __init__(self):
        super().__init__("the added elements cause a match violation \n" +
                         ", since atleast 2 elements belong to the same model.\n")


class Match:
    def __init__(self, elements:Set[Element] = set(), do_check:bool = True) -> None:
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

    def __repr__(self):
        elements_preview = ', '.join([repr(ele) for ele in list(self.eles)[:3]])  # Show up to 3 elements
        return f"Match(id={self.id}, num_elements={len(self.eles)}, elements=[{elements_preview}]...)"


    def get_elements(self) -> Set[Element]:
        return self.eles

    def add_elements(self, *elements:Element) -> None:

        if self.do_check and not self.are_valid(elements):
            raise MatchViolation()
        eles_copy = self.eles.copy()
        eles_copy.update(elements)
        return Match(eles_copy, self.do_check)


    def are_valid(self, elements: Iterable[Element]) -> bool:
        seen_eles = set()

        for ele in elements:
            # Check within elements to ensure no duplicates
            if ele in seen_eles:
                return False  # Duplicate found
            seen_eles.add(ele)

            # Check against elements in self.eles
            for ele2 in self.eles:
                if ele.get_model_id() == ele2.get_model_id():
                    return False
            # Check against other elements in elements
            for ele2 in elements:
                if ele != ele2 and ele.get_model_id() == ele2.get_model_id():
                    return False

        # If no violations found
        return True

        return True

    @staticmethod
    def valid_merge(*matches:'Match') -> bool:
        eles = set()
        for m in matches:
            eles.update(m.get_elements())
        return Match().are_valid(eles)

    @staticmethod
    def merge_matches(*matches:'Match', do_check:bool = True) -> None:
        eles = set()
        if len(matches) == 1:
            return matches[0]

        for m in matches:
            eles.update(m.get_elements())
        return Match(eles, do_check)


# this class represents the output of phase 2 of the RaQuN algorithm.
# the matching Candidates consists of a ordered list of valid matches, without the requirement that the matches need to be
# mutually exclusive
class MatchCandidates:
    def __init__(self) -> None:
        self.is_sorted = True
        self.matches = OrderedDict()

    def __iter__(self):
        return iter(self.matches.keys())

    def add_matches(self, *matches:Match) -> None:
        for match in matches:
            self.matches[match] = None
        if len(matches) > 0:
            self.is_sorted = False

    # removes all candidates whoms similarity lies below a certain threshold
    def filter(self, similarity:'Similarity', threshold:float) -> None:
        keys_to_remove = [key for key, value in self.matches.items() if similarity.similarity(key) < threshold]
        for key in keys_to_remove:
            del self.matches[key]

    # sort candidates matches accoriding to a similarity fucntion
    def sort(self, similarity:'Similarity'):
        self.matches = OrderedDict(
            sorted(self.matches.items(), key=lambda item: similarity.similarity(item[0]), reverse=True))
        self.is_sorted = True

    # returns the first match of the match candidates, should be called after sorting the matches to get the candidate
    # with the highest similarity
    def head(self) -> Match:
        if self.is_sorted:
            return next(iter(self.matches))
        else:
            raise Exception("you need to call the sort function before calling this method")

    def pop(self) -> Match:
        if self.is_sorted:
            ne = self.head()
            self.delete(ne)
            return ne
        else:
            raise Exception("you need to call the sort function before calling this method")

    def is_empty(self) -> bool:
        return len(self.matches) == 0

    def delete(self, match:Match) -> None:
        del self.matches[match]

    def __repr__(self):
        # Convert all matches into string representations
        all_matches = "\n".join([repr(match) for match in self.matches])
        return f"MatchCandidates(num_matches={len(self.matches)}, is_sorted={self.is_sorted}, matches=\n{all_matches})"


# this class represents the matching object on which the algorithm operates in the third phase of the algorithm.
# it enforces that the collection of matches is mutually exclusive

class Matching:
    def __init__(self) -> None:
        self.matches = set()

    def get_match_by_element(self, ele:Element) -> Match:
        for m in self.matches:
            if ele in m:
                return m
        raise ValueError("element is not contained in the matching")

    # merges a collection of matches
    def merge_matches(self, *matches:Match, do_check:bool = True) -> None:
        for match in matches:
            self.matches.remove(match)
        self.matches.add(Match.merge_matches(*matches, do_check=do_check))

    def __repr__(self):
        matches_preview = ', '.join([repr(match) for match in list(self.matches)[:3]])  # Show up to 3 matches
        return f"Matching(num_matches={len(self.matches)}, matches=[{matches_preview}]...)"

    def is_mutual_exclusive(self, match):
        for e in match:
            for m in self.matches:
                if e in m:
                    return False
        return True


    def add_match(self, match, do_check = True):
        if do_check:
            if not self.is_mutual_exclusive(match):
                raise Exception("The match you are trying to add violates mutual exclusivity. A match contained"
                                "in the Matching allready contains one of the elements in the input match")
        self.matches.add(match)
    @staticmethod
    def trivial_matching(model_set:ModelSet) -> 'Matching':
        matching = Matching()
        matching.matches = set()
        for element in model_set.get_elements():
            matching.matches.add(Match({element}))
        return matching

    def __iter__(self):
        return iter(self.matches)

    def store(self, path: str) -> None:
        store_obj(self,path)

    @staticmethod
    def load(path: str) -> 'Matching':
        return load_obj(path)


if __name__ == "__main__":
    from PyQuN_Lab.Matching import GreedyMatching
    from PyQuN_Lab.CandidateSearch import NNCandidateSearch
    from PyQuN_Lab.DataLoading import CSVLoader
    from PyQuN_Lab.Similarity import WeightMetric, JaccardIndex, Similarity

    loader = CSVLoader("C:/Users/41766/Desktop/full_subjects/hospitals.csv", StringAttribute)
    out = loader.parse_input()
    model_set = out.model_set
    print(model_set)
    cs = NNCandidateSearch()
    candidates = cs.candidate_search(model_set)
    similarity = WeightMetric(model_set)
    matching = GreedyMatching(candidates, similarity).match(model_set)
    print(matching)
