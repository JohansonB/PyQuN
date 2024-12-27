import random
from typing import Set, Union, Iterable, Tuple

from RaQuN_Lab.datamodel.DataModel import DataModel
from RaQuN_Lab.datamodel.modelset.Model import Model
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.vectorization.VectorizedModelSet import \
    VectorizedModelSet
from RaQuN_Lab.utils.Stopwatch import Stopwatch


class ModelSet(DataModel):

    def __init__(self, models:Set['Model'] = None, id_dictionary:bool = True) -> None :
        if models is None:
            models = set()
        else:
            if not isinstance(models, set):
                raise ValueError(f"Attributes must be initialized with a set. Got {type(models)} instead.")

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


    def num_attributes(self) -> int:
        atts = set()
        for e in self.__elements:
            atts.update(e.attributes)

        return len(atts)



    def __iter__(self):
        return iter(self.models)

    def __len__(self):
        return len(self.models)

    def __repr__(self):
        return f"ModelSet(num_models={len(self.models)}, num_elements={len(self.get_elements())})"

    def get_models(self) -> Set[Model]:
        return self.models

    def get_subset(self, models: Union[Iterable[int], int]) -> 'ModelSet':
        if isinstance(models, int):
            li = list(self.models)
            random.shuffle(li)
            models = li[:models]
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
            li = list(m.get_elements())
            random.shuffle(li)
            m.set_elements(li[:cur_len])
        return ze_copy







    def _innit_id_map(self) -> None:
        for model in self.models:
            self.id_map[model.get_id()] = model
            for ele in model:
                self.id_map[ele.get_id()] = ele

    def _linear_id_search(self, id:int) -> Union[Model, 'Element']:
        for model in self.models:
            if model.get_id() == id:
                return model
            for ele in model:
                if ele.get_id() == id:
                    return ele
        raise KeyError(f"Id '{id}' was not found in the ModelSet.")

    # returns the model/element corresponding to an id
    def get_by_id(self, id: int) -> Union[Model, 'Element']:
        if self.use_dictionary:
            if not self.id_map:
                self._innit_id_map()
            if id not in self.id_map:
                raise KeyError(f"Id '{id}' was not found in the ModelSet.")
            else:
                return self.id_map[id]
        else:
            return self._linear_id_search(id)

    def get_elements(self) -> Set['Element']:
        return self.__elements

    def notify_ele_add(self, ele:'Element') -> None:
        self.__elements.add(ele)
        if self.use_dictionary:
            self.id_map[ele.get_id()] = ele

    def notify_ele_del(self, ele:'Element') -> None:
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


    def vectorize(self, vectorizer:'Vectorizer', reduction: 'DimensionalityReduction' = None) -> 'VectorizedModelSet':
        return VectorizedModelSet(self, vectorizer, reduction)

    def timed_vectorize(self, vectorizer:'Vectorizer', reduction: 'DimensionalityReduction' = None) -> Tuple[VectorizedModelSet, Stopwatch]:
        s = Stopwatch()
        s.start_timer("vectorization")
        ret = VectorizedModelSet(self, vectorizer, reduction)
        s.stop_timer("vectorization")
        return ret, s
