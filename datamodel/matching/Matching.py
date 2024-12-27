
# this class represents the matching object on which the algorithm operates in the third phase of the algorithm.
# it enforces that the collection of matches is mutually exclusive
from typing import Set

from RaQuN_Lab.datamodel.matching.Match import Match
from RaQuN_Lab.datamodel.modelset.Model import Model
from RaQuN_Lab.datamodel.modelset.ModelSet import ModelSet
from RaQuN_Lab.utils.Utils import store_obj, load_obj


class Matching:
    def __init__(self, matches: Set['Match'] = None) -> None:
        if matches is None:
            matches = set()
        self.matches = matches

    def get_matches(self):
        return self.matches

    def clone(self) -> 'Matching':
        li = self.matches.copy()
        return Matching(li)

    def get_match_by_element(self, ele:'Element') -> 'Match':
        for m in self.matches:
            if ele in m:
                return m
        raise ValueError("element is not contained in the matching")

    # merges a collection of matches
    def merge_matches(self, *matches:'Match', do_check:bool = True) -> None:
        for match in matches:
            self.matches.remove(match)
        self.matches.add(Match.merge_matches(*matches, do_check=do_check))

    def __repr__(self):
        matches_preview = ', '.join([repr(match) for match in list(self.matches)[:3]])
        return f"Matching(num_matches={len(self.matches)}, matches=[{matches_preview}]...)"

    #if match is None check if every match in the matching mutual exclusive
    #otherwise check if match is mutual exclusive to all matches in the matching
    def is_mutual_exclusive(self, match = None):
        if match is None:
            for m in self:
                for e in m:
                    for m2 in self:
                        if m == m2:
                            continue
                        if e in m2:
                            return False
            return True
        else:
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
    def trivial_matching(model_set:'ModelSet') -> 'Matching':
        matching = Matching()
        matching.matches = set()
        for element in model_set.get_elements():
            matching.matches.add(Match({element}))
        return matching

    def __iter__(self):
        return iter(self.matches)

    def store(self, path: str) -> None:
        store_obj(self.compress(),path)

    @staticmethod
    def load(path: str) -> 'Matching':
        return load_obj(path).decompress()

    def compress(self):
        return CompressedMatching(self)

    def num_elements(self):
        return sum([len(m) for m in self])

    def num_models(self) -> int:
        seen = set()
        for m in self:
            seen.update([e.get_model_id() for e in m])
        return len(seen)

    def create_modelset(self) -> 'ModelSet':
        model_id_models_map = {}
        for m in self:
            for e in m:
                model = model_id_models_map.setdefault(e.get_model_id(),Model(e.get_model_id()))
                model.add_element(e)
        return ModelSet(set(model_id_models_map.values()))




#Matching reduced to ids of elements instead of entire element objects
class CompressedMatching:
    def __init__(self, matching: Matching):
        self.matches = set()
        for m in matching:
            cur_m = set()
            for ele in m:
                cur_m.add(ele.get_id())
            self.matches.add(frozenset(cur_m))

    def decompress(self, dataset_path: str, strategy : 'Strategy') -> Matching:
        ret = Matching()
        ze_model = strategy.get_data_loader(dataset_path).read_file(dataset_path).get_data_model()
        for m in self.matches:
            cur_m = set()
            for ele_id in m:
                cur_m.add(ze_model.get_by_id(ele_id))
            ret.add_match(cur_m,False)
        return ret



