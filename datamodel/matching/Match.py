import uuid
from typing import Set, Iterable


class MatchViolation(Exception):
    def __init__(self):
        super().__init__("the added elements cause a match violation \n" +
                         ", since atleast 2 elements belong to the same model.\n")


class Match:
    def __init__(self, elements:Set['Element'] = set(), do_check:bool = True) -> None:
        self.id = uuid.uuid4()
        self.eles = set()
        if do_check and not self.are_valid(elements):
            raise MatchViolation()
        self.eles = elements
        self.do_check = do_check

    def get_id(self) -> int:
        return self.id

    def __len__(self):
        return len(self.eles)

    def __eq__(self, other):
        return self.id == other.get_id()

    def __hash__(self):
        return hash(self.id)

    def __iter__(self):
        return iter(self.eles)

    def __repr__(self):
        elements_preview = ', '.join([repr(ele) for ele in list(self.eles)[:3]])
        return f"Match(id={self.id}, num_elements={len(self.eles)}, elements=[{elements_preview}]...)"


    def get_elements(self) -> Set['Element']:
        return self.eles

    def add_elements(self, *elements:'Element') -> None:

        if self.do_check and not self.are_valid(elements):
            raise MatchViolation()
        eles_copy = self.eles.copy()
        eles_copy.update(elements)
        return Match(eles_copy, self.do_check)


    def are_valid(self, elements: Iterable['Element']) -> bool:

        seen_eles = set()

        for ele in elements:
            if ele in seen_eles:
                return False
            seen_eles.add(ele)

            for ele2 in self.eles:
                if ele.get_model_id() == ele2.get_model_id():
                    return False
            for ele2 in elements:
                if ele != ele2 and ele.get_model_id() == ele2.get_model_id():
                    return False

        return True

    @staticmethod
    def is_disjoint(m1: 'Match', m2: 'Match') -> bool:
        for e1 in m1.get_elements():
            if e1 in m2.get_elements():
                return False

        return True

    @staticmethod
    def valid_merge(*matches:'Match') -> bool:
        #check if two matches are the same and return false in this case
        for i in range(len(matches)):
            for j in range(i + 1, len(matches)):
                if matches[i] == matches[j]:
                    return False
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

