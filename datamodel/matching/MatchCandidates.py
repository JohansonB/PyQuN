
# this class represents the output of phase 2 of the RaQuN algorithm.
# mutually exclusive
import random
from collections import OrderedDict


class MatchCandidates:
    def __init__(self, index : int = 0) -> None:
        self.index = index
        self.is_sorted = True
        self.is_filtered = True
        self.matches = []

    def __iter__(self):
        return iter(self.matches)

    def add_matches(self, *matches:'Match') -> None:
        self.matches.extend(matches)
        if len(matches) > 0:
            self.is_sorted = False
            self.is_filtered = False

    def get_index(self, index : int):
        return self.matches[index]


    # removes all candidates whoms similarity lies below a certain threshold
    def filter(self, similarity:'Similarity', threshold:float) -> None:
        if self.is_filtered:
            return
        self.matches = [x for x in self.matches if similarity.similarity(x) >= threshold]
        self.is_filtered = True

    # sort candidates matches accoriding to a similarity fucntion
    def sort(self, similarity:'Similarity'):
        sorted(self.matches, key=lambda x: similarity.similarity(x), reverse=True)
        self.is_sorted = True

    # returns the first match of the match candidates, should be called after sorting the matches to get the candidate
    # with the highest similarity
    def head(self) -> 'Match':
        return self.matches[self.index]

    def pop(self) -> 'Match':
        self.index += 1
        return self.matches[self.index - 1]

    def is_empty(self) -> bool:
        return len(self.matches) == self.index


    def __repr__(self):
        all_matches = "\n".join([repr(match) for match in self.matches])
        return f"MatchCandidates(num_matches={len(self.matches)}, is_sorted={self.is_sorted}, matches=\n{all_matches})"

    def shuffle(self, perc: float):
        tot_proc = int(len(self.matches) * perc)
        if tot_proc <= 0:
            return

        first_ms = self.matches[:tot_proc]
        random.shuffle(first_ms)
        self.matches = first_ms.extend(self.matches[tot_proc:])

