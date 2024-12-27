
# this class represents the output of phase 2 of the RaQuN algorithm.
# the matching Candidates consists of a ordered list of valid matches, without the requirement that the matches need to be
# mutually exclusive
import random
from collections import OrderedDict


class MatchCandidates:
    def __init__(self) -> None:
        self.is_sorted = True
        self.is_filtered = True
        self.matches = OrderedDict()

    def __iter__(self):
        return iter(self.matches.keys())

    def add_matches(self, *matches:'Match') -> None:
        for match in matches:
            self.matches[match] = None
        if len(matches) > 0:
            self.is_sorted = False
            self.is_filtered = False

    # removes all candidates whoms similarity lies below a certain threshold
    def filter(self, similarity:'Similarity', threshold:float) -> None:
        if self.is_filtered:
            return
        keys_to_remove = [key for key, value in self.matches.items() if similarity.similarity(key) < threshold]
        for key in keys_to_remove:
            del self.matches[key]
        self.is_filtered = True

    # sort candidates matches accoriding to a similarity fucntion
    def sort(self, similarity:'Similarity'):
        if self.is_sorted:
            return
        self.matches = OrderedDict(
            sorted(self.matches.items(), key=lambda item: similarity.similarity(item[0]), reverse=True))
        self.is_sorted = True

    # returns the first match of the match candidates, should be called after sorting the matches to get the candidate
    # with the highest similarity
    def head(self) -> 'Match':
        if self.is_sorted:
            return next(iter(self.matches))
        else:
            raise Exception("you need to call the sort function before calling this method")

    def pop(self) -> 'Match':
        if self.is_sorted:
            ne = self.head()
            del self.matches[ne]
            return ne
        else:
            raise Exception("you need to call the sort function before calling this method")

    def is_empty(self) -> bool:
        return len(self.matches) == 0


    def __repr__(self):
        all_matches = "\n".join([repr(match) for match in self.matches])
        return f"MatchCandidates(num_matches={len(self.matches)}, is_sorted={self.is_sorted}, matches=\n{all_matches})"

    def shuffle(self, perc: float):
        tot_proc = int(len(self.matches) * perc)
        if tot_proc <= 0:
            return

        keys = list(self.matches.keys())
        first_keys = keys[:tot_proc]

        shuffle_items = [(key, self.matches[key]) for key in first_keys]
        random.shuffle(shuffle_items)

        shuffled_dict = OrderedDict()
        for key, value in shuffle_items:
            shuffled_dict[key] = value
        for key in keys[tot_proc:]:
            shuffled_dict[key] = self.matches[key]

        self.matches = shuffled_dict

