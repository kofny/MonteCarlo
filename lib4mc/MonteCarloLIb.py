import bisect
from math import log2, ceil
from typing import List, Tuple, Iterator, TextIO

import numpy
from tqdm import tqdm


class MonteCarloLib:
    def __init__(self, minus_log_prob_list: List[float]):
        self.__minus_log_prob_list = minus_log_prob_list
        minus_log_probs, positions = self.__gen_rank_from_minus_log_prob()
        self.__minus_log_probs = minus_log_probs
        self.__positions = positions
        self.__gc = None
        pass

    def __gen_rank_from_minus_log_prob(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        calculate the ranks according to Monte Carlo method
        :return: minus_log_probs and corresponding ranks
        """
        minus_log_probs = numpy.fromiter(self.__minus_log_prob_list, float)
        minus_log_probs.sort()
        logn = log2(len(minus_log_probs))
        positions = (2 ** (minus_log_probs - logn)).cumsum()
        return minus_log_probs, positions
        pass

    def minus_log_prob2rank(self, minus_log_prob):
        idx = bisect.bisect_right(self.__minus_log_probs, minus_log_prob)
        return self.__positions[idx - 1] if idx > 0 else 1

    def mlps2gc(self, minus_log_prob_iter: List[Tuple[str, int, float]], need_resort: bool = False) \
            -> List[Tuple[str, float, int, int, int, float]]:
        """

        :param need_resort:
        :param minus_log_prob_iter: sorted
        :return:
        """
        if need_resort:
            minus_log_prob_iter = sorted(minus_log_prob_iter, key=lambda x: x[2])
        gc = []
        prev_rank = 0
        cracked = 0
        total = sum([a for _, a, _ in minus_log_prob_iter])
        for pwd, appearance, mlp in tqdm(minus_log_prob_iter, desc="Ranking: "):
            idx = bisect.bisect_right(self.__minus_log_probs, mlp)
            rank = ceil(max(self.__positions[idx - 1] if idx > 0 else 1, prev_rank + 1))
            cracked += appearance
            prev_rank = rank
            gc.append((pwd, mlp, appearance, rank, cracked, cracked / total * 100))
        self.__gc = gc
        return gc

    def write2(self, fd: TextIO):
        if not fd.writable():
            raise Exception(f"{fd.name} is not writable")
        if self.__gc is None:
            raise Exception(f"run mlps2gc before invoke this method")
        for pwd, mlp, appearance, rank, cracked, cracked_ratio in tqdm(self.__gc, desc="Saving: "):
            fd.write(f"{pwd}\t{mlp:.8f}\t{appearance}\t{rank}\t{cracked}\t{cracked_ratio:5.2f}\n")
        self.__gc = None
        pass
