"""
Simulator for N Words
"""
from math import log2
from typing import TextIO, List, Tuple

from lib4mc.MonteCarloParent import MonteCarlo
from lib4mc.ProbLib import expand_2d, pick_expand
from nwords.nwords_trainer import nwords_counter


class NWordsMonteCarlo(MonteCarlo):
    def __init__(self, training_set: TextIO, n: int, end_chr: str = "\x03"):
        nwords = nwords_counter(training_set, n, end_chr)
        self.__nwords = expand_2d(nwords)
        self.__n = n
        self.end_chr = end_chr
        pass

    def calc_minus_log_prob(self, pwd: str) -> float:
        pass

    def sample_one(self) -> (float, str):
        pwd = tuple()
        prob = .0
        pwd_len = 0
        while True:
            if pwd_len < self.__n:
                p, addon = pick_expand(self.__nwords.get(pwd))
            else:
                p, addon = pick_expand(self.__nwords.get(tuple(pwd[1 - self.__n:])))
            prob -= log2(p)
            if addon == self.end_chr:
                if pwd_len >= self.__n:
                    break
                else:
                    pwd = tuple()
                    prob = .0
            _tmp = list(pwd)
            _tmp.append(addon)
            pwd = tuple(_tmp)
            pwd_len += len(addon)
            if pwd_len >= 256:
                pwd = ""
                prob = .0
        return prob, "".join(pwd)


if __name__ == '__main__':
    nwmc = NWordsMonteCarlo(open("/home/cw/Documents/Experiments/SegLab/NWords/csdn-rded.txt"), 4)
    print(nwmc.sample_one())
