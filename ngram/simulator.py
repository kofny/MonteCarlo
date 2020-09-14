import bisect
import os
import pickle
import random
import sys
from collections import defaultdict
from math import log2
from typing import TextIO, BinaryIO, Dict, List, Tuple

import numpy
from tqdm import tqdm

from lib4mc.FileLib import wc_l
from lib4mc.MonteCarloLIb import MonteCarloLib
from lib4mc.MonteCarloParent import MonteCarlo
from ngram.trainer import ngram_counter


def expand(ngram_dict: Dict[str, Dict[str, float]]):
    n_ngram_dict = {}
    for k, addons in ngram_dict.items():
        items = list(addons.keys())
        cumsums = numpy.array(list(addons.values())).cumsum()
        n_ngram_dict[k] = (addons, items, cumsums)
        pass
    return n_ngram_dict
    pass


def pick_expand(expanded: (Dict[str, float], List[str], List[float])):
    addons, items, cumsums = expanded
    total = cumsums[-1]
    idx = bisect.bisect_right(cumsums, random.uniform(0, total))
    item = items[idx]
    return addons.get(item), item


class NGramMonteCarlo(MonteCarlo):
    def __init__(self, pwd_list: TextIO, n: int = 4, end_chr: str = "\x03"):
        print(f"Training set: {os.path.abspath(pwd_list.name)}", file=sys.stderr)
        self.ngram_dict = expand(ngram_counter(pwd_list, n, end_chr))
        self.n = n
        self.end_chr = end_chr
        pass

    def sample(self, size: int = 100000) -> List[float]:
        results = []
        for _ in tqdm(iterable=range(size), desc="Sampling: "):
            prob, pwd = self.sample_one()
            results.append(prob)
        return results

    def calc_minus_log_prob(self, pwd: str):
        n_pwd = pwd + self.end_chr
        log_prob = 0
        for i, c in enumerate(n_pwd):
            if i < self.n:
                prefix = n_pwd[:i]
            else:
                prefix = n_pwd[i - self.n + 1:i]
            addons = self.ngram_dict.get(prefix, [{}])[0]
            if c not in addons:
                return sys.maxsize
            prob = addons.get(c)
            log_prob -= log2(prob)
            # log_prob += self.ngram_dict.get(prefix, [{}])[0].get(c, 10000)
        return log_prob

    def sample_one(self) -> Tuple[float, str]:
        pwd = ""
        prob = .0
        while True:
            if len(pwd) < self.n:
                p, addon = pick_expand(self.ngram_dict.get(pwd))
            else:
                p, addon = pick_expand(self.ngram_dict.get(pwd[1 - self.n:]))
            prob -= log2(p)
            if addon == self.end_chr:
                if len(pwd) > 3:
                    break
                else:
                    pwd = ""
                    prob = .0
                    continue
            pwd += addon
            if len(pwd) >= 256:
                pwd = ""
                prob = .0
            pass
        return prob, pwd

    @classmethod
    def from_pickle(cls, model: BinaryIO):
        obj = cls.__new__(cls)
        obj.n, obj.end_chr, ngram_dict = pickle.load(model)
        obj.ngram_dict = expand(ngram_dict)
        return obj

    def parse_file(self, test: TextIO) -> List[Tuple[str, int, float]]:
        """
        get minus log prob for test set
        :param test: test set
        :return: List of tuple (pwd, appearance, minus log prob)
        """
        line_num = wc_l(test)
        pwd_counter = defaultdict(int)
        for line in tqdm(test, desc="Counting: ", total=line_num):
            line = line.strip("\r\n")
            pwd_counter[line] += 1
        res = []
        for pwd, num in tqdm(pwd_counter.items(), desc="Parsing test: "):
            _mlp = self.calc_minus_log_prob(pwd)
            res.append((pwd, num, _mlp))
        res = sorted(res, key=lambda x: x[2])
        return res

    pass


def main():
    for corpus in ["csdn", "rockyou", "webhost", "dodonew", "xato"]:
        for n in [2, 3, 4, 5]:
            print(f"---------------------{corpus}-{n}gram--------------------------")
            ngram = NGramMonteCarlo(
                pwd_list=open(f"/home/cw/Documents/Experiments/SegLab/Corpora/{corpus}-src.txt"),
                n=n)

            mlps = ngram.sample(1000000)
            ll = ngram.parse_file(
                open(f"/home/cw/Documents/Experiments/SegLab/Corpora/{corpus}-tar.txt"))
            mc = MonteCarloLib(mlps)
            mc.mlps2gc(ll)
            mc.write2(
                open(f"/home/cw/Documents/Experiments/SegLab/SimulatedNGram/{corpus}-{n}gram.txt", "w"))
        pass


if __name__ == '__main__':
    main()
