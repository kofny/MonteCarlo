"""
Simulator for N Words
"""
import argparse
import copy
import sys
from typing import TextIO, List, Union, Tuple

from lib4mc.MonteCarloLib import MonteCarloLib
from lib4mc.MonteCarloParent import MonteCarlo
from lib4mc.ProbLib import expand_2d, pick_expand
from nwords.nwords_trainer import nwords_counter


class NWordsMonteCarlo(MonteCarlo):
    def __init__(self, training_set: Union[TextIO, None], n: int = 2, splitter: str = ' ', start4word: int = 0,
                 skip4word: int = 1,
                 end_chr: str = "\x03"):
        if training_set is None:
            return
        nwords, words = nwords_counter(training_set, n, splitter, end_chr, start4word, skip4word)
        self.nwords = expand_2d(nwords)
        self.__n = n
        self.words = words
        self.end_chr = end_chr
        self.min_len = 4
        print("Preparing Done", file=sys.stderr)
        pass

    def _get_prefix(self, pwd: Union[List, Tuple]):
        if len(pwd) < self.__n:
            return tuple(pwd)
        else:
            return tuple(pwd[1 - self.__n:])

    def __structures(self, pwd, possible_list: List[List], container: List, probabilities: List, target_len: int):

        for index in range(1, len(pwd) + 1):
            left = pwd[0:index]
            if left in self.words:
                prev = self._get_prefix(container)
                if prev in self.nwords and left in self.nwords.get(prev)[0]:
                    container.append(left)
                    probabilities.append(self.nwords.get(prev)[0].get(left))
                else:
                    continue
                if len("".join(container)) == target_len:
                    possible = [(c, p) for c, p in zip(copy.deepcopy(container), copy.deepcopy(probabilities))]
                    possible_list.append(possible)
                self.__structures(pwd[index:], possible_list, container, probabilities, target_len)
                container.pop()
                probabilities.pop()

    def calc_ml2p(self, pwd: str) -> float:
        possible_list = []
        self.__structures(pwd + self.end_chr, possible_list, [], [], len(pwd) + len(self.end_chr))
        probabilities = []
        for possible in possible_list:
            prob = sum([self.minus_log2(p) for _, p in possible])
            probabilities.append(prob)
        if len(probabilities) == 0:
            return self.minus_log2(sys.float_info.min)
        else:
            min_prob = min(probabilities)
            return min_prob

    def sample1(self) -> (float, str):
        pwd = tuple()
        prob = .0
        pwd_len = 0
        while True:
            tar = self._get_prefix(pwd)
            p, addon = pick_expand(self.nwords.get(tar))
            prob += self.minus_log2(p)
            if addon == self.end_chr:
                if pwd_len >= self.min_len:
                    break
                else:
                    pwd = tuple()
                    prob = .0
                    pwd_len = 0
                    continue
            _tmp = list(pwd)
            _tmp.append(addon)
            pwd = tuple(_tmp)
            pwd_len += len(addon)
            if pwd_len >= 256:
                pwd = tuple()
                prob = .0
                pwd_len = 0
        return prob, "".join(pwd)


if __name__ == '__main__':
    cli = argparse.ArgumentParser("N words simulator")
    cli.add_argument("-i", "--input", dest="input", type=argparse.FileType('r'), required=True, help="nwords file")
    cli.add_argument("-t", "--test", dest="test", type=argparse.FileType('r'), required=True, help="testing file")
    cli.add_argument("-s", "--save", dest="save", type=argparse.FileType('w'), required=True,
                     help="save Monte Carlo results here")
    cli.add_argument("-n", "--ngram", dest="ngram", type=int, required=False, default=2, choices=[2, 3, 4, 5, 6],
                     help="ngram")
    cli.add_argument("--size", dest="size", type=int, required=False, default=100000, help="sample size")
    cli.add_argument("--splitter", dest="splitter", type=str, required=False, default="\t",
                     help="splitter, set empty if splitting the password char by char")
    cli.add_argument("--start4word", dest="start4word", type=int, required=False, default=0,
                     help="start index for words")
    cli.add_argument("--skip4word", dest="skip4word", type=int, required=False, default=1,
                     help="step between two words")
    args = cli.parse_args()
    if args.splitter == 'empty':
        args.splitter = ''
    nword_mc = NWordsMonteCarlo(args.input, splitter=args.splitter, n=args.ngram, start4word=args.start4word,
                                skip4word=args.skip4word)
    ml2p_list = nword_mc.sample(size=args.size)
    mc = MonteCarloLib(ml2p_list)
    scored_testing = nword_mc.parse_file(args.test)
    ranked = mc.ml2p_iter2gc(minus_log_prob_iter=scored_testing)
    mc.write2(args.save)
    pass
