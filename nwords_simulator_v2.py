import argparse
import sys
from typing import TextIO

from lib4mc.MonteCarloLib import MonteCarloLib
from lib4mc.MonteCarloParent import MonteCarlo
from lib4mc.ProbLib import expand_2d, pick_expand
from nwords_v2.nwords_trainer_v2 import nwords_counter


class NWords2MonteCarlo(MonteCarlo):
    def __init__(self, training_set: TextIO, n: int, end_chr: str = "\x03"):
        nwords, words = nwords_counter(training_set, n, end_chr)
        self.__nwords = expand_2d(nwords)
        self.__n = n
        self.__words = words
        self.end_chr = end_chr
        self.__word_max_len = max(words.values())
        pass

    def __dfs(self, pwd: str, min_ml2prob: float, container: str, probability: float, target_len: int):
        for index in range(1, len(pwd) + 1, 1):
            left = pwd[0:index]
            len_container = len(container)
            if len_container < self.__n:
                prev = container
            else:
                prev = container[1 - self.__n:]
            if prev in self.__nwords and left in self.__nwords.get(prev)[0]:
                # probabilities.append(self.__nwords.get(prev)[0].get(left))
                prob = self.__nwords.get(prev)[0].get(left)
                n_prob = probability + prob
                if len(container) + index == target_len:
                    if n_prob < min_ml2prob:
                        min_ml2prob = n_prob
                self.__dfs(pwd[index:], min_ml2prob, container + left, probability + prob, target_len)
                # probabilities.pop()

    def calc_ml2p(self, pwd: str) -> float:
        min_ml2prob = self.minus_log2(sys.float_info.min)
        if len(pwd) <= 30:
            self.__dfs(pwd + self.end_chr, min_ml2prob, "", 0, len(pwd) + len(self.end_chr))
        else:
            n_pwd = pwd + self.end_chr
            log_prob = 0
            for i, c in enumerate(n_pwd):
                if i < self.__n:
                    prefix = n_pwd[:i]
                else:
                    prefix = n_pwd[i - self.__n + 1:i]
                addons = self.__nwords.get(prefix, [{}])[0]
                if c not in addons:
                    return sys.maxsize
                prob = addons.get(c)
                log_prob += self.minus_log2(prob)
            if min_ml2prob > log_prob:
                min_ml2prob = log_prob
        return min_ml2prob

    def sample1(self) -> (float, str):
        pwd = ""
        prob = .0
        pwd_len = 0
        while True:
            if pwd_len < self.__n:
                p, addon = pick_expand(self.__nwords.get(pwd))
            else:
                p, addon = pick_expand(self.__nwords.get(pwd[1 - self.__n:]))
            prob += self.minus_log2(p)
            if addon == self.end_chr:
                if pwd_len > 3:
                    break
                else:
                    pwd = ""
                    prob = .0
                    continue
            pwd += addon
            pwd_len += len(addon)
            if pwd_len >= 256:
                pwd = ""
                prob = .0
        return prob, pwd


def test():
    nwmc = NWords2MonteCarlo(open("/home/cw/Documents/Experiments/SegLab/NWords/csdn-rded.txt"), 4)
    ml2p_list = nwmc.sample()
    mc = MonteCarloLib(ml2p_list)
    ipt = ""
    while ipt != "exit":
        ipt = input("Enter password: ")
        ml2p = nwmc.calc_ml2p(ipt)
        print(ml2p)
        print(mc.ml2p2rank(ml2p))
    pass


def main():
    cli = argparse.ArgumentParser("NWords v2")
    cli.add_argument("-f", "--file", dest="training", required=True, type=argparse.FileType("r"), help="training set")
    cli.add_argument("-t", "--target", dest="testing", required=True, type=argparse.FileType("r"), help="testing set")
    cli.add_argument("-s", "--save", dest="save", required=False, default=sys.stdout, type=argparse.FileType("w"),
                     help="save results")
    args = cli.parse_args()
    nwmc = NWords2MonteCarlo(args.training, 4)
    ml2p_list = nwmc.sample()
    mc = MonteCarloLib(ml2p_list)
    scored_testing = nwmc.parse_file(args.testing)
    mc.ml2p_iter2gc(minus_log_prob_iter=scored_testing)
    mc.write2(args.save)


if __name__ == '__main__':
    main()
