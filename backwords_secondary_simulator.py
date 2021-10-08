import argparse
import pickle
import sys
from typing import BinaryIO

from backwords.backwords_secondary_trainer import freq2prob
from backwords_simulator import BackWordsMonteCarlo
from lib4mc.MonteCarloLib import MonteCarloLib
from lib4mc.ProbLib import expand_2d


class BackWordsSecondaryMonteCarlo(BackWordsMonteCarlo):
    def __init__(self, model: BinaryIO, max_iter: int = 10 ** 100):
        super().__init__(None)
        backwords, words, config = pickle.load(model)
        backwords = freq2prob(backwords, config['threshold'])
        self.nwords = expand_2d(backwords)
        self.end_chr = config['end_chr']
        self.words = words
        self.min_len = 4
        self.default_start = config['start_chr']
        self.start_chr = config['start_chr']
        self.max_iter = max_iter


def wrapper():
    cli = argparse.ArgumentParser("Backoff words simulator")
    cli.add_argument("-m", "--model", dest="model", type=argparse.FileType('rb'), required=True, help="trained model")
    cli.add_argument("-t", "--test", dest="test", type=argparse.FileType('r'), required=True, help="testing file")
    cli.add_argument("-s", "--save", dest="save", type=argparse.FileType('w'), required=True,
                     help="save Monte Carlo results here")
    cli.add_argument("--size", dest="size", type=int, required=False, default=100000, help="sample size")
    cli.add_argument("--debug-mode", dest="debug_mode", required=False, action="store_true",
                     help="enter passwords and show probability of the password")
    cli.add_argument("--max-iter", dest="max_iter", required=False, default=10 ** 20, type=int,
                     help="max iteration when calculating the maximum probability of a password")
    args = cli.parse_args()
    backword_mc = BackWordsSecondaryMonteCarlo(args.model, max_iter=args.max_iter)
    if args.debug_mode:
        usr_i = ""
        while usr_i != "exit":
            usr_i = input("type in passwords: ")
            prob = backword_mc.calc_ml2p(usr_i)
            print(prob)
        return
    ml2p_list = backword_mc.sample(size=args.size)
    mc = MonteCarloLib(ml2p_list)
    scored_testing = backword_mc.parse_file(args.test)
    mc.ml2p_iter2gc(minus_log_prob_iter=scored_testing)
    mc.write2(args.save)


if __name__ == '__main__':
    try:
        wrapper()
    except KeyboardInterrupt:
        print("You canceled the process", file=sys.stderr)
        sys.exit(-1)
