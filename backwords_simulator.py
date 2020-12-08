import argparse
from typing import TextIO, Union, List, Tuple

from backwords.backwords_trainer import backwords_counter
from lib4mc.MonteCarloLib import MonteCarloLib
from lib4mc.ProbLib import expand_2d
from nwords_simulator import NWordsMonteCarlo


class BackWordsMonteCarlo(NWordsMonteCarlo):
    def __init__(self, training_set: TextIO, splitter: str, start4word: int, skip4word: int,
                 threshold: int, start_chr: str = '\x00', end_chr: str = "\x03", max_gram: int = 256):
        super().__init__(None)
        backwords, words = backwords_counter(training_set, splitter, start_chr, end_chr, start4word, skip4word,
                                             threshold=threshold, max_gram=max_gram)
        self.nwords = expand_2d(backwords)
        self.end_chr = end_chr
        self.words = words
        self.min_len = 4
        self.default_start = start_chr
        self.start_chr = start_chr

    def _get_prefix(self, pwd: Union[List, Tuple], transition: str):
        tar = (self.default_start,)
        for i in range(0, len(pwd)):
            tmp_tar = tuple(pwd[i:])
            if tmp_tar not in self.nwords or \
                    (transition != "" and transition not in self.nwords.get(tmp_tar)[0]):
                continue
            tar = tmp_tar
            break
        return tar


def wrapper():
    cli = argparse.ArgumentParser("Backoff words simulator")
    cli.add_argument("-i", "--input", dest="input", type=argparse.FileType('r'), required=True, help="nwords file")
    cli.add_argument("-t", "--test", dest="test", type=argparse.FileType('r'), required=True, help="testing file")
    cli.add_argument("-s", "--save", dest="save", type=argparse.FileType('w'), required=True,
                     help="save Monte Carlo results here")
    cli.add_argument("--size", dest="size", type=int, required=False, default=100000, help="sample size")
    cli.add_argument("--splitter", dest="splitter", type=lambda x: str(x).replace("\\\\", "\\"), required=False,
                     default="\t", help="how to divide different columns from the input file")
    cli.add_argument("--start4word", dest="start4word", type=int, required=False, default=0,
                     help="start index for words, to fit as much as formats of input. An entry per line. "
                          "We get an array of words by splitting the entry. "
                          "\"start4word\" is the index of the first word in the array")
    cli.add_argument("--skip4word", dest="skip4word", type=int, required=False, default=1,
                     help="there may be other elements between words, such as tags. "
                          "Set skip4word larger than 1 to skip unwanted elements.")
    cli.add_argument("--threshold", dest="threshold", required=False, type=int, default=10,
                     help="grams whose frequencies less than the threshold will be ignored")
    cli.add_argument("--debug-mode", dest="debug_mode", required=False, action="store_true",
                     help="enter passwords and show probability of the password")
    args = cli.parse_args()
    if args.splitter == 'empty':
        args.splitter = ''
    backword_mc = BackWordsMonteCarlo(args.input, splitter=args.splitter, start4word=args.start4word,
                                      skip4word=args.skip4word,
                                      threshold=args.threshold)
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
    wrapper()
