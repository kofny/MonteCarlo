import argparse

from nwords.nwords_enumerator import enumerator
from nwords.nwords_trainer import nwords_counter


def wrapper():
    cli = argparse.ArgumentParser("Backoff Enumerator")
    cli.add_argument("-f", '--pwd-file', dest="pwd_file", required=True, type=argparse.FileType('r'),
                     help="Training file")
    cli.add_argument("-n", '--ngram', dest="ngram", required=True, type=int,
                     help="ngram of the model")
    cli.add_argument('--splitter', required=True, dest="splitter", type=str, choices=['empty', 'space', 'tab'],
                     help="split the line into several pieces, each piece is a sub-word")
    cli.add_argument("--start", required=False, dest="start4words", type=int,
                     help="if pieces look like <others> <sub-word> <sub-word>, set start4words to 1 to skip index 0")
    cli.add_argument("--step", required=False, dest="skip4words", type=int,
                     help="sub-word is from index of `start`, "
                          "then we move forward `step` steps to obtain next sub-word")
    cli.add_argument("-p", '--min-prob', dest="min_prob", required=True, type=float,
                     help="Minimal probability that a password candidate should have")
    cli.add_argument("-l", '--min-length', dest='min_len', required=False, type=int, default=4,
                     help="Minimal length of password candidates")
    cli.add_argument("-s", '--save', dest="f_save", required=True, type=argparse.FileType('w'),
                     help="save password candidates here")
    args = cli.parse_args()
    splitter_map = {"empty": "", "space": " ", "tab": "\t"}
    splitter = splitter_map[args.splitter]
    nwords_dict_float, _ = nwords_counter(
        args.pwd_file, n=args.ngram, splitter=splitter,
        end_chr='\x00', start_chr='\x03',
        start4words=args.start4words, skip4words=args.skip4words)
    enumerator(nwords_dict_float, threshold=args.min_prob, end_chr='\x00', start_chr='\x03',
               min_len=args.min_len, f_save=args.f_save, order=args.ngram - 1)
    pass


if __name__ == '__main__':
    wrapper()
