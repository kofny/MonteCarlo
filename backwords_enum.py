import argparse

from backwords.backwords_enumerator import enumerator
from backwords.backwords_trainer import backwords_counter


def wrapper():
    cli = argparse.ArgumentParser("Backoff Enumerator")
    cli.add_argument("-f", '--pwd-file', dest="pwd_file", required=True, type=argparse.FileType('r'),
                     help="Training file")
    cli.add_argument("-p", '--min-prob', dest="min_prob", required=True, type=float,
                     help="Minimal probability that a password candidate should have")
    cli.add_argument("-l", '--min-length', dest='min_len', required=False, type=int, default=4,
                     help="Minimal length of password candidates")
    cli.add_argument("-s", '--save', dest="f_save", required=True, type=argparse.FileType('w'),
                     help="save password candidates here")
    args = cli.parse_args()
    backwords_dict_float, _ = backwords_counter(args.pwd_file, '', '\x00', '\x03', 0, 1, 10, 3)
    enumerator(backwords_dict_float, args.min_prob, '\x00', '\x03', args.min_len, args.f_save)
    pass


if __name__ == '__main__':
    wrapper()
