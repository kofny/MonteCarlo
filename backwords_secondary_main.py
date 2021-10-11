"""
This file is the portal of secondary training for backwords
"""
import argparse
import os.path
import pickle
import random
import sys

from backwords.backwords_secondary_trainer import backwords_counter
from backwords_secondary_simulator import BackWordsSecondaryMonteCarlo
from lib4mc.MonteCarloLib import MonteCarloLib


def secondary_cracker(backwords, words, config, guess_number_threshold, **kwargs):
    save_in_folder = kwargs['save']
    tag = f"{guess_number_threshold:.0e}"
    nwords_dict, _words = backwords_counter(
        nwords_list=kwargs['training'], splitter=kwargs['splitter'], start_chr=config['start_chr'],
        end_chr=config['end_chr'],
        start4words=kwargs['start4words'], step4words=kwargs['skip4words'], max_gram=kwargs['max_gram'],
        nwords_dict=backwords, words=words)
    fmodel = os.path.join(save_in_folder, f"model-to-crack-{tag}.pickle")
    with open(fmodel, 'wb') as fd:
        training = kwargs['training']
        if isinstance(training, list):
            config['training_list'].append(tag)
        else:
            config['training_list'].append(training.name)
        pickle.dump((nwords_dict, words, config), file=fd)
    backword_mc = BackWordsSecondaryMonteCarlo((nwords_dict, _words, config), max_iter=kwargs['max_iter'])
    ml2p_list = backword_mc.sample(size=kwargs['size'])
    mc = MonteCarloLib(ml2p_list)
    scored_testing = backword_mc.parse_file(kwargs['testing'])
    gc = mc.ml2p_iter2gc(minus_log_prob_iter=scored_testing)
    secondary_training = []
    fcracked = os.path.join(save_in_folder, f"cracked-{tag}.txt")
    with open(fcracked, 'w') as fout:
        for pwd, prob, num, gn, _, _ in gc:
            if gn <= guess_number_threshold:
                secondary_training.extend([pwd] * num)
                fout.write(f"{pwd}\t{prob:.8f}\t{num}\t{gn}\n")
        pass
    secondary_sample_size = kwargs['secondary_sample']
    if secondary_sample_size < len(secondary_training):
        print(f"We sample {secondary_sample_size} passwords to perform secondary training in the next round",
              file=sys.stderr)
        fsample = os.path.join(save_in_folder, f"sampled-{tag}.txt")
        secondary_training = random.sample(secondary_training, secondary_sample_size)
        with open(fsample, 'w') as fout:
            for pwd in secondary_training:
                fout.write(f"{pwd}\n")
            pass
    return nwords_dict, _words, config, secondary_training


def wrapper():
    cli = argparse.ArgumentParser('Backwords secondary main')
    cli.add_argument("-i", "--training", dest="training", type=argparse.FileType('r'), required=True,
                     help="The training file, each password a line")
    cli.add_argument("-t", "--testing", dest="testing", type=argparse.FileType('r'), required=True,
                     help="The testing file, each password a line")
    cli.add_argument("-g", "--guess-number-thresholds", dest="guess_number_thresholds", type=int, nargs="+",
                     help="Each threshold refers to a guess number threshold. "
                          "The model will crack passwords under the threshold "
                          "and use the cracked passwords as secondary training file")
    cli.add_argument("-s", "--save", dest="save", required=True, type=str,
                     help='A folder, results will be saved in this folder')
    cli.add_argument("--size", dest="size", type=int, required=False, default=100000, help="sample size")
    cli.add_argument("--secondary-sample", dest="secondary_sample", type=int, required=False, default=10000000000,
                     help="use some of the cracked passwords for secondary training.")
    cli.add_argument("--splitter", dest="splitter", type=str, required=False, default="empty",
                     help="how to divide different columns from the input file, "
                          "set it \"empty\" to represent \'\', \"space\" for \' \', \"tab\" for \'\t\'")
    cli.add_argument("--start4word", dest="start4words", type=int, required=False, default=0,
                     help="start index for words, to fit as much as formats of input. An entry per line. "
                          "We get an array of words by splitting the entry. "
                          "\"start4word\" is the index of the first word in the array")
    cli.add_argument("--skip4word", dest="skip4words", type=int, required=False, default=1,
                     help="there may be other elements between words, such as tags. "
                          "Set skip4word larger than 1 to skip unwanted elements.")
    cli.add_argument("--max-gram", dest="max_gram", required=False, type=int, default=256, help="max gram")
    cli.add_argument("--threshold", dest="threshold", required=False, type=int, default=10,
                     help="grams whose frequencies less than the threshold will be ignored")
    cli.add_argument("--max-iter", dest="max_iter", required=False, default=10 ** 20, type=int,
                     help="max iteration when calculating the maximum probability of a password")
    args = cli.parse_args()
    splitter_map = {'empty': '', 'space': ' ', 'tab': '\t'}
    if args.splitter.lower() in splitter_map:
        args.splitter = splitter_map[args.splitter.lower()]
    start_chr, end_chr, training_list = '\x03', '\x00', []
    config = {'start_chr': start_chr, 'end_chr': end_chr, 'max_gram': args.max_gram, 'threshold': args.threshold,
              'training_list': training_list}
    backwords, words = None, None
    training = args.training
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    for guess_number_threshold in args.guess_number_thresholds:
        print(f"Training Model, obtaining passwords whose guess numbers are less than {guess_number_threshold:.0e}",
              file=sys.stderr)
        backwords, words, config, training = secondary_cracker(
            backwords, words, config=config,
            guess_number_threshold=guess_number_threshold,
            training=training, splitter=args.splitter,
            start4words=args.start4words, skip4words=args.skip4words,
            max_gram=args.max_gram, size=args.size, max_iter=args.max_iter,
            testing=args.testing, save=args.save, secondary_sample=args.secondary_sample,
        )
        pass
    pass


if __name__ == '__main__':
    wrapper()
