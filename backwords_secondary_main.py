"""
This file is the portal of secondary training for backwords
"""
import argparse
import json
import math
import os.path
import pickle
import random
import sys
from typing import List, Tuple

from backwords.backwords_secondary_trainer import backwords_counter
from backwords_secondary_simulator import BackWordsSecondaryMonteCarlo
from lib4mc.FileLib import wc_l
from lib4mc.MonteCarloLib import MonteCarloLib


def secondary_cracker(backwords, words, config,
                      func_threshold: Tuple[int, int], **kwargs):
    save_in_folder = kwargs['save']
    tag = kwargs['tag']
    nwords_dict, _words = backwords_counter(
        nwords_list=kwargs['training'], splitter=kwargs['splitter'], start_chr=config['start_chr'],
        end_chr=config['end_chr'],
        start4words=kwargs['start4words'], step4words=kwargs['skip4words'], max_gram=kwargs['max_gram'],
        nwords_dict=backwords, words=words, threshold=kwargs['threshold'])
    fmodel = os.path.join(save_in_folder, f"model-to-crack-{tag}.pickle")
    with open(fmodel, 'wb') as fd:
        sign = kwargs['sign']
        config['training_list'].append(f"{sign}")
        pickle.dump((nwords_dict, words, config), file=fd)
    backword_mc = BackWordsSecondaryMonteCarlo((nwords_dict, _words, config), max_iter=kwargs['max_iter'])
    # Note: this part is to "generate" some guesses and crack passwords in the testing dataset
    #
    # Besides, here we allow the user to provide a list which holds the sampled passwords
    # Then, we calculate the intersection of the sampled passwords and the testing dataset to
    # obtain the cracked passwords
    using_sample_attack = kwargs['using_sample_attack']
    sampled_pwds = None
    if using_sample_attack:
        sampled_pwds = {}
    ml2p_list = backword_mc.sample(size=kwargs['size'], sampled_pwds=sampled_pwds)
    if using_sample_attack:
        f_samples = os.path.join(save_in_folder, f"samples-{tag}.txt")
        with open(f_samples, 'w') as fout_samples:
            sidx = 1
            for pwd, (prob, cnt) in sorted(sampled_pwds.items(), key=lambda x: x[1][0]):
                fout_samples.write(f"{pwd}\t{prob:.8f}\t{cnt}\n")
                sampled_pwds[pwd] = sidx
                sidx += cnt
        pass
    mc = MonteCarloLib(ml2p_list)
    scored_testing = backword_mc.parse_file(kwargs['testing'], using_component=True)
    gc = mc.ml2p_iter2gc(minus_log_prob_iter=scored_testing)
    secondary_training = []
    fcracked = os.path.join(save_in_folder, f"cracked-{tag}.txt")
    already_cracked = kwargs['already_cracked']
    cum: List[Tuple[str, float, int, int]] = kwargs['cum']
    with open(fcracked, 'w') as fout:
        unique, max_gn = 0, 0
        gn_upper_bound, hits_upper_bound = func_threshold
        for pwd, prob, num, gn, cracked, ratio in gc:
            _pwd = kwargs['splitter'].join(pwd)
            if _pwd in already_cracked:
                continue
            valid1 = (using_sample_attack and _pwd in sampled_pwds)
            if valid1:
                gn = sampled_pwds[_pwd]
            if valid1 or (not using_sample_attack and (gn < gn_upper_bound and unique < hits_upper_bound)):
                unique += 1
                max_gn = max(max_gn, gn)
                secondary_training.extend([_pwd] * num)
                cum.append((_pwd, prob, num, gn))
                fout.write(f"{_pwd}\t{prob:.8f}\t{num}\t{gn}\n")
                already_cracked.add(_pwd)
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
    return nwords_dict, _words, config, secondary_training, max_gn


def wrapper():
    cli = argparse.ArgumentParser('Backwords secondary main')
    cli.add_argument("-i", "--training", dest="training", type=argparse.FileType('r'), required=True,
                     help="The training file, each password a line")
    cli.add_argument("-t", "--testing", dest="testing", type=argparse.FileType('r'), required=True,
                     help="The testing file, each password a line")
    cli.add_argument("-s", "--save", dest="save", required=True, type=str,
                     help='A folder, results will be saved in this folder')
    cli.add_argument("--strategy", dest="strategy", required=True, type=str, nargs="+",
                     # choices=['guesses', 'hits', 'samples'],
                     help='`guesses <guesses1> <guesses2> ...` means guess number thresholds, '
                          '`hits <cracked1> <cracked2>` means cracked passwords, '
                          '`auto_hits <factor> <base> <termination>` means auto generate '
                          '<cracked1 = factor * base> <cracked2> <cracked2 = factor ** 2 * base>'
                          '`samples <rounds>` means the number of iterations of'
                          'Monte Carlo simulation')
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
    strategy_value = args.strategy
    strategy = strategy_value[0]
    permits = {'guesses', 'hits', 'samples', 'auto_hits'}
    if strategy not in permits:
        print(f"strategy should be one of `{', '.join(permits)}`", file=sys.stderr)
        return
    if len(strategy_value) < 2:
        print(f"strategy should have at least 2 values", file=sys.stderr)
        return

    using_sample_attack, signs = False, []
    upper_bound, hits_upper_bound = 10 ** 14, 10 ** 14
    func_thresholds = []
    if strategy == 'guesses':
        print(f"using guesses", file=sys.stderr)
        values = strategy_value[1:]
        values = [int(v) for v in values]
        for i, v in enumerate(values):
            func_thresholds.append((v, hits_upper_bound))
            signs.append(f"guesses-{v:,}")
        pass
    elif strategy == 'hits':
        print(f"using hits", file=sys.stderr)
        values = strategy_value[1:]
        values = [int(v) for v in values]
        for i, v in enumerate(values):
            func_thresholds.append((upper_bound, v))
            signs.append(f"hits-{v:,}")
        pass
    elif strategy == 'auto_hits':
        print(f"using auto_hits", file=sys.stderr)
        factor, base, termination = int(strategy_value[1]), int(strategy_value[2]), int(strategy_value[3])
        end = math.ceil(math.log(termination / max(base, 1)) / math.log(max(factor, 1)))
        for i, v in enumerate(range(1, end)):
            nv = (factor ** v) * base

            func_thresholds.append((upper_bound, nv))
            signs.append(f"auto_hits-{v:,}")
    else:
        print(f"using samples", file=sys.stderr)
        v = int(strategy_value[1])
        func_thresholds = [(upper_bound, hits_upper_bound) for _ in range(v)]
        signs = [f"samples-{args.size}" for _ in range(v)]
        using_sample_attack = True
        pass
    rounds = len(func_thresholds)
    splitter_map = {'empty': '', 'space': ' ', 'tab': '\t'}
    if args.splitter.lower() in splitter_map:
        args.splitter = splitter_map[args.splitter.lower()]
    start_chr, end_chr, training_list = '\x03', '\x00', [args.training.name]
    config = {'start_chr': start_chr, 'end_chr': end_chr, 'max_gram': args.max_gram, 'threshold': args.threshold,
              'training_list': training_list}
    backwords, words = None, None
    training = args.training
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    already_cracked = set()

    print(f"We will have {rounds} rounds", file=sys.stderr, end=', ')
    cums: List[List[Tuple[str, float, int, int]]] = []
    max_guess_numbers = []
    for idx in range(rounds):
        # guess_number_threshold have default value of [args.size, ..., args.size] if it is None
        func_threshold = func_thresholds[idx]
        # Therefore, prior_guesses will always be args.size if `--using-samples`
        print(f"The {idx}-th iteration", file=sys.stderr)
        cum = []
        backwords, words, config, training, max_gn = secondary_cracker(
            backwords, words, config=config,
            func_threshold=func_threshold,
            training=training, splitter=args.splitter,
            start4words=args.start4words, skip4words=args.skip4words,
            max_gram=args.max_gram, size=args.size, max_iter=args.max_iter,
            testing=args.testing, save=args.save, secondary_sample=args.secondary_sample,
            already_cracked=already_cracked, cum=cum,
            threshold=args.threshold, sign=signs[idx],
            using_sample_attack=using_sample_attack, tag=f"iter-{idx}",
        )
        cums.append(cum)
        max_guess_numbers.append(max_gn)
        if max_gn >= upper_bound:
            print(f"Too large guess number reached: {max_gn}, the training process is terminated", file=sys.stderr)
            break
        pass
    backwords, words = backwords_counter(
        training, splitter=args.splitter, start_chr=start_chr, end_chr=end_chr,
        start4words=args.start4words, step4words=args.skip4words, max_gram=args.max_gram,
        nwords_dict=backwords, words=words, threshold=args.threshold
    )
    f_final_model = os.path.join(args.save, "final_model.pickle")
    with open(f_final_model, 'wb') as fout_final_model:
        pickle.dump((backwords, words, config), file=fout_final_model)
    print("Training phase done.", file=sys.stderr)
    backword_mc = BackWordsSecondaryMonteCarlo((backwords, words, config), max_iter=args.max_iter)
    ml2p_list = backword_mc.sample(size=args.size)
    mc = MonteCarloLib(ml2p_list)
    scored_testing = backword_mc.parse_file(args.testing)
    gc = mc.ml2p_iter2gc(minus_log_prob_iter=scored_testing)
    # note that this is the cracked passwords obtained according to the final model
    f_iter_result = os.path.join(args.save, "iter_result.txt")
    with open(f_iter_result, 'w') as fout_iter_result:
        cum = []
        for pwd, prob, num, gn, cracked, ratio in gc:
            fout_iter_result.write(f"{pwd}\t{prob:.8f}\t{num}\t{gn}\t{cracked}\t{ratio:5.2f}\n")
            if pwd not in already_cracked:
                cum.append((pwd, prob, num, gn))
            pass
        cums.append(cum)
        pass
    # note that this is the union of all intermediate results
    # each guess matters in this result file
    f_sectional_result = os.path.join(args.save, "sectional_result.txt")
    with open(f_sectional_result, "w") as fout_sectional_result:
        _cracked = 0
        _total = wc_l(args.testing)
        for gnt, cum in zip([0, *max_guess_numbers], cums):
            for (_pwd, _prob, _n, _gn) in cum:
                _cracked += _n
                _ratio = _cracked / _total * 100
                fout_sectional_result.write(f"{_pwd}\t{_prob:.8f}\t{_n}\t{_gn + gnt}\t{_cracked}\t{_ratio:5.2f}\n")
        pass
    f_config = os.path.join(args.save, "config.json")
    with open(f_config, 'w') as fout_config:
        json.dump(config, fp=fout_config, indent=2)
    args.testing.close()
    pass


if __name__ == '__main__':
    wrapper()
