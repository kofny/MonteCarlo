import argparse
import pickle
import sys

from backwords.backwords_secondary_trainer import backwords_counter


def wrapper():
    cli = argparse.ArgumentParser("Backoff: subword level trainer using secondary training file")
    cli.add_argument("-t", '--training', required=True, type=argparse.FileType('r'), dest='training',
                     help='training file')
    cli.add_argument('-s', '--save', required=True, type=argparse.FileType('wb'), dest='save',
                     help="save trained model here")
    cli.add_argument("-m", '--model', required=False, default=None, type=argparse.FileType('rb'), dest='model',
                     help="we will train the model based on the given `model`")
    cli.add_argument("--splitter", dest="splitter", type=str, required=False, default="\t",
                     help="how to divide different columns from the input file, set it \"empty\" to represent \'\'")
    cli.add_argument("--start4word", dest="start4word", type=int, required=False, default=0,
                     help="start index for words, to fit as much as formats of input. An entry per line. "
                          "We get an array of words by splitting the entry. "
                          "\"start4word\" is the index of the first word in the array")
    cli.add_argument("--skip4word", dest="skip4word", type=int, required=False, default=1,
                     help="there may be other elements between words, such as tags. "
                          "Set skip4word larger than 1 to skip unwanted elements.")
    cli.add_argument("--threshold", dest="threshold", required=False, type=int, default=10,
                     help="grams whose frequencies less than the threshold will be ignored")
    cli.add_argument("--max-gram", dest="max_gram", required=False, type=int, default=256, help="max gram")
    args = cli.parse_args()
    nwords_dict = None
    if not args.save.writable():
        print(f"{args.save.name} is not writable", file=sys.stderr)
        sys.exit(1)
    start_chr, end_chr = '\x03', '\x00'
    if args.model is not None:
        nwords_dict, _, config = pickle.load(args.model)
        start_chr, end_chr = config['start_chr'], config['end_chr']
    nwords_dict, words = backwords_counter(
        nwords_list=args.training, splitter=args.splitter, start_chr=start_chr, end_chr=end_chr,
        start4words=args.start4word, step4words=args.skip4words, max_gram=args.max_gram, nwords_dict=nwords_dict)

    pickle.dump(
        (
            nwords_dict,
            words,
            {'start_chr': start_chr, 'end_chr': end_chr, 'max_gram': args.max_gram}
        ), file=args.save)
    pass
