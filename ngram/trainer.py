import random
from collections import defaultdict
from math import log10
from typing import TextIO, Dict

from tqdm import tqdm

from lib4mc.FileLib import wc_l


def ngram_counter(pwd_list: TextIO, n: int = 4, end_chr: str = "\x03") -> Dict[str, Dict[str, float]]:
    ngram_dict = defaultdict(lambda: defaultdict(int))
    prefix_len = n - 1
    line_num = wc_l(pwd_list)
    for line in tqdm(pwd_list, total=line_num, desc="Parsing: "):
        line = line.strip("\r\n")
        line += end_chr
        for i, c in enumerate(line):
            if i <= prefix_len:
                ngram_dict[line[:i]][line[i]] += 1
            else:
                ngram_dict[line[i - prefix_len:i]][line[i]] += 1
    ngram_float_dict = {}
    for prefix, ends in tqdm(ngram_dict.items(), "Converting: "):
        ngram_float_dict[prefix] = {}
        total = sum(ends.values())
        for e, v in ends.items():
            ngram_float_dict[prefix][e] = (v / total)
    del ngram_dict
    return ngram_float_dict


def count_space(ngrams: dict, prefix: str, total: int):
    addons = ngrams.get(prefix, {})

    count = len(addons)
    if count == 0:
        return 1
    _next = 0
    rand_addon = random.choice(list(addons))
    new_prefix = prefix[1:] + rand_addon
    if total > 40:
        return count * 1
    return count * count_space(ngrams, new_prefix, total + 1)
    # for addon in addons:
    #     new_prefix = prefix[1:] + addon
    #     _next += len(ngrams.get(new_prefix))
    pass


def main():
    for corpus in ['webhost']:
        nd = ngram_counter(open(f"/home/cw/Codes/Python/PwdTools/corpora/src/{corpus}-src.txt"), n=6)
        # print(nd)
        _total = 0
        _range = 10000
        for _ in range(_range):
            _total += count_space(nd, '', 0)
        print(f"{log10(_total/_range):5.2f}")
        # fd = open("./hello.pickle", "wb")
        # save_ngram(nd, 4, "\x03", fd)
    pass


if __name__ == '__main__':
    main()
