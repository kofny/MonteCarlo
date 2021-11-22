"""
N words
"""
import re
from collections import defaultdict
from typing import TextIO, Dict, Tuple

from tqdm import tqdm

from lib4mc.FileLib import wc_l


def parse_line(line: str, splitter: str, start4words: int, skip4words: int):
    line = line.strip("\r\n")
    if splitter == '':
        return list(line)
    items = re.split(splitter, line)
    words = items[start4words:len(items):skip4words]
    return words


def nwords_counter(nwords_list: TextIO, n: int, splitter: str, end_chr: str, start4words: int,
                   skip4words: int, start_chr: str = '\x00'):
    valid_l = {*list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"), start_chr, end_chr}
    valid_d = {*list("0123456789"), start_chr, end_chr}
    nwords_dict: Dict[Tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    prefix_words_num = n - 1
    line_num = wc_l(nwords_list)
    section_dict = defaultdict(int)
    words: Dict[str, int] = defaultdict(int)
    # default_start = start_chr * (n - 1)
    for line in tqdm(nwords_list, total=line_num, desc="Reading: "):  # type: str
        line = line.strip("\r\n")
        sections = [start_chr for _ in range(n - 1)]
        extends = parse_line(line, splitter, start4words, skip4words)
        sections.extend(extends)
        sections.append(end_chr)
        for sec in sections:
            words[sec] += 1
        section_dict[tuple(sections)] += 1
    nwords_list.close()
    for sections, cnt in tqdm(section_dict.items(), desc="Counting: "):
        for i in range(len(sections) - prefix_words_num):
            context = tuple(sections[i:i + prefix_words_num])
            transition = sections[i + prefix_words_num]
            ngram = "".join(sections[i:i + prefix_words_num + 1])
            valid_l_ngram = transition in valid_l or transition in valid_d
            for c in ngram:
                if c not in valid_l:
                    valid_l_ngram = False
                    break
            valid_d_ngram = True
            if not valid_l_ngram:
                for c in ngram:
                    if c not in valid_d:
                        valid_d_ngram = False
                        break
            if valid_l_ngram or valid_d_ngram:
                nwords_dict[context][transition] += cnt
    del section_dict
    nwords_float_dict: Dict[Tuple, Dict[str, float]] = {}
    for prefix, ends in tqdm(nwords_dict.items(), "Converting: "):
        nwords_float_dict[prefix] = {}
        total = sum(ends.values())

        for e, v in ends.items():
            next_prefix = (*prefix[1:], e)
            if e != end_chr and next_prefix not in nwords_dict:
                total -= v
                continue
        for e, v in ends.items():
            next_prefix = (*prefix[1:], e)
            if e != end_chr and next_prefix not in nwords_dict:
                continue
            nwords_float_dict[prefix][e] = (v / total)
    del nwords_dict
    return nwords_float_dict, words


if __name__ == '__main__':
    with open('/home/cw/Codes/Python/MonteCarlo/toyrockyou.txt', 'r') as fin:
        # with open('/home/cw/Documents/Experiments/SegLab/Corpora632/rockyou-src.txt', 'r') as fin:
        nwords_counter(fin, n=6, splitter='', end_chr='\x03', start4words=0, skip4words=1, start_chr='\x00')
    pass
