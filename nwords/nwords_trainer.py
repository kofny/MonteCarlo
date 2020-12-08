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
    nwords_dict: Dict[Tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    prefix_words_num = n - 1
    line_num = wc_l(nwords_list)
    section_dict = defaultdict(int)
    words: Dict[str, int] = defaultdict(int)
    default_start = start_chr * (n - 1)
    for line in tqdm(nwords_list, total=line_num, desc="Reading: "):  # type: str
        line = line.strip("\r\n")
        sections = parse_line(default_start + line, splitter, start4words, skip4words)
        sections.append(end_chr)
        for sec in sections:
            words[sec] += 1
        section_dict[tuple(sections)] += 1
    nwords_list.close()
    for sections, cnt in tqdm(section_dict.items(), desc="Counting: "):
        for i in range(len(sections) - prefix_words_num):
            grams = tuple(sections[i:i + prefix_words_num])
            transition = sections[i + prefix_words_num]
            nwords_dict[grams][transition] += cnt
    del section_dict
    nwords_float_dict: Dict[Tuple, Dict[str, float]] = {}
    for prefix, ends in tqdm(nwords_dict.items(), "Converting: "):
        nwords_float_dict[prefix] = {}
        total = sum(ends.values())
        for e, v in ends.items():
            nwords_float_dict[prefix][e] = (v / total)
    del nwords_dict
    return nwords_float_dict, words
