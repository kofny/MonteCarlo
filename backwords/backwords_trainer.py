"""
Backoff words
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


def backwords_counter(nwords_list: TextIO, min_gram: int, max_gram: int, splitter: str, end_chr: str, start4words: int,
                      skip4words: int, threshold: int):
    nwords_dict: Dict[Tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    line_num = wc_l(nwords_list)
    words: Dict[str, int] = defaultdict(int)
    section_dict = defaultdict(int)
    for line in tqdm(nwords_list, total=line_num, desc="Parsing: "):  # type: str
        line = line.strip("\r\n")
        sections = parse_line(line, splitter, start4words, skip4words)
        sections.append(end_chr)
        for sec in sections:
            words[sec] += 1
        section_dict[tuple(sections)] += 1
    pass
    for sections, cnt in tqdm(section_dict.items(), desc="Counting: "):
        for n in range(min_gram, max_gram + 1):
            prefix_words_num = n - 1
            for i in range(len(sections)):
                if i <= prefix_words_num:
                    nwords_dict[tuple(sections[:i])][sections[i]] += cnt
                else:
                    nwords_dict[tuple(sections[i - prefix_words_num:i])][sections[i]] += cnt
            pass
    del section_dict
    nwords_float_dict: Dict[Tuple, Dict[str, float]] = {}
    for prefix, ends in tqdm(nwords_dict.items(), "Converting: "):
        nwords_float_dict[prefix] = {}
        total = sum(ends.values())
        for e, v in ends.items():
            if len(prefix) > min_gram and v < threshold:
                continue
            nwords_float_dict[prefix][e] = (v / total)
    del nwords_dict

    return nwords_float_dict, words
