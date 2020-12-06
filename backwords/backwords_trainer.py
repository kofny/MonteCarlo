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


def backwords_counter(nwords_list: TextIO, splitter: str, start_chr: str, end_chr: str,
                      start4words: int, skip4words: int, threshold: int):
    nwords_dict: Dict[Tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    zero = tuple()
    nwords_float_dict = {zero: {}}
    line_num = wc_l(nwords_list)
    words: Dict[str, int] = defaultdict(int)
    section_dict = defaultdict(lambda: defaultdict(int))
    for line in tqdm(nwords_list, total=line_num, desc="Parsing: "):  # type: str
        line = line.strip("\r\n")
        sections = [start_chr]
        sections.extend(parse_line(line, splitter, start4words, skip4words))
        sections.append(end_chr)
        for sec in sections:
            words[sec] += 1
            if sec not in {start_chr}:
                nwords_dict[zero][sec] += 1
        section_dict[len(sections)][tuple(sections)] += 1
    pass

    zero_sum = sum(nwords_dict[zero].values())
    for trans, p in nwords_dict[zero].items():
        nwords_float_dict[zero][trans] = p / zero_sum
    min_gram = 2
    max_gram = max([_l for _l, s in section_dict.items() if sum(s.values()) >= threshold])
    for n in tqdm(range(min_gram, max_gram + 1), desc="Counting: "):
        nwords_dict: Dict[Tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for sec_len, sections_cnt in section_dict.items():
            if n >= sec_len:
                continue
            for sections, cnt in sections_cnt.items():
                prefix_words_num = n - 1
                for i in range(len(sections) - prefix_words_num):
                    grams = tuple(sections[i:i + prefix_words_num])
                    transition = sections[i + prefix_words_num]
                    nwords_dict[grams][transition] += cnt
            pass
        for prefix, trans_cnt in nwords_dict.items():
            total = sum(trans_cnt.values())
            if total < threshold:
                continue
            trans_prob = {trans: cnt / total for trans, cnt in trans_cnt.items() if cnt >= threshold}
            missing = 1 - sum(trans_prob.values())
            if missing == 1:
                continue
            if missing > 0:
                parent_prefix = prefix[1:]
                for trans, p in nwords_float_dict[parent_prefix].items():
                    trans_prob[trans] = trans_prob.get(trans, 0) + p * missing
            nwords_float_dict[prefix] = trans_prob
    del section_dict
    return nwords_float_dict, words
