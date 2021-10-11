"""
Backoff words
"""
import re
from collections import defaultdict
from typing import TextIO, Dict, Tuple

from tqdm import tqdm

from lib4mc.FileLib import wc_l


def parse_line(line: str, splitter: str, start4words: int, step4words: int):
    line = line.strip("\r\n")
    if splitter == '':
        return list(line)
    items = re.split(splitter, line)
    words = items[start4words:len(items):step4words]
    return words


def backwords_counter(nwords_list: TextIO, splitter: str, start_chr: str, end_chr: str,
                      start4words: int, step4words: int, max_gram: int,
                      nwords_dict: Dict[Tuple, Dict[str, int]] = None, words: Dict[str, int] = None):
    if nwords_dict is None:
        nwords_dict: Dict[Tuple, Dict[str, int]] = {}
        words: Dict[str, int] = {}
    zero = tuple()
    # nwords_float_dict = {zero: {}}
    if isinstance(nwords_list, list):
        line_num = len(nwords_list)
    else:
        line_num = wc_l(nwords_list)
    section_dict = defaultdict(lambda: defaultdict(int))
    for line in tqdm(nwords_list, total=line_num, desc="Reading: "):  # type: str
        line = line.strip("\r\n")
        sections = [start_chr]
        sections.extend(parse_line(line, splitter, start4words, step4words))
        sections.append(end_chr)
        for sec in sections:
            if sec not in words:
                words[sec] = 0
            words[sec] += 1
            if sec not in {start_chr}:
                if zero not in nwords_dict:
                    nwords_dict[zero] = {}
                if sec not in nwords_dict[zero]:
                    nwords_dict[zero][sec] = 0
                nwords_dict[zero][sec] += 1

        section_dict[len(sections)][tuple(sections)] += 1
    pass

    for n in range(2, max_gram + 1):
        for sec_len, sec_len_dict in section_dict.items():
            if sec_len < n:
                continue
            order = n - 1
            for sec, num in sec_len_dict.items():
                for i in range(0, len(sec) - order):
                    prefix = sec[i:i + order]
                    transition = sec[i + order]

                    if prefix not in nwords_dict:
                        nwords_dict[prefix] = {}
                    if transition not in nwords_dict[prefix]:
                        nwords_dict[prefix][transition] = 0
                    nwords_dict[prefix][transition] += num
                pass
            pass
        pass
    return nwords_dict, words


def freq2prob(nwords_dict: Dict, threshold: int) -> Dict:
    nwords_float_dict: Dict[Tuple, Dict[str, float]] = {}
    for prefix, trans_cnt in sorted(nwords_dict.items(), key=lambda x: len(x[0])):
        total = sum(trans_cnt.values())
        if total < threshold:
            continue
        trans_prob = {trans: cnt / total for trans, cnt in trans_cnt.items() if cnt >= threshold}
        if len(trans_prob) == 0:
            # all transitions are ignored
            # because their frequencies are less than the threshold
            continue

        if len(trans_prob) < len(trans_cnt) and len(prefix) > 0:
            # some transitions are ignored
            # continue when prefix is empty, it occurs when the training file is too small
            missing = 1.0 - sum(trans_prob.values())
            parent_prefix = prefix[1:]
            for trans, p in nwords_float_dict[parent_prefix].items():
                trans_prob[trans] = trans_prob.get(trans, .0) + p * missing
        nwords_float_dict[prefix] = trans_prob

    return nwords_float_dict
