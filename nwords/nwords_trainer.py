"""
N words
"""
from collections import defaultdict
from typing import TextIO

from tqdm import tqdm

from lib4mc.FileLib import wc_l


def nwords_counter(nwords_list: TextIO, n: int = 4, end_chr: str = "\x03"):
    nwords_dict = defaultdict(lambda: defaultdict(int))
    prefix_words = n - 1
    line_num = wc_l(nwords_list)
    section_dict = defaultdict(int)
    for line in tqdm(nwords_list, total=line_num, desc="Parsing: "):  # type: str
        line = line.strip("\r\n")
        items = line.split("\t")
        pwd = items[0]
        raw_sections = items[1::2]
        start = 0
        sections = []
        for sec in raw_sections:
            sections.append(pwd[start:start + len(sec)])
            start += len(sec)
        sections.append(end_chr)
        section_dict[tuple(sections)] += 1
    nwords_list.close()
    for sections, cnt in tqdm(section_dict.items(), desc="Counting: "):
        for i, sec in enumerate(sections):
            if i <= prefix_words:
                nwords_dict[tuple(sections[:i])][sections[i]] += cnt
            else:
                nwords_dict[tuple(sections[i - prefix_words:i])][sections[i]] += cnt
    del section_dict
    nwords_float_dict = {}
    for prefix, ends in tqdm(nwords_dict.items(), "Converting: "):
        nwords_float_dict[prefix] = {}
        total = sum(ends.values())
        for e, v in ends.items():
            nwords_float_dict[prefix][e] = (v / total)
        pass
    del nwords_dict
    return nwords_float_dict


if __name__ == '__main__':
    nfd = nwords_counter(open("/home/cw/Documents/Experiments/SegLab/NWords/csdn-rded.txt"))
    # print(nfd.items())
