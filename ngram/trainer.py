from collections import defaultdict
from typing import TextIO, Dict

from tqdm import tqdm

from lib4mc.FileLib import wc_l
from lib4mc.SaveModelLib import save_ngram


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


if __name__ == '__main__':
    nd = ngram_counter(open("/home/cw/Codes/Python/PwdTools/corpora/src/csdn-src.txt"))
    fd = open("./hello.pickle", "wb")
    save_ngram(nd, 4, "\x03", fd)
    pass
