from math import log2
from typing import Dict, Tuple, TextIO, List


def minus_log2(nwords_dict_float: Dict[Tuple, Dict[str, float]]):
    for previous, items in nwords_dict_float.items():
        for item, prob in items.items():
            items[item] = -log2(prob)
            pass
    return nwords_dict_float
    pass


def enumerator(nwords_dict_float: Dict[Tuple, Dict[str, float]], threshold: float, start_chr, end_chr, min_len,
               f_save, order):
    nwords_dict_log2 = minus_log2(nwords_dict_float)
    cnt = [0]
    cur_pwd = tuple([start_chr for _ in range(order)])
    iterate(nwords_dict_log2, cur_pwd, .0, 0, min_len, end_chr, threshold, f_save, cnt, order)
    pass


def iterate(nwords_dict_log2: Dict, cur_pwd: Tuple, cur_prob: float, cur_len: int, min_len: int, end_chr: str,
            threshold: float, f_save: TextIO, cnt: List[int], order: int):
    previous = cur_pwd[-order:]
    next_candidates = nwords_dict_log2[previous]
    if cur_len > 30 or cur_prob >= threshold:
        return
    for char, m_log2 in next_candidates.items():
        new_cur_prob = cur_prob + m_log2
        if new_cur_prob < threshold:
            if char == end_chr:
                if cur_len >= min_len:
                    cnt[0] += 1
                    f_save.write(f'{"".join(cur_pwd[order:])}\t{new_cur_prob:.8f}\n')
                    if cnt[0] % 10000 == 0:
                        f_save.flush()
                continue
            iterate(nwords_dict_log2, cur_pwd + (char,), new_cur_prob, cur_len + len(char),
                    min_len, end_chr, threshold, f_save, cnt, order)
        pass
    pass
