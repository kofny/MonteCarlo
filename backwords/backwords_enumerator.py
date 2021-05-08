from math import log2
from typing import Dict, Tuple, TextIO, List


def minus_log2(backwords_dict_float: Dict[Tuple, Dict[str, float]]):
    for previous, items in backwords_dict_float.items():
        for item, prob in items.items():
            items[item] = -log2(prob)
            pass
    return backwords_dict_float
    pass


def enumerator(backwords_dict_float: Dict[Tuple, Dict[str, float]], threshold: float, start_chr, end_chr, min_len,
               f_save):
    backwords_dict_log2 = minus_log2(backwords_dict_float)
    cnt = [0]
    iterate(backwords_dict_log2, (start_chr,), .0, 0, min_len, end_chr, threshold, f_save, cnt)
    pass


def iterate(backwords_dict_log2: Dict, cur_pwd: Tuple, cur_prob: float, cur_len: int, min_len: int, end_chr: str,
            threshold: float, f_save: TextIO, cnt: List[int]):
    previous = None
    for i in range(0, len(cur_pwd) + 1):
        previous = cur_pwd[i:]
        if previous in backwords_dict_log2:
            break
    if previous is None:
        raise Exception("previous is None! Unknown error!\n"
                        f"cur_pwd = {cur_pwd}")
    next_candidates = backwords_dict_log2[previous]
    if cur_len > 256 or cur_prob >= threshold:
        return
    for char, m_log2 in next_candidates.items():
        new_cur_prob = cur_prob + m_log2
        if new_cur_prob < threshold:
            if char == end_chr and cur_len >= min_len:
                cnt[0] += 1
                f_save.write(f'{"".join(cur_pwd[1:])}\t{new_cur_prob:.5f}\n')
                continue
            iterate(backwords_dict_log2, cur_pwd + (char,), new_cur_prob, cur_len + len(char),
                    min_len, end_chr, threshold, f_save, cnt)
        pass
    pass
