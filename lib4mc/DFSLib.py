"""
This is a library for Depth First Search and Dict Tree Compare.
"""
from typing import Dict, Any, List, Tuple


def extract(dtree: Dict[str, Any], pwd: str, max_len: int, end: str = "\x03"):
    """
    find overlaps between dtree and pwd.
    Find the longest match first
    :param max_len:
    :param end: the end symbol of search a dict tree
    :param dtree:
    :param pwd:
    :return: List[(start idx, length of segment, whether this segment is tagged or not)]
    """
    kbd_list = []
    a_kbd = ""
    lower_pwd = pwd.lower()
    len_pwd = len(pwd)
    i = 0
    cur_i = i
    len_kbd = 0
    bk_dtree = dtree
    while i < len_pwd and cur_i < len_pwd:
        c = lower_pwd[cur_i]
        if c in bk_dtree:
            a_kbd += c
            bk_dtree = bk_dtree[c]
            if end in bk_dtree:
                # find end symbol, meaning this may be end of a keyboard pattern
                # following chars instead of end symbol may be added.
                add_a_kbd = ""
                bak_add_a_kbd = ""
                # find as much as possible chars added
                # the max len for kbd pattern is max_kbd_len,
                # current kbd len is len(a_kbd), therefore our end index is cur_i + max_kbd_len - len(a_kbd) + 1
                for addi in range(cur_i + 1, min(cur_i + max_len - len(a_kbd) + 1, len_pwd)):
                    addc = lower_pwd[addi]
                    # not found, return previously found
                    if addc not in bk_dtree:
                        break
                    bk_dtree = bk_dtree[addc]
                    add_a_kbd += addc
                    # update position of end symbol
                    if end in bk_dtree:
                        bak_add_a_kbd = add_a_kbd
                if bak_add_a_kbd != "":
                    a_kbd += bak_add_a_kbd
                    cur_i += len(bak_add_a_kbd)
                len_a_kbd = len(a_kbd)
                # start index of this kbd, length of this kbd, this is kbd
                kbd_list.append((cur_i - len_a_kbd + 1, len_a_kbd, True))
                # update global values
                len_kbd += len_a_kbd
                i += len_a_kbd
                cur_i = i
                a_kbd = ""
                bk_dtree = dtree
            cur_i += 1
        else:
            i += 1
            cur_i = i
            a_kbd = ""
            bk_dtree = dtree
    if len_kbd == len_pwd:
        return kbd_list
    elif len(kbd_list) == 0:
        return [(0, len_pwd, False)]
    else:
        # n_list keeps all nodes that a kbd starts or ends
        # is_kbd_set keeps that whether the node is the start of kbd
        n_list = set()
        is_kbd_set = set()
        n_list.add(0)
        # i: start idx of kbd, kl: length of kbd
        for i, kl, is_kbd in kbd_list:
            n_list.add(i)
            n_list.add(i + kl)
            is_kbd_set.add(i)
        n_list.add(len_pwd)
        n_list = sorted(n_list)
        n_kbd_list = []
        for n_i, pwd_i in enumerate(n_list[:-1]):
            n_kbd_list.append((pwd_i, n_list[n_i + 1] - pwd_i, pwd_i in is_kbd_set))
        n_kbd_list = sorted(n_kbd_list, key=lambda x: x[0])
        return n_kbd_list
    pass


def post_parse4case_free(res: List[Tuple[int, int, bool]], pwd: str, tag: str) -> (List[Tuple[str, str]], List[str]):
    """
    result gotten from extract() is hard to read.
    this function will convert it into easy-to-read form.
    :param res:
    :param pwd:
    :param tag:
    :return:
    """
    section_list = []
    tag_list = []
    for idx, len_seg, is_tagged in res:
        seg = pwd[idx:idx + len_seg]
        if is_tagged:
            section_list.append((seg, f"{tag}{len_seg}"))
            tag_list.append(seg)
        else:
            section_list.append((seg, None))
        pass
    return section_list, tag_list


def gen_dtree(entries: Dict[str, int], end: str = "\x03") -> Tuple[Dict, int]:
    """
    get dict tree form a dict
    :param entries:
    :param end: End of a keyboard pattern
    :return: dict tree and max_len
    """
    lst = sorted(entries.keys(), key=lambda x: len(x), reverse=True)
    if len(lst) == 0:
        return {}, 20
    # min_len = len(lst[-1])
    max_len = len(lst[0])
    dtree = {}
    for entry in entries:
        tmp_dtree = dtree
        for c in entry:
            if c not in tmp_dtree:
                tmp_dtree[c] = {}
            tmp_dtree = tmp_dtree[c]
        tmp_dtree[end] = True
    return dtree, max_len
