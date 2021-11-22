import bisect
import random
import sys
from math import log2
from typing import Dict, Tuple, List, Any

import numpy


def expand_2d(two_d_dict: Dict[Any, Dict[Any, float]], minus_log_based: bool = False) \
        -> Dict[Any, Tuple[Dict[Any, float], List[Any], List[float]]]:
    new_two_d_dict = {}
    for k, items in two_d_dict.items():
        if len(items) == 0:
            continue
        new_two_d_dict[k] = expand_1d(items, minus_log_based=minus_log_based)
    return new_two_d_dict


def expand_1d(one_d_dict: Dict[Any, float], minus_log_based: bool = False) \
        -> Tuple[Dict[Any, float], List[Any], List[float]]:
    keys = list(one_d_dict.keys())
    cum_sums = numpy.array(list(one_d_dict.values())).cumsum()
    n_one = one_d_dict
    if minus_log_based:
        n_one = {k: -log2(v) for k, v in one_d_dict.items()}
    new_one_d_dict = (n_one, keys, cum_sums)
    return new_one_d_dict


def pick_expand(expanded: Tuple[Dict[str, float], List[str], List[float]]) -> Tuple[float, str]:
    try:
        items, keys, cum_sums = expanded
    except TypeError:
        print(f"TypeError: expanded is {expanded}", file=sys.stderr)
        sys.exit(-1)
    if len(cum_sums) < 1:
        print(keys)
        pass
    total = cum_sums[-1]
    idx = bisect.bisect_right(cum_sums, random.uniform(0, total))
    k: str = keys[idx]
    return items.get(k), k
