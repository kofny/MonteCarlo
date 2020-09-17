import re
import sys
from collections import defaultdict
from math import log2
from typing import TextIO, List, Tuple, Dict, Any, Set

from tqdm import tqdm

from bpeX.modelreader import read_bpe
from lib4mc.MonteCarloLib import MonteCarloLib
from lib4mc.MonteCarloParent import MonteCarlo
from lib4mc.ProbLib import expand_2d, pick_expand, expand_1d

re_digits = re.compile(r"\d+")


def luds(pwd: str):
    struct = []
    prev_tag = ""
    t_len = 0
    cur_tag = " "
    for c in pwd:
        if c.isalpha():
            if c.isupper():
                cur_tag = "U"
            else:
                cur_tag = "L"
        elif c.isdigit():
            cur_tag = "D"
        else:
            cur_tag = "S"
        if cur_tag == prev_tag:
            t_len += 1
        else:
            if len(prev_tag) > 0:
                struct.append((prev_tag, t_len))
            prev_tag = cur_tag
            t_len = 1
    struct.append((cur_tag, t_len))
    return tuple(struct)

    pass


def count_luds(structures: Dict[str, float]) -> (Dict[Any, Set], Dict[str, set]):
    skipped_list = []
    converts = defaultdict(set)
    for structure in structures:
        parsed_structure = []

        skip = False
        for tag, t_len in structure:
            if len(parsed_structure) > 0:
                prev_tag, prev_len = parsed_structure[-1]
                if prev_tag != tag:
                    parsed_structure.append((tag, t_len))
                else:
                    parsed_structure[-1] = (tag, prev_len + t_len)
            else:
                parsed_structure.append((tag, t_len))
            if 'M' in tag:
                skip = True
        parsed_structure = tuple(parsed_structure)
        if skip:
            skipped_list.append(structure)
            continue
        converts[parsed_structure].add(structure)
    novels = defaultdict(set)
    print("Preprocess done!")
    for k in converts.keys():
        novels[len(k)].add(k)

    def the_same(struct_a, struct_b) -> bool:
        if len(struct_a) != len(struct_b):
            return False
        for s_a, s_b in zip(struct_a, struct_b):
            if s_a != s_b and 'M' not in s_a and 'M' not in s_b:
                return False
        return True

    struct_speedup = {}
    not_parsed = defaultdict(set)
    for skipped in tqdm(skipped_list, desc="Refining: "):
        candidates = novels[len(skipped)]
        speed_skipped = []
        for s_tag, s_len in skipped:
            speed_skipped.extend([s_tag] * s_len)

        for candidate in candidates:
            if candidate not in struct_speedup:
                backup = []
                for s_tag, s_len in candidate:
                    backup.extend([s_tag] * s_len)
                struct_speedup[candidate] = backup
            speed_candidate = struct_speedup[candidate]
            if the_same(speed_candidate, speed_skipped):
                converts[candidate].add(skipped)
        length = sum([_len for _, _len in skipped])
        not_parsed[length].add(skipped)

    return converts, not_parsed


class BpePcfgSim(MonteCarlo):
    def sample_one(self) -> (float, str):
        pwd = ""
        prob = .0
        p, struct = pick_expand(self.__grammars)
        prob += p
        # lst = [(struct, 2 ** (-p))]
        for tag_len in struct:
            target_terminal = self.__terminals[tag_len]
            p, replacement = pick_expand(target_terminal)
            prob += p
            pwd += replacement
            # lst.append((replacement, 2 ** (-p)))
        # lst.append((pwd, 2 ** (-prob)))
        # print(lst)
        return prob, pwd

    def sample(self, size: int = 1000000) -> List[float]:
        results = []
        for _ in tqdm(iterable=range(size), desc="Sampling: "):
            prob, _ = self.sample_one()
            results.append(prob)
        return results

    def parse_file(self, testing_set: TextIO) -> List[Tuple[str, int, float]]:
        pwd_counter = defaultdict(int)
        for line in testing_set:
            pwd = line.strip("\r\n")
            pwd_counter[pwd] += 1
            # prob = self.calc_minus_log_prob(pwd)
            pass
        results = []
        for pwd, cnt in tqdm(iterable=pwd_counter.items(), desc="Scoring: "):
            prob = self.calc_minus_log_prob(pwd)
            results.append((pwd, cnt, prob))
        del pwd_counter
        return results

    def calc_minus_log_prob(self, pwd: str) -> float:
        label = luds(pwd)
        candidate_structures = self.__converted.get(label, set())
        log_max = log2(sys.float_info.max)
        if len(candidate_structures) == 0:
            length = sum([_len for _, _len in label])
            addon_candidate_structures = self.__not_parsed.get(length, set())
            candidate_structures.update(addon_candidate_structures)
            if len(candidate_structures) == 0:
                return log_max
        grammars, _, _ = self.__grammars
        results = []
        for candidate in candidate_structures:
            p = grammars.get(candidate, log_max)
            if p == log_max:
                break
            start = 0
            for tag, t_len in candidate:
                terminal, _, _ = self.__terminals.get((tag, t_len))
                replacement = pwd[start:start + t_len]
                start += t_len
                if replacement not in terminal:
                    p = log_max
                    break
                else:
                    p += terminal[replacement]
            if p < log_max:
                results.append((candidate, p))
        if len(results) == 0:
            min_minus_log_prob = log_max
        else:
            _, min_minus_log_prob = min(results, key=lambda x: x[1])
        return min_minus_log_prob

    def __init__(self, model_path: str):
        grammars, terminals = read_bpe(model_path=model_path)
        self.__grammars = expand_1d(grammars, minus_log_based=True)
        self.__terminals = expand_2d(terminals, minus_log_based=True)
        self.__converted, self.__not_parsed = count_luds(grammars)
        pass


def test():
    bpePcfg = BpePcfgSim(model_path="/home/cw/Documents/tmp/model")
    samples = bpePcfg.sample(size=10000000)
    monte_carlo = MonteCarloLib(minus_log_prob_list=samples)
    while True:
        pwd = input("type in a password: ")
        if pwd == 'exit!':
            sys.exit(0)
        prob = bpePcfg.calc_minus_log_prob(pwd=pwd)
        print(f"prob: {2 ** (-prob)}", end=", ")
        rank = monte_carlo.minus_log_prob2rank(prob)
        print(f"rank: {rank}")
    pass


def wrapper(model_path: str, testing_set: TextIO, save2: TextIO):
    # "/home/cw/Documents/tmp/model"
    bpePcfg = BpePcfgSim(model_path=model_path)
    samples = bpePcfg.sample(size=10000000)
    # open("/home/cw/Documents/tmp/178_new.txt")
    scored = bpePcfg.parse_file(testing_set)
    monte_carlo = MonteCarloLib(minus_log_prob_list=samples)
    monte_carlo.mlps2gc(scored, need_resort=True, add1=False)
    # open("/home/cw/Documents/tmp/scored_178.txt", "w")
    monte_carlo.write2(save2)

    pass


if __name__ == '__main__':
    # test()
    wrapper(model_path="/home/cw/Documents/tmp/model",
            testing_set=open("/home/cw/Documents/tmp/178_new.txt"),
            save2=open("/home/cw/Documents/tmp/scored_178.txt", "w"))
