import abc
from collections import defaultdict
from math import log2
from typing import List, TextIO, Tuple, Union, Dict

from tqdm import tqdm

from lib4mc.FileLib import wc_l


class MonteCarlo(metaclass=abc.ABCMeta):
    @staticmethod
    def minus_log2(prob: float) -> float:
        return -log2(prob)

    @abc.abstractmethod
    def calc_ml2p(self, pwd: str) -> Tuple[float, List[str]]:
        """

        :param pwd: password
        :return:
        """
        return .0, []

    @abc.abstractmethod
    def sample1(self) -> (float, str):
        """
        get one sample
        :return: (prob, sample)
        """
        return .0, ""

    def sample(self, size: int, sampled_pwds: Dict[str, int] = None, clearIfNotNone: bool = True) -> List[float]:
        results = []
        samples = defaultdict(int)
        for _ in tqdm(iterable=range(size), desc="Sampling: "):
            prob, pwd = self.sample1()
            results.append(prob)
            samples[pwd] += 1
        if isinstance(sampled_pwds, defaultdict):
            if clearIfNotNone:
                sampled_pwds.clear()
            sampled_pwds.update(samples)
        return results

    def parse_file(self, testing_set: TextIO, using_component: bool = False) -> \
            List[Tuple[Union[str, List[str]], int, float]]:
        """
        get minus log prob for test set
        :param using_component:
        :param testing_set: test set
        :return: List of tuple (pwd, appearance, minus log prob)
        """
        line_num = wc_l(testing_set)
        pwd_counter = defaultdict(int)
        for line in tqdm(testing_set, desc="Reading: ", total=line_num):
            line = line.strip("\r\n")
            pwd_counter[line] += 1
        res: List[Tuple[Union[str, List[str]], int, float]] = []
        for pwd, num in tqdm(pwd_counter.items(), desc="Scoring: "):  # type: str, int
            _mlp, components = self.calc_ml2p(pwd)
            if using_component:
                res.append((components, num, _mlp))
            else:
                res.append((pwd, num, _mlp))
        res = sorted(res, key=lambda x: x[2])
        return res
