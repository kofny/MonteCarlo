import abc
from collections import defaultdict
from typing import List, TextIO, Tuple

from tqdm import tqdm

from lib4mc.FileLib import wc_l


class MonteCarlo(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def calc_minus_log_prob(self, pwd: str) -> float:
        """

        :param pwd: password
        :return:
        """
        return .0

    @abc.abstractmethod
    def sample_one(self) -> (float, str):
        return .0, ""

    def sample(self, size: int) -> List[float]:
        results = []
        for _ in tqdm(iterable=range(size), desc="Sampling: "):
            prob, pwd = self.sample_one()
            results.append(prob)
        return results

    def parse_file(self, testing_set: TextIO) -> List[Tuple[str, int, float]]:
        """
        get minus log prob for test set
        :param testing_set: test set
        :return: List of tuple (pwd, appearance, minus log prob)
        """
        line_num = wc_l(testing_set)
        pwd_counter = defaultdict(int)
        for line in tqdm(testing_set, desc="Counting: ", total=line_num):
            line = line.strip("\r\n")
            pwd_counter[line] += 1
        res: List[Tuple[str, int, float]] = []
        for pwd, num in tqdm(pwd_counter.items(), desc="Parsing test: "):
            _mlp = self.calc_minus_log_prob(pwd)
            res.append((pwd, num, _mlp))
        res = sorted(res, key=lambda x: x[2])
        return res
