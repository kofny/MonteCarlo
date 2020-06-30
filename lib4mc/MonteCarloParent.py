import abc
from typing import List, TextIO, Tuple


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

    @abc.abstractmethod
    def sample(self, size: int) -> List[float]:
        return [.0]

    @abc.abstractmethod
    def parse_file(self, test: TextIO) -> List[Tuple[str, int, float]]:
        return [("", 0, .0)]
        pass
