import pickle
from typing import Dict, BinaryIO


def save_ngram(ngram_float_dict: Dict[str, Dict[str, float]], n: int, end_chr: str, file: BinaryIO) -> None:
    if not file.writable():
        raise Exception("file should be writable")
    pickle.dump((n, end_chr, ngram_float_dict), file)
    pass


def load_ngram(file: BinaryIO) -> (int, str, Dict[str, Dict[str, float]]):
    n, end_chr, ngram_float_dict = pickle.load(file)
    return n, end_chr, ngram_float_dict
