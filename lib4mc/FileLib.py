import sys
from typing import TextIO


def wc_l(file: TextIO, new_line: str = "\n", silence: bool = False):
    """
    a pure function, file will not be closed and move the pointer to the begin
    :param new_line: how to detect a new line
    :param silence: show warnings
    :param file: file to count lines
    :return: number of lines
    """
    if file.seekable():
        file.seek(0)
    elif not silence:
        print("WARNING: file cannot seekable", file=sys.stderr)
    buf_size = 8 * 1024 * 1024
    count = 0
    while True:
        buffer = file.read(buf_size)
        if not buffer:
            count += 1
            break
        count += buffer.count(new_line)
    if file.seekable():
        file.seek(0)
    return count
