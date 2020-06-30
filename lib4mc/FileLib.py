from typing import TextIO


def wc_l(file: TextIO):
    """
    a pure function, file will not be closed and move the pointer to the begin
    :param file: file to count lines
    :return: number of lines
    """
    file.seek(0)
    new_line = "\n"
    buf_size = 8 * 1024 * 1024
    count = 0
    while True:
        buffer = file.read(buf_size)
        if not buffer:
            count += 1
            break
        count += buffer.count(new_line)
    file.seek(0)
    return count
