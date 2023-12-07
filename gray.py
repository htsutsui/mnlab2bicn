#!/usr/bin/env python3

import sys
import re
import numpy as np

"""
This module provides gray code coding and decoding functions.

These functions are based on
<https://rosettacode.org/wiki/Gray_code#Python>
but extended to support numpy array inputs.
"""


def gray_encode(n):
    """Return gray coded values.

    Parameters
    ----------
    n: array_like
        All values in n must be integers.

    Returns
    -------
    Gray coded object
    """
    if not isinstance(n, int):
        if (n < 0).any():
            raise ValueError("Negative value(s) are found.")
    else:
        if n < 0:
            raise ValueError("Negative value(s) are found.")
    return n ^ n >> 1


def gray_decode(n):
    """Return gray decoded values.

    Parameters
    ----------
    n: array_like
        All values in n must be integers.

    Returns
    -------
    Gray decoded object
    """
    i = isinstance(n, int)
    n = np.array([n]) if i else n.copy()
    if (n < 0).any():
        raise ValueError("Negative value(s) are found.")
    m = n >> 1
    while not (m == 0).all():
        n ^= m
        m >>= 1
    return n[0] if i else n


def __main():
    n = 5
    if len(sys.argv) > 1 and re.match(r'^\d+$', sys.argv[1]):
        m = int(sys.argv[1])
        if 1 <= m <= 16:
            n = m
    v = 2 ** n - 1

    nd = max(3, len(f"{v:d}"))
    nb = len(f"{v:b}")
    nbs = max(4, nb)
    keys = [("DEC", nd), ("BIN", nbs), ("GRAY", nbs), ("DEC", nd)]
    keys = [f"{i[0]:>{i[1]}}" for i in keys]
    print(', '.join([keys[0], ' => '.join(keys[1:])]))

    gray = gray_encode(np.array(range(2 ** n)))
    dec = gray_decode(gray)
    for i in range(2 ** n):
        bin_s = f"{i:>0{nb}b}"
        gray_s = f"{gray[i]:>0{nb}b}"
        s = ', '.join([f"{i:>{nd}d}",
                       ' => '.join([f"{bin_s:>{nbs}}",
                                    f"{gray_s:>{nbs}}",
                                    f"{dec[i]:>0{nd}d}"])])
        print(s)


if __name__ == '__main__':
    __main()
