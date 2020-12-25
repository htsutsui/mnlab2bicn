#!/usr/bin/env python3

"""
This module provides gray code coding and decoding functions.

These functions are based on
<https://rosettacode.org/wiki/Gray_code#Python>
but extended to support numpy array inputs.
"""

import numpy as np


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
    print("DEC,   BIN =>  GRAY => DEC")
    for i in range(32):
        gray = gray_encode(i)
        dec = gray_decode(gray)
        print(f" {i:>2d}, {i:>05b} => {gray:>05b} => {dec:>2d}")


if __name__ == '__main__':
    __main()
