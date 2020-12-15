#!/usr/bin/env python3

"""
Pytest script
"""

import numpy as np
import mnlab2bicn as icn


def test_gray():
    """Test gray code encode/decode
    """
    a = np.arange(16).reshape((2, 2, 2, 2))
    assert (icn.gray2int(icn.int2gray(a)) == a).all()


def test_awgn():
    """Test `awgn`
    """
    a = np.arange(16).reshape((2, 2, 2, 2))
    icn.awgn(a, 10)


def test_ser_ber():
    """Test SER and BER calculation
    """
    a = np.zeros(16 * 1000, dtype='int')
    a = a.reshape((2, 2, 2, 2, 1000))
    b = a.copy()
    b[::2] = 1
    for i in range(1, 8):
        assert icn.calc_ser(a, b) == 0.5
        assert icn.calc_ber(a, b, 2**i) == 1/2/i


def __main():
    test_gray()
    test_awgn()
    test_ser_ber()


if __name__ == '__main__':
    __main()
