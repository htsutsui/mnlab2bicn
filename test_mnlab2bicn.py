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
    a = np.arange(16)
    assert (np.array(([icn.gray2int(icn.int2gray(i)) for i in a])) == a).all()
    i = -10
    try:
        icn.gray2int(i)
        assert False
    except ValueError:
        assert True
    try:
        icn.int2gray(i)
        assert False
    except ValueError:
        assert True


def test_awgn():
    """Test `awgn`
    """
    a = np.arange(16).reshape((2, 2, 2, 2))
    icn.awgn(a, 10)


def test_ser_ber(size=1000):
    """Test SER and BER calculation
    """
    a = np.zeros(16 * size, dtype='int')
    a = a.reshape((2, 2, 2, 2, size))
    b = a.copy()
    for i in range(1, 8):
        assert icn.calc_ser(a, b) == 0
        assert icn.calc_ber(a, b, 2**i) == 0
    b[:, :, :, :, ::2] = 1
    for i in range(1, 8):
        assert icn.calc_ser(a, b) == 0.5
        assert icn.calc_ber(a, b, 2**i) == 1/2/i
    b = b.reshape((4, 4, size))
    try:
        print(icn.calc_ser(a, b))
        assert False
    except ValueError:
        assert True
    try:
        i = 2
        print(icn.calc_ber(a, b, 2**i))
        assert False
    except ValueError:
        assert True


def __main():
    test_gray()
    test_awgn()
    test_ser_ber()


if __name__ == '__main__':
    __main()
