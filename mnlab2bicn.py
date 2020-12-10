"""
MNLab2BICN
==========

メディアネットワーク実験IIA 項目I で利用する関数をまとめています．
"""

import matplotlib.pyplot as plt
import numpy as np
import gray


def __power(x):
    """Return signal power"""
    return np.average(np.abs(x)**2)


def awgn(x, snr, verbose=False):
    """AWGN, additive white Gaussian noise

    Parameters
    ----------
    x: array_like
    snr: SNR in dB
    verbose: it True, print power

    Returns
    -------
    Date where noise data is added.
    """
    sig_power = __power(x)
    noise_power = sig_power / np.power(10, snr/10)
    noise_gain = np.sqrt(noise_power/2)

    noise = np.random.randn(*(x.shape+(2,))).view(np.complex128)
    noise = noise.reshape(x.shape)
    noise *= noise_gain
    if verbose:
        print(f"SNR: {snr}, signal: {sig_power}, "
              f"noise: {__power(noise)}, noise_gain: {noise_gain}")
    return x + noise


def calc_ser(src, dst):
    """Calculate symbol error rate (SER)

    Parameters
    ----------
    src: array_like
    dst: array_like, whose shape should be same to `src`.

    Returns
    -------
    SER
    """
    return np.count_nonzero(src != dst) / np.size(src)


def __ber_count(num):
    c = 0
    while num != 0:
        if num & 1 != 0:
            c += 1
        num >>= 1
    return c


def calc_ber(src, dst, m):
    """Calculate bit error rate (BER)

    Parameters
    ----------
    src: array_like
    dst: array_like, whose shape should be same to `src`.
    m: modulation level, such as 2, 4, 8, ...
        This is required to calculate number of bits for each symbol.

    Returns
    -------
    SER
    """
    x = src ^ dst
    c = 0
    for i in x[np.where(x != 0)]:
        c += __ber_count(i)
    return c / (np.log2(m)*len(src))


def scatter_plot(x, filename=None, alpha=0.1, fontsize=None):
    """Generate scatter plot (IQ)

    Parameters
    ----------
    x: array_like, complex value data
    filename (optional): if specified, figures will be output to
        -    f"{filename}.png"
        -    f"{filename}.pdf"
    alpha (optional): alpha value for each point. Default is 0.1.
    fontsize (optional): fontsize for labels and ticks.

    Returns
    -------
    SER
    """
    plt.scatter(x.real, x.imag, alpha=alpha)
    plt.ylim(-1.2, 1.2)
    plt.xlim(-1.2, 1.2)
    if fontsize:
        plt.xlabel("I", fontsize=fontsize)
        plt.ylabel("Q", fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)
    else:
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.tick_params()
    plt.tight_layout()
    if filename:
        plt.savefig(f"{filename}.png")
        plt.savefig(f"{filename}.pdf")
    else:
        plt.show()
    plt.close()


int2gray = gray.gray_encode
gray2int = gray.gray_decode
