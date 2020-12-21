#!/usr/bin/env python3

"""
Sample script for MNLab2BICN.

メディアネットワーク実験IIB 項目Iで利用するサンプル．

PSK および QAM は適切に実装されていません．`要修正`を要修正です．
"""

# NumPy と Matplotlib を使うので`import`する．
import matplotlib.pyplot as plt
import numpy as np

# 実験で使用するあらかじめ用意されている関数を`import`する．
from mnlab2bicn import awgn, calc_ser, calc_ber, \
    int2gray, gray2int, scatter_plot

# このscript中で利用するdebug用の変数．
verbose = True # True だと for loop の処理の状況(進み具合)が確認できる
verbose_awgn = False

# PSKのシミュレーションを行う関数を定義する．
def psk_test(m_level, i_snr, i_size, plot=False, gray=False):
    """PSK test

    Parameters
    ----------
    m_level: modulation level (integer)
    i_snr: SNR in dB
    i_size: number of samples
    plot: optional. if True, scatter plot will be generated.
    gray: optional. if True, gray code is used.

    Returns
    -------
    (SER, BER)
    """
    src = np.random.randint(m_level, size=i_size)

    x = int2gray(src) if gray else src

    # Start of 要修正
    y = x / (m_level - 1)
    # End of 要修正

    y_noisy = awgn(y, i_snr, verbose_awgn)

    if plot:
        scatter_plot(y_noisy, f"PSK_{m_level}_{i_snr}_{i_size}")

    # Start of 要修正
    z = np.round(y_noisy.real * (m_level - 1))
    z = np.where(z < 0, 0, z)
    z = np.where(z > m_level - 1, m_level - 1, z)
    # End of 要修正

    z = np.array(z, dtype='int')

    dst = gray2int(z) if gray else z

    return (calc_ser(src, dst), calc_ber(src, dst, m_level))


# QAMのシミュレーションを行う関数を定義する．
def qam_test(m_level, i_snr, i_size, plot=False, gray=False):
    """QAM test

    Parameters
    ----------
    m_level: modulation level (integer)
    i_snr: SNR in dB
    i_size: number of samples
    plot: optional. if True, scatter plot will be generated.
    gray: optional. if True, gray code is used.

    Returns
    -------
    (SER, BER)
    """
    src = np.random.randint(m_level, size=i_size)
    k = int(np.log2(m_level))
    smax = 2 ** (k // 2) - 1

    yr = src >> (k // 2)
    yi = src & smax

    yr = int2gray(yr) if gray else yr
    yi = int2gray(yi) if gray else yi

    # Start of 要修正
    y = np.exp(1j * (2 * np.pi) / (smax + 1) * yr)
    y *= ((yi + 1) / (smax + 1))
    # End of 要修正

    y_noisy = awgn(y, i_snr, verbose_awgn)

    if plot:
        scatter_plot(y_noisy, f"QAM_{m_level}_{i_snr}_{i_size}")

    # Start of 要修正
    zr = np.round(np.angle(y_noisy)/((2 * np.pi) / (smax + 1)))
    zr += (zr < 0)*(smax + 1)
    zr = np.array(zr, dtype='int')

    zi = np.round(np.abs(y_noisy) * (smax + 1))
    zi = np.array(zi, dtype='int')
    zi -= 1
    # End of 要修正

    zr = np.where(zr < 0, 0, zr)
    zr = np.where(zr > smax, smax, zr)
    zi = np.where(zi < 0, 0, zi)
    zi = np.where(zi > smax, smax, zi)

    zr = gray2int(zr) if gray else zr
    zi = gray2int(zi) if gray else zi

    dst = zr << (k // 2)
    dst += zi

    return (calc_ser(src, dst), calc_ber(src, dst, m_level))

# グラフのフォントサイズを調整する．
plt.rcParams.update({'font.size': 16})

# PSKのシミュレーションを行う．
# 図は`PSK_4_30_100.png`および`PSK_4_30_100.pdf`に保存される．以下同様．
# 
# 返り値は(SER, BER)．
# 多値数，SNR，およびサンプル数を様々に変更して
# シミュレーションする(課題1，2)．
psk_test(4, 30, 100, plot=True)
plt.close()
psk_test(8, 50, 100, plot=True)
plt.close()
psk_test(16, 40, 4000, plot=True)
plt.close()

# QAMのシミュレーションを行う．
# 
# 返り値は(SER, BER)．
# 多値数，SNR，およびサンプル数を様々に変更して
# シミュレーションする(課題1，2)．
qam_test(16, 30, 4000, plot=True)
plt.close()
qam_test(64, 30, 4000, plot=True)
plt.close()

# グラフのフォントサイズを調整する．
plt.rcParams.update({'font.size': 12})

# PSK/QAM Comparison
# 課題3．サンプル数(`size`)は適宜調整すること．
size = 10000
for m in [64, 16, 4]:
    snrs = []
    bers = []
    for snr in range(0, 55, 2):
        ser, ber = psk_test(m, snr, size)
        if verbose:
            print([snr, m, ser, ber])
        if ber < 1e-6:
            break
        snrs.append(snr)
        bers.append(ber)
    plt.semilogy(snrs, bers, '-o', label=f"{m}PSK")
    snrs = []
    bers = []
    for snr in range(0, 55, 2):
        ser, ber = qam_test(m, snr, size)
        if verbose:
            print([snr, m, ser, ber])
        if ber < 1e-6:
            break
        snrs.append(snr)
        bers.append(ber)
    plt.semilogy(snrs, bers, '-o', label=f"{m}QAM")

plt.grid()
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.legend()
plt.tight_layout()
plt.ylim(top=1)
plt.savefig("PSK_QAM.png")
plt.savefig("PSK_QAM.pdf")
plt.close()

# PSK Gray Code Comparison
# 課題4．サンプル数(`size`)は適宜調整すること．
size = 10000
for m in [64, 16, 4]:
    snrs = []
    bers = []
    for snr in range(0, 55, 2):
        ser, ber = psk_test(m, snr, size)
        if verbose:
            print([snr, m, ser, ber])
        if ber < 1e-6:
            break
        snrs.append(snr)
        bers.append(ber)
    plt.semilogy(snrs, bers, '-o', label=f"{m}PSK")
    snrs = []
    bers = []
    for snr in range(0, 55, 2):
        ser, ber = psk_test(m, snr, size, gray=True)
        if verbose:
            print([snr, m, ser, ber])
        if ber < 1e-6:
            break
        snrs.append(snr)
        bers.append(ber)
    plt.semilogy(snrs, bers, '-o', label=f"{m}PSK gray")

plt.grid()
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.legend()
plt.tight_layout()
plt.ylim(top=1)
plt.savefig("PSK_gray.png")
plt.savefig("PSK_gray.pdf")
plt.close()

# QAM Gray Code Comparison
# 課題4．サンプル数(`size`)は適宜調整すること．
size = 10000
for m in [64, 16, 4]:
    snrs = []
    bers = []
    for snr in range(0, 55, 2):
        ser, ber = qam_test(m, snr, size)
        if verbose:
            print([snr, m, ser, ber])
        if ber < 1e-6:
            break
        snrs.append(snr)
        bers.append(ber)
    plt.semilogy(snrs, bers, '-o', label=f"{m}QAM")
    snrs = []
    bers = []
    for snr in range(0, 55, 2):
        ser, ber = qam_test(m, snr, size, gray=True)
        if verbose:
            print([snr, m, ser, ber])
        if ber < 1e-6:
            break
        snrs.append(snr)
        bers.append(ber)
    plt.semilogy(snrs, bers, '-o', label=f"{m}QAM gray")

plt.grid()
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.legend()
plt.tight_layout()
plt.ylim(top=1)
plt.savefig("QAM_gray.png")
plt.savefig("QAM_gray.pdf")
plt.close()
