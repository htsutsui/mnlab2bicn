#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from mnlab2bicn import awgn, calc_ser, calc_ber, \
    int2gray, gray2int, jikken_scatter_plot

verbose = True


def psk_test(m_level, i_snr, i_size, plot=False, gray=False):
    src = np.random.randint(m_level, size=i_size)

    x = int2gray(src) if gray else src

    # Start of 要修正
    y = x / (m_level - 1)
    # End of 要修正

    ynoisy = awgn(y, i_snr)

    if plot:
        jikken_scatter_plot(ynoisy, f"PSK_{m_level}_{i_snr}_{i_size}")

    # Start of 要修正
    z = np.round(ynoisy.real * (m_level - 1))
    z = np.where(z < 0, 0, z)
    z = np.where(z > m_level - 1, m_level - 1, z)
    # End of 要修正

    z = np.array(z, dtype='int')

    dst = gray2int(z) if gray else z

    return (calc_ser(src, dst), calc_ber(src, dst, m_level))


def qam_test(m_level, i_snr, i_size, plot=False, gray=False):
    src = np.random.randint(m_level, size=i_size)
    nb = int(np.log2(m_level))
    smax = 2 ** (nb // 2) - 1

    xu = src >> (nb // 2)
    xl = src & smax

    xu = int2gray(xu) if gray else xu
    xl = int2gray(xl) if gray else xl

    # Start of 要修正
    xu = xu / smax
    xl = xl / smax
    y = xu + 1j * xl
    # End of 要修正

    ynoisy = awgn(y, i_snr)

    if plot:
        jikken_scatter_plot(ynoisy, f"QAM_{m_level}_{i_snr}_{i_size}")

    # Start of 要修正
    zr = np.array(np.round(ynoisy.real * smax), dtype='int')
    zi = np.array(np.round(ynoisy.imag * smax), dtype='int')
    # End of 要修正

    zr = np.where(zr < 0, 0, zr)
    zr = np.where(zr > smax, smax, zr)
    zi = np.where(zi < 0, 0, zi)
    zi = np.where(zi > smax, smax, zi)

    zr = gray2int(zr) if gray else zr
    zi = gray2int(zi) if gray else zi

    dst = zr << (nb // 2)
    dst += zi

    return (calc_ser(src, dst), calc_ber(src, dst, m_level))


plt.rcParams.update({'font.size': 16})

psk_test(4, 30, 100, plot=True)
psk_test(8, 50, 100, plot=True)
psk_test(16, 40, 4000, plot=True)
qam_test(16, 40, 4000, plot=True)

plt.rcParams.update({'font.size': 14})

# PSK QAM Comparison

size = 10000
for m in [64, 16, 4]:
    snrs = []
    bers = []
    for snr in range(0, 55, 2):
        ser, ber = psk_test(m, snr, size)
        if verbose:
            print([snr, m, ser, ber])
        if ber < 1e-5:
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
        if ber < 1e-5:
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

# PSK Gray Comparison

size = 10000
for m in [64, 16, 4]:
    snrs = []
    bers = []
    for snr in range(0, 55, 2):
        ser, ber = psk_test(m, snr, size)
        if verbose:
            print([snr, m, ser, ber])
        if ber < 1e-5:
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
        if ber < 1e-5:
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

# QAM Gray Comparison

size = 10000
for m in [64, 16, 4]:
    snrs = []
    bers = []
    for snr in range(0, 55, 2):
        ser, ber = qam_test(m, snr, size)
        if verbose:
            print([snr, m, ser, ber])
        if ber < 1e-5:
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
        if ber < 1e-5:
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
