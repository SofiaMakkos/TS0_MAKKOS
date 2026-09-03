# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 20:41:13 2026

@author: Zsofi
"""

# %%

import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# %%

fs=1000
N=1000
# %%

def mi_funcion_sen( vmax = np.sqrt(2), dc = 0, ff = 1, ph=0, nn = N, fs = fs):
    tt=np.arange(0.0, N/fs, 1/fs)
    xx=vmax*np.sin(2*np.pi*ff*tt +ph)+dc
    return(tt, xx)
tt, xx= mi_funcion_sen(ff=2000)

tt, xx= mi_funcion_sen(vmax = np.sqrt(2),ff=fs/N)

bits = 4
k= 10
Vfs = 2
q = (2*Vfs)/(np.pow(2,bits))
P_q = q**2/12
P_n = k* P_q
sigma = np.sqrt(P_n)
norm = np.random.normal(scale=sigma,size=len(tt))

sn = xx + norm
sn_cuantizada = np.round(sn/q)*q

plt.figure(1)
plt.plot(tt,sn_cuantizada,color = "hotpink",label="$Sn_q$ = $Q_{B,V_f}$(ADC in)")
plt.plot(tt,sn,color = "blue",label="$S_n$ = s+n (ADC in)")
plt.title("Señal muestreada por un ADC de 4 bits, $V_r$ = 2.0V y q = 0.125V")
plt.xlabel("tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.legend()

plt.show()

fftsn = fft(sn)
fftsn = fftsn[:N//2]
freq = fftfreq(N,1/fs)
freq = freq[:N//2]
fftsn_abs = np.abs(fftsn)
ref = np.max(fftsn_abs)
sn_fft_dB = 20 * np.log10(fftsn_abs/ref)

fftsn_cuant = fft(sn_cuantizada)
fftsn_cuant = fftsn_cuant[:N//2]
freq = fftfreq(N,1/fs)
freq = freq[:N//2]
fftsn_cuant_abs = np.abs(fftsn_cuant)
sn_cuant_fft_dB = 20 * np.log10(fftsn_cuant_abs/ref)

fftnorm = fft(norm)
fftnorm = fftnorm[:N//2]
freq = fftfreq(N,1/fs)
freq = freq[:N//2]
fftnorm_abs = np.abs(fftnorm)
norm_fft_dB = 20 * np.log10(fftnorm_abs/ref)

n_q = sn_cuantizada - sn
fftn_q = fft(n_q)
fftn_q = fftn_q[:N//2]
freq = fftfreq(N,1/fs)
freq = freq[:N//2]
fftn_q_abs = np.abs(fftn_q)
n_q_fft_dB = 20 * np.log10(fftn_q_abs/ref)

plt.figure(2)
plt.plot(freq,sn_cuant_fft_dB,color="purple",label="Sn_q")
plt.plot(freq,norm_fft_dB,linestyle ="-", color="green")
plt.axhline(np.mean(norm_fft_dB), color="green", linestyle="--", label=f"n: {np.mean(norm_fft_dB)}dB (piso analog.)")
plt.plot(freq,n_q_fft_dB,color="orange")
plt.axhline(np.mean(n_q_fft_dB), color="orange", linestyle='--', label=f"n_q: {np.mean(n_q_fft_dB)}dB (piso digital)")

plt.legend()
plt.title("FFT de Señal muestreada por un ADC de 4 bits, $V_r$ = 2.0V y q = 0.125V")
plt.ylabel("|X|")
plt.xlabel("frecuencia [Hz]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)


