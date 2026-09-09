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

tt, xx= mi_funcion_sen(vmax = np.sqrt(2),ff=fs/N)

bits = 4
k= 1
Vfs = 2
q = (2*Vfs)/(np.pow(2,bits))
P_q = q**2/12
P_n = k* P_q
sigma = np.sqrt(P_n)
norm = np.random.normal(scale=sigma,size=len(tt))

sn = xx + norm
sn_cuantizada = np.round(sn/q)*q

plt.figure(1)
plt.plot(tt,sn_cuantizada,color = "hotpink",label="$Sn_q$ = $Q_{B,V_f}$(ADC out)")
plt.plot(tt,sn, ":.", color = "blue",label="$S_n$ = xx+norm (ADC in)")
plt.plot(tt, xx, linestyle="-", linewidth=2, color="red", label="Función seno (xx)")
plt.title(f"Señal muestreada por un ADC de {bits} bits, ±$V_r$ = {Vfs}V y q = {q}V")
plt.xlabel("tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.legend()

plt.show()

# %%
def funcion_fft (xx):
    fftxx = fft(xx)
    fftxx = fftxx[:N//2]
    freq = fftfreq(N,1/fs)
    freq = freq[:N//2]
    fftxx_abs = np.abs(fftxx)
    return fftxx_abs, freq

fftsn_abs, freq=funcion_fft(sn)
ref = np.max(fftsn_abs)
sn_fft_dB = 20 * np.log10(fftsn_abs/ref)

fftsn_cuant_abs, freq=funcion_fft(sn_cuantizada)
sn_cuant_fft_dB = 20 * np.log10(fftsn_cuant_abs/ref)

fftnorm_abs, freq=funcion_fft(norm)
norm_fft_dB = 20 * np.log10(fftnorm_abs/ref)

#Ruido de cuantización
n_q = sn_cuantizada - sn
fftn_q_abs, freq=funcion_fft(n_q)
n_q_fft_dB = 20 * np.log10(fftn_q_abs/ref)

plt.figure(2)
plt.plot(freq,sn_cuant_fft_dB,":.", color="purple",label="Sn_q")
plt.plot(freq,norm_fft_dB,":.", color="green")
plt.axhline(np.mean(norm_fft_dB), color="green", linestyle="--", label=f"n: {np.mean(norm_fft_dB):.2f}dB (piso analog.)")
plt.plot(freq,n_q_fft_dB,":.", color="orange")
plt.axhline(np.mean(n_q_fft_dB), color="orange", linestyle='--', label=f"n_q: {np.mean(n_q_fft_dB):.2f}dB (piso digital)")

plt.legend()
plt.title(f"FFT de Señal muestreada por un ADC de {bits} bits, ±$V_r$ = {Vfs}V y q = {q}V")
plt.ylabel("Densidad de potencia [dB]")
plt.xlabel("frecuencia [Hz]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

# %%
plt.figure (3)
plt.hist(n_q, color="cyan")
plt.hlines(100, xmin=-q/2, xmax=q/2,color="orchid", linestyle="--", label=f"{N}/bins")
plt.vlines(-q/2, ymin=0, ymax=100, color="coral", linestyle="--", label=f"{q}/2")
plt.vlines(q/2,ymin=0, ymax=100, color="salmon", linestyle="--", label=f"{q}/2")
plt.title(f"Ruido de cuantización para {bits} bits, ±$V_r$ = {Vfs}V y q = {q}V")
plt.legend()
plt.xlabel("Amplitud [V]")
plt.ylabel("Nro de muestras")

plt.show()

# %%

tt, xx= mi_funcion_sen(vmax = np.sqrt(2),ff=fs/N)

bits = 16
k= 1/10
Vfs = 2
q = (2*Vfs)/(np.pow(2,bits))
P_q = q**2/12
P_n = k* P_q
sigma = np.sqrt(P_n)
norm = np.random.normal(scale=sigma,size=len(tt))

sn = xx + norm
sn_cuantizada = np.round(sn/q)*q

plt.figure(4)
plt.plot(tt,sn_cuantizada,color = "blue",linewidth=6,label="$Sn_q$ = $Q_{B,V_f}$(ADC out)")
plt.plot(tt,sn, ":.", color = "lavender",label="$S_n$ = xx+norm (ADC in)")
plt.plot(tt, xx, linestyle="-", linewidth=0.5, color="red", label="Función seno (xx)")
plt.title(f"Señal muestreada por un ADC de {bits} bits, ±$V_r$ = {Vfs}V y q = {q:.2e}V")
plt.xlabel("tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.legend()

plt.show()
# %%
 
fftsn_abs, freq=funcion_fft(sn)
ref = np.max(fftsn_abs)
sn_fft_dB = 20 * np.log10(fftsn_abs/ref)

fftsn_cuant_abs, freq=funcion_fft(sn_cuantizada)
sn_cuant_fft_dB = 20 * np.log10(fftsn_cuant_abs/ref)

fftnorm_abs, freq=funcion_fft(norm)
norm_fft_dB = 20 * np.log10(fftnorm_abs/ref)

#Ruido de cuantización
n_q = sn_cuantizada - sn
fftn_q_abs, freq=funcion_fft(n_q)
n_q_fft_dB = 20 * np.log10(fftn_q_abs/ref)

plt.figure(5)
plt.plot(freq,sn_cuant_fft_dB,":.", color="teal",label="Sn_q")
plt.plot(freq,norm_fft_dB,":.", color="chartreuse")
plt.axhline(np.mean(norm_fft_dB), color="chartreuse", linestyle="--", label=f"n: {np.mean(norm_fft_dB):.2f}dB (piso analog.)")
plt.plot(freq,n_q_fft_dB,":.", color="goldenrod")
plt.axhline(np.mean(n_q_fft_dB), color="goldenrod", linestyle='--', label=f"n_q: {np.mean(n_q_fft_dB):.2f}dB (piso digital)")

plt.legend()
plt.title(f"FFT de Señal muestreada por un ADC de {bits} bits, ±$V_r$ = {Vfs}V y q = {q:.2e}V")
plt.ylabel("Densidad de potencia [dB]")
plt.xlabel("frecuencia [Hz]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)


# %%

plt.figure (6)
plt.hist(n_q, color="orchid")
plt.hlines(100, xmin=-q/2, xmax=q/2,color="lightblue", linestyle="--", label=f"{N}/bins")
plt.vlines(-q/2, ymin=0, ymax=100, color="lightgreen", linestyle="--", label=f"{q}/2")
plt.vlines(q/2,ymin=0, ymax=100, color="lime", linestyle="--", label=f"{q}/2")
plt.title(f"Ruido de cuantización para {bits} bits, ±$V_r$ = {Vfs}V y q = {q}V")
plt.legend()
plt.xlabel("Amplitud [V]")
plt.ylabel("Nro de muestras")

plt.show()























